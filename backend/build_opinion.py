"""BUILD OPINION — bake la RÉPARTITION d'opinion (favorable / défavorable / nuance)
par thème FEUILLE et la persiste dans `analysis/opinion.json`.

Productionise l'archi VALIDÉE par le proto (`research/opinion_proto.py`,
[[agora-opinion-target-verdict]]) : mesurer l'opinion citoyenne, ce n'est pas servir
un seul côté — c'est, pour chaque thème, dériver l'OBJET DE CLIVAGE (T2, une
proposition polaire débattable, `cleavage_system` conditionné sur le titre) puis classer la stance de chaque
claim ENVERS cette proposition (`STANCE_SYSTEM`) et agréger une répartition honnête.

Garde-fous d'honnêteté :
  - on n'émet une répartition QUE sur les thèmes assez PURS (engagement = (fav+def)/n
    ≥ MIN_ENGAGEMENT) : sinon `profil='impur'` (pas de barre, signal trop diffus) ;
  - le `profil` distingue `clivant` (opposition réelle ≥ seuil) de `consensuel`
    (large adhésion) — une consultation ouverte est consensuelle PAR CONSTRUCTION,
    le clivage vit dans une minorité de sceptiques qu'on surface au lieu de la lisser.

Artefact À PART : ce build LIT les caches claims/embeddings existants (idempotent,
zéro ré-extraction si déjà fait) mais n'écrit QUE `opinion.json` — il ne touche jamais
`analysis.json`, les citations, ni les insights.

Usage CLI :
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python -m backend.build_opinion --dataset granddebat
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Callable

from backend import analysis_store as store
from backend.analysis import (
    DEFAULT_EMBEDDER,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    ThemeNode,
    ThemeTree,
    build_theme_tree,
)
from backend.build_analysis import EXTRACT_MODEL, load_dataset
from backend.titles import title_for_node
from pipeline.cluster import mistral_client

# Modèle CHEAP (cleavage + stance, ~1 + claims/BATCH appels par feuille) — surchargeable.
MODEL = os.environ.get(
    "AGORA_OPINION_MODEL", os.environ.get("AGORA_ENRICH_MODEL", "mistral-large-latest")
)
BATCH = 10                       # claims par appel de stance
# Plafond de claims classés par feuille. Par défaut quasi-illimité : on classe la stance
# de TOUS les claims des thèmes émis (le garde-fou de pureté écarte déjà les feuilles
# diffuses). Reste surchargeable (AGORA_OPINION_CAP) pour borner le coût si besoin.
CAP = max(1, int(os.environ.get("AGORA_OPINION_CAP", "100000")))
MIN_CLAIMS = 8                   # sous ce seuil, signal trop faible → impur
MIN_ENGAGEMENT = 0.35            # garde-fou pureté : (fav+def)/n ≥ ce seuil sinon impur
OPPOSITION_CLIVANT = 0.15        # opposition ≥ ce seuil → 'clivant', sinon 'consensuel'
REP_FOR_TITLE = 8                # claims représentatifs pour le repli de titre
LLM_MAX_WORKERS = max(1, int(os.environ.get("AGORA_LLM_MAX_WORKERS", "4")))
# Seuil de FIT cible↔titre sous lequel on MARQUE la cible « peu représentative »
# (`cleavage_fit_low`). Le fit est cos(emb(proposition), emb(titre)) — voir build_opinion.
# NB : c'est un MARQUEUR (audit/affichage prudent), pas un filtre dur — on n'efface rien.
CLEAVAGE_FIT_LOW = float(os.environ.get("AGORA_CLEAVAGE_FIT_LOW", "0.75"))

ProgressFn = Callable[[str, int, int], None]


def _log(msg: str) -> None:
    print(f"[build_opinion] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Prompts — REPRIS TELS QUELS du proto validé (research/opinion_proto.py).
# --------------------------------------------------------------------------- #
def cleavage_system(title: str) -> str:
    """Prompt cleavage CONDITIONNÉ sur le TITRE du thème (v2, [[agora-opinion-target-verdict]]).

    v1 (sans titre, « la PLUS SAILLANTE ») faisait dériver la cible vers une FACETTE
    bruyante au lieu du centre du thème (ex. « Restaurer la confiance par l'écoute » →
    « cesser de mentir »). v2 = deux leviers validés (research/cleavage_v2_note.md) :
      1. CONDITIONNER sur le titre — la proposition doit capturer le sujet de CE thème ;
      2. « CENTRAL » > « saillant » — résumer le débat du thème, pas le détail le plus bruyant.
    Le titre est injecté tel quel (déjà court, neutre, dérivé des claims représentatives).
    """
    return (
        "Tu es analyste de consultations citoyennes. On te donne le TITRE d'un THÈME, ses "
        "MOTS-CLÉS et des CONTRIBUTIONS verbatim. Identifie l'OBJET DE CLIVAGE qui RÉSUME "
        f"le débat CENTRAL de CE thème, intitulé « {title} » : la proposition ou mesure "
        "PRÉCISE, au cœur du thème, sur laquelle des citoyens peuvent être POUR ou CONTRE. "
        "Elle doit capturer le SUJET CENTRAL du thème (ce dont parle le titre), PAS une "
        "facette secondaire ni le détail le plus bruyant. Formule-la comme une proposition "
        "polaire COURTE (≤12 mots), neutre et débattable, à l'infinitif ou nominale — ex. "
        "« instaurer le référendum d'initiative citoyenne », « rendre le vote obligatoire », "
        "« réduire le nombre d'élus », « tirer au sort des citoyens pour légiférer ». "
        "Réponds en JSON strict : {\"objet\":\"<proposition>\",\"justif\":\"<≤14 mots>\"}."
    )

STANCE_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne UNE CIBLE — une PROPOSITION "
    "D'ACTION débattable (p. ex. « réguler l'usage d'un service », « instaurer une mesure ») "
    "— et des CONTRIBUTIONS citoyennes verbatim. Pour chaque contribution, classe si son "
    "auteur SOUTIENT ou S'OPPOSE À CETTE ACTION (et NON son sentiment envers le sujet) :\n"
    "  - \"favorable\"   : la contribution VA DANS LE SENS de l'action — elle la réclame, OU "
    "elle décrit un PROBLÈME/méfait que cette action viserait à corriger (décrire les dangers "
    "d'un sujet = soutenir une action pour le réguler/limiter) ;\n"
    "  - \"defavorable\" : la contribution S'OPPOSE à l'action — elle défend le sujet tel quel, "
    "juge l'action inutile/excessive/nuisible, ou refuse toute intervention ;\n"
    "  - \"nuance\"      : position ambivalente/conditionnelle, ou aucune position claire sur "
    "l'ACTION elle-même.\n"
    "ATTENTION — le piège à éviter : ne confonds JAMAIS un sentiment négatif ENVERS LE SUJET "
    "avec une opposition à l'action. Quelqu'un qui critique ou subit un problème est FAVORABLE "
    "à une action qui vise à le corriger. Juge la position sur l'ACTION, pas la tonalité.\n"
    "Pour CHAQUE contribution, indique aussi ta CONFIANCE : \"high\" (position explicite et "
    "nette), \"medium\" (probable mais indirecte), \"low\" (ambigu/hors-sujet — tu hésites). "
    "Réponds en JSON strict : {\"results\":[{\"i\":<int>,\"stance\":\"favorable|defavorable|"
    "nuance\",\"confidence\":\"high|medium|low\",\"justif\":\"<≤14 mots>\"}]}. Une entrée par "
    "contribution, dans l'ordre, rien d'autre."
)

# Niveaux de confiance valides (auto-évaluation du modèle). Toute valeur absente/inconnue
# est normalisée en repli prudent `low` (on n'invente pas de certitude).
CONFIDENCE_LEVELS = {"high", "medium", "low"}


def _norm_confidence(value) -> str:
    c = str(value or "").strip().lower()
    return c if c in CONFIDENCE_LEVELS else "low"


# --------------------------------------------------------------------------- #
# Cleavage T2 — objet de clivage dérivé (1 appel LLM par feuille).
# --------------------------------------------------------------------------- #
def derive_cleavage(node: ThemeNode, sample_texts: list[str], title: str,
                    *, model: str) -> dict:
    kw = ", ".join((node.keywords or [])[:10])
    contribs = "\n".join(f"- {t[:160]}" for t in sample_texts[:14])
    user = f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    messages = [{"role": "system", "content": cleavage_system(title)},
                {"role": "user", "content": user}]
    fallback = title or node.title or node.label
    try:
        raw = mistral_client.chat(messages, model=model, temperature=0.0,
                                  max_tokens=200, json_mode=True)
        data = json.loads(raw)
        objet = str(data.get("objet", "")).strip()
        return {"objet": objet or fallback,
                "justif": str(data.get("justif", "")).strip()}
    except (mistral_client.MistralError, json.JSONDecodeError):
        return {"objet": fallback, "justif": "(repli label)"}


# Synthèse PARENT — condense les objets de clivage des sous-thèmes en UNE phrase.
CLEAVAGE_SYNTH_SYSTEM = (
    "Tu reçois le TITRE d'un thème et les objets de clivage (propositions polaires) de ses "
    "sous-thèmes. Formule en UNE SEULE PHRASE l'objet de clivage GLOBAL du thème : la tension "
    "centrale qui synthétise ces sous-objets, écrite comme une proposition débattable (on peut "
    "être POUR ou CONTRE). Pas de préambule ni de liste — une seule phrase.\n"
    'Réponds en JSON : {"objet":"<une phrase>"}'
)


def synthesize_cleavage(propositions: list[str], title: str, *, model: str) -> str:
    """Synthétise les objets de clivage d'enfants en UNE phrase = objet de clivage du parent.

    0 objet → le titre ; 1 objet → tel quel ; sinon un appel LLM. Repli gracieux = titre.
    """
    props = [p for p in propositions if p]
    if not props:
        return title
    if len(props) == 1:
        return props[0]
    user = (f"TITRE : {title}\n\nOBJETS DE CLIVAGE DES SOUS-THÈMES :\n"
            + "\n".join(f"- {p}" for p in props[:20]))
    messages = [{"role": "system", "content": CLEAVAGE_SYNTH_SYSTEM},
                {"role": "user", "content": user}]
    try:
        raw = mistral_client.chat(messages, model=model, temperature=0.0,
                                  max_tokens=120, json_mode=True)
        objet = str(json.loads(raw).get("objet", "")).strip()
        return objet or title
    except (mistral_client.MistralError, json.JSONDecodeError):
        return title


# --------------------------------------------------------------------------- #
# Stance — classe chaque claim envers la cible (batché, repli unitaire).
# --------------------------------------------------------------------------- #
def stance_batch(cible: str, items: list[tuple[int, str]], *, model: str) -> dict[int, dict]:
    lines = [f"[{i}] {text}" for i, text in items]
    user = (f"CIBLE : {cible}\n\n"
            f"CONTRIBUTIONS (réponds pour chaque [indice]) :\n" + "\n".join(lines))
    messages = [{"role": "system", "content": STANCE_SYSTEM},
                {"role": "user", "content": user}]
    raw = mistral_client.chat(messages, model=model, temperature=0.0,
                              max_tokens=1500, json_mode=True)
    data = json.loads(raw)
    out: dict[int, dict] = {}
    for rec in data.get("results", []):
        try:
            idx = int(rec["i"])
        except (KeyError, ValueError, TypeError):
            continue
        stance = str(rec.get("stance", "")).strip().lower()
        if stance not in {"favorable", "defavorable", "nuance"}:
            stance = "nuance"
        out[idx] = {"stance": stance,
                    "confidence": _norm_confidence(rec.get("confidence")),
                    "justif": str(rec.get("justif", "")).strip()}
    return out


def run_stance(cible: str, items: list[tuple[int, str]], *, model: str) -> dict[int, dict]:
    results: dict[int, dict] = {}
    for start in range(0, len(items), BATCH):
        batch = items[start:start + BATCH]
        try:
            got = stance_batch(cible, batch, model=model)
        except (mistral_client.MistralError, json.JSONDecodeError):
            got = {}
        for i, text in batch:
            if i not in got:
                try:
                    got.update(stance_batch(cible, [(i, text)], model=model))
                except (mistral_client.MistralError, json.JSONDecodeError):
                    got[i] = {"stance": "nuance", "confidence": "low",
                              "justif": "(échec LLM)"}
        results.update(got)
        time.sleep(0.02)
    return results


# --------------------------------------------------------------------------- #
# Agrégation — répartition + profil clivant/consensuel/impur.
# --------------------------------------------------------------------------- #
def aggregate(theme_id: str, proposition: str, counts: Counter, n: int) -> dict:
    """Répartition d'opinion d'un thème, avec garde-fou de pureté.

    `engagement = (fav+def)/n` mesure si la cible fait prendre position (1 − %nuance) ;
    `opposition = min(fav,def)/(fav+def)` révèle le clivage réel ; `pct_favorable` = la
    part favorable PARMI LES ENGAGÉS. Profil : 'impur' si le signal est trop diffus
    (engagement faible / trop peu de claims), sinon 'clivant' vs 'consensuel'.
    """
    fav = counts.get("favorable", 0)
    dfv = counts.get("defavorable", 0)
    nu = counts.get("nuance", 0)
    pol = fav + dfv
    engagement = pol / n if n else 0.0
    opposition = min(fav, dfv) / pol if pol else 0.0
    pct_favorable = fav / pol if pol else 0.0
    if n < MIN_CLAIMS or engagement < MIN_ENGAGEMENT:
        profil = "impur"
    elif opposition >= OPPOSITION_CLIVANT:
        profil = "clivant"
    else:
        profil = "consensuel"
    return {
        "theme_id": theme_id,
        "proposition": proposition,
        "fav": fav,
        "def": dfv,
        "nuance": nu,
        "n": n,
        "engagement": round(engagement, 3),
        "opposition": round(opposition, 3),
        "pct_favorable": round(pct_favorable, 3),
        "profil": profil,
    }


def _attach_cleavage_fit(opinions: list[dict], *, embedder: str = DEFAULT_EMBEDDER) -> None:
    """Ajoute `cleavage_fit` (cos cible↔titre) + `cleavage_fit_low` à chaque opinion, EN PLACE.

    Mesure de représentativité de la cible : embedde proposition ET titre avec l'encodeur
    PROD (nomic-v2, même espace que les claims), puis cosinus. Un fit bas = la proposition
    dérivée s'écarte du sujet déclaré du thème (cible peu représentative). Validé contre le
    cos-vs-centroïde, trompeur (research/cleavage_v2_note.md). Repli gracieux : si l'embed
    échoue (pas d'extra embed), on laisse `cleavage_fit=None` sans marquer (ne lève jamais).
    """
    if not opinions:
        return
    try:
        from pipeline.claims.pipeline import embed_claim_texts
        import numpy as np
        props = [o.get("proposition", "") or "" for o in opinions]
        titles = [o.get("title", "") or "" for o in opinions]
        pv = embed_claim_texts(props, embedder=embedder)
        tv = embed_claim_texts(titles, embedder=embedder)
        for o, p, t in zip(opinions, pv, tv):
            fit = max(0.0, float(np.dot(p, t)))   # vecteurs déjà L2-normalisés
            o["cleavage_fit"] = round(fit, 4)
            o["cleavage_fit_low"] = bool(fit < CLEAVAGE_FIT_LOW)
    except Exception as exc:  # embed indisponible / erreur torch — diagnostic facultatif
        _log(f"cleavage_fit indisponible ({type(exc).__name__}) — fit non calculé")
        for o in opinions:
            o.setdefault("cleavage_fit", None)
            o.setdefault("cleavage_fit_low", False)


def _leaf_claims(node: ThemeNode, prepared) -> list[tuple[int, str, str]]:
    """(claim_index, avis_id, claim_text) verbatim pour les claims du nœud feuille.

    `claim_index` est l'index GLOBAL du claim (même clé que `/avis` : le claim servi a
    pour id `f"{avis_id}#{claim_index}"`), conservé pour ancrer la stance par claim.
    Le texte est `text_clean` (PII masquée, mêmes offsets que les spans servis).
    """
    out: list[tuple[int, str, str]] = []
    for i in node.members:
        t = (prepared.claim_texts[i] or "").strip()
        if len(t) >= 12:
            aid = prepared.avis[prepared.claim_owner[i]].id
            out.append((i, aid, t))
    return out


def analyse_leaf(node: ThemeNode, tree: ThemeTree, rng: random.Random, dataset: str,
                 *, model: str) -> tuple[dict, dict[str, dict]]:
    """Dérive la cible T2 d'une feuille, classe les claims, agrège la répartition.

    Renvoie `(opinion, claim_stance)` où `claim_stance` mappe le claim_id servi par
    `/avis` (`f"{avis_id}#{index}"`) → `{stance, justif, proposition, theme_id}`. Par
    cohérence avec le garde-fou de pureté (on ne montre une répartition que sur les
    thèmes assez purs), un thème `impur` n'émet AUCUNE stance par claim (map vide).
    """
    if not node.representative_claims:  # repli si le build d'analyse n'a pas titré ce nœud
        reps = [tree.prepared.claim_texts[i] for i in node.members[:REP_FOR_TITLE]]
        node.representative_claims = [r[:240] for r in reps]
    # TITRE court du thème — CONDITIONNE le cleavage v2. Caché par contenu (titres/) : si
    # le build d'analyse l'a déjà nommé, c'est un cache HIT (zéro LLM) ; sinon repli label.
    title = title_for_node(dataset, node) or node.title or node.label

    cl = _leaf_claims(node, tree.prepared)
    if len(cl) > CAP:
        cl = [cl[i] for i in sorted(rng.sample(range(len(cl)), CAP))]
    sample = [(j, txt) for j, (_gi, _aid, txt) in enumerate(cl)]

    cleavage = derive_cleavage(node, [t for _, t in sample], title, model=model)
    proposition = cleavage["objet"]

    st = run_stance(proposition, sample, model=model)
    counts = Counter(st[j]["stance"] for j, _ in sample if j in st)
    opinion = aggregate(node.id, proposition, counts, len(sample))
    opinion["title"] = title
    opinion["cleavage_justif"] = cleavage.get("justif", "")

    # Stance PAR CLAIM — clé = id servi par `/avis`. Émise seulement si le thème est pur
    # (sinon le signal est trop diffus pour être affiché/audité par claim).
    claim_stance: dict[str, dict] = {}
    if opinion["profil"] != "impur":
        for j, (gi, aid, _txt) in enumerate(cl):
            rec = st.get(j)
            if not rec:
                continue
            claim_stance[f"{aid}#{gi}"] = {
                "stance": rec["stance"],
                "stance_confidence": _norm_confidence(rec.get("confidence")),
                "justif": rec.get("justif", ""),
                "proposition": proposition,
                "theme_id": node.id,
            }
    return opinion, claim_stance


# --------------------------------------------------------------------------- #
def build_opinion(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    extract_model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Construit l'arbre (mêmes paramètres que `build_analysis` → mêmes theme_id), dérive
    l'opinion par FEUILLE, persiste `opinion.json`, renvoie le payload.

    L'arbre est rebâti en mémoire à partir des caches claims/embeddings existants (zéro
    ré-extraction si déjà fait). On ne traite QUE les feuilles (1 feuille ≈ 1 proposition) ;
    une feuille avec trop peu de claims sort en 'impur' sans répartition.

    `model` = modèle CLEAVAGE+STANCE (cheap, défaut `MODEL`). `extract_model` = modèle
    d'EXTRACTION de l'arbre : il doit matcher celui de `build_analysis` (`EXTRACT_MODEL`
    par défaut) sinon la clé de cache claims diffère et l'arbre RÉ-EXTRAIT. On le forwarde
    donc explicitement à `build_theme_tree` → réutilise l'extraction déjà cachée.
    """
    t0 = perf_counter()
    dataset = ds.id
    model = model or MODEL
    extract_model = extract_model or EXTRACT_MODEL
    rng = random.Random(seed)
    mistral_client.reset_usage()  # suivi tokens/coût de la phase opinion (cleavage + stance)

    _log(f"{dataset} · construction de l'arbre (caché si déjà extrait)…")
    tree = build_theme_tree(ds, backend=backend, model=extract_model, embedder=embedder,
                            resolution=resolution, seed=seed)

    leaves = [tree.nodes[nid] for nid in tree.order if not tree.nodes[nid].children]
    total = len(leaves)
    _log(f"{dataset} · {total} feuilles à traiter (cap {CAP} claims/feuille, modèle {model})")

    done = 0
    lock = threading.Lock()
    results: list[tuple[dict, dict[str, dict]]] = []

    def _work(node: ThemeNode) -> tuple[dict, dict[str, dict]]:
        return analyse_leaf(node, tree, rng, dataset, model=model)

    def _record(_k: int) -> None:
        if on_progress:
            on_progress("opinion", _k, total)
        if _k == total or _k % 5 == 0:
            _log(f"{dataset} · opinion {_k}/{total}")

    if LLM_MAX_WORKERS <= 1 or total <= 1:
        for node in leaves:
            results.append(_work(node))
            done += 1
            _record(done)
    else:
        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS,
                                thread_name_prefix="agora-opinion") as ex:
            futures = [ex.submit(_work, node) for node in leaves]
            for fut in as_completed(futures):
                results.append(fut.result())
                with lock:
                    done += 1
                    k = done
                _record(k)

    opinions = [o for o, _cs in results]
    # Stance PAR CLAIM agrégée sur toutes les feuilles (claim_id → record), persistée à part.
    claim_stance: dict[str, dict] = {}
    for _o, cs in results:
        claim_stance.update(cs)

    # FIT cible↔titre — la cible RÉSUME-t-elle le thème ? cos(emb(proposition), emb(titre)),
    # MÊME encodeur que les claims. On a écarté le cos vs CENTROÏDE (research/cleavage_v2_note.md) :
    # le centroïde est dominé par la facette la PLUS BRUYANTE → il récompense le biais « saillant »
    # qu'on combat (sur le cas-test il classait la PIRE cible plus haut). Un seul batch d'embed,
    # hors thread (torch). Marque `cleavage_fit_low` si fit < seuil — MARQUEUR, pas filtre.
    _attach_cleavage_fit(opinions)

    # Ordre STABLE (par theme_id dans l'ordre de l'arbre) — indépendant de l'ordonnancement.
    rank = {nid: i for i, nid in enumerate(tree.order)}
    opinions.sort(key=lambda o: rank.get(o["theme_id"], 1 << 30))

    # ── Remontée aux PARENTS : pour chaque nœud non-feuille, MOYENNE PONDÉRÉE (par nombre de
    #    claims) du sentiment de ses feuilles-descendantes + SYNTHÈSE LLM de leurs objets de
    #    clivage en une phrase. Un thème parent porte ainsi le sentiment agrégé de ses enfants.
    leaf_op = {o["theme_id"]: o for o in opinions}

    def _leaf_descendants(nid: str) -> list[str]:
        node = tree.nodes[nid]
        if not node.children:
            return [nid]
        acc: list[str] = []
        for c in node.children:
            acc.extend(_leaf_descendants(c))
        return acc

    parent_ops: list[dict] = []
    for nid in tree.order:
        node = tree.nodes[nid]
        if not node.children:
            continue  # feuille : déjà traitée
        kids = [leaf_op[l] for l in _leaf_descendants(nid)
                if l in leaf_op and leaf_op[l]["profil"] != "impur"]
        if not kids:
            continue  # aucun signal exploitable sous ce parent
        fav = sum(o["fav"] for o in kids)
        dfv = sum(o["def"] for o in kids)
        nu = sum(o["nuance"] for o in kids)
        ptitle = node.title or node.label
        child_props = [o["proposition"] for o in kids if o.get("proposition")]
        agg = aggregate(nid, synthesize_cleavage(child_props, ptitle, model=model),
                        Counter({"favorable": fav, "defavorable": dfv, "nuance": nu}),
                        fav + dfv + nu)
        agg["title"] = ptitle
        agg["is_aggregate"] = True
        agg["n_children"] = len(kids)
        agg["child_propositions"] = child_props[:20]
        parent_ops.append(agg)

    _log(f"{dataset} · {len(parent_ops)} thèmes parents agrégés (moyenne pondérée + objet synthétisé)")
    opinions = opinions + parent_ops
    opinions.sort(key=lambda o: rank.get(o["theme_id"], 1 << 30))

    n_clivant = sum(1 for o in opinions if o["profil"] == "clivant")
    n_consensuel = sum(1 for o in opinions if o["profil"] == "consensuel")
    n_impur = sum(1 for o in opinions if o["profil"] == "impur")
    took_s = round(perf_counter() - t0, 1)

    payload = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "cap_claims_per_leaf": CAP,
        "thresholds": {
            "min_engagement": MIN_ENGAGEMENT,
            "min_claims": MIN_CLAIMS,
            "opposition_clivant": OPPOSITION_CLIVANT,
        },
        "cleavage_prompt_system": cleavage_system("<TITRE>"),
        "stance_prompt_system": STANCE_SYSTEM,
        "cleavage_fit_low_threshold": CLEAVAGE_FIT_LOW,
        "counts": {"clivant": n_clivant, "consensuel": n_consensuel, "impur": n_impur,
                   "fit_low": sum(1 for o in opinions if o.get("cleavage_fit_low"))},
        "n_leaves": total,
        "took_seconds": took_s,
        "themes": opinions,
    }
    store.write_opinion(dataset, payload)
    store.write_claim_stance(dataset, claim_stance)
    # Coût LLM de la phase opinion (cleavage + stance + synthèses parents) — jamais bloquant.
    try:
        from backend import cost as _cost
        _cost.record_phase(dataset, "opinion", mistral_client.get_usage(), duration_seconds=took_s)
    except Exception as _e:
        _log(f"{dataset} · (coût opinion non enregistré: {_e})")
    _log(f"{dataset} · ✓ opinion.json écrit · {total} feuilles "
         f"({n_clivant} clivant / {n_consensuel} consensuel / {n_impur} impur) · {took_s}s")
    _log(f"{dataset} · ✓ claim_stance.json écrit · {len(claim_stance)} claims classés "
         f"(thèmes purs uniquement)")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bake la répartition d'opinion par thème feuille (objet de clivage T2 + stance).")
    ap.add_argument("--dataset", required=True, help="id du dataset (sous backend/cache/)")
    ap.add_argument("--backend", default=None, help="api (défaut) | mac | auto")
    ap.add_argument("--model", default=None, help=f"modèle cleavage+stance (défaut {MODEL})")
    ap.add_argument("--extract-model", default=None,
                    help=f"modèle d'extraction de l'arbre (défaut {EXTRACT_MODEL} — doit "
                         f"matcher build_analysis pour réutiliser le cache claims)")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")

    ds = load_dataset(args.dataset)
    build_opinion(ds, backend=args.backend, model=args.model,
                  extract_model=args.extract_model, embedder=args.embedder,
                  resolution=args.resolution, seed=args.seed)


if __name__ == "__main__":
    main()

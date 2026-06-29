"""BUILD OPINION — bake la RÉPARTITION d'opinion (favorable / défavorable / nuance)
par thème FEUILLE et la persiste dans `analysis/opinion.json`.

Productionise l'archi VALIDÉE par le proto (`research/opinion_proto.py`,
[[agora-opinion-target-verdict]]) : mesurer l'opinion citoyenne, ce n'est pas servir
un seul côté — c'est, pour chaque thème, dériver l'OBJET DE CLIVAGE (T2, une
proposition polaire débattable, `CLEAVAGE_SYSTEM`) puis classer la stance de chaque
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
from backend.analysis import DEFAULT_EMBEDDER, DEFAULT_SEED, ThemeNode, ThemeTree, build_theme_tree
from backend.build_analysis import load_dataset
from pipeline.cluster import mistral_client

# Modèle CHEAP (cleavage + stance, ~1 + claims/BATCH appels par feuille) — surchargeable.
MODEL = os.environ.get(
    "AGORA_OPINION_MODEL", os.environ.get("AGORA_ENRICH_MODEL", "mistral-small-latest")
)
BATCH = 10                       # claims par appel de stance
# Plafond de claims classés par feuille (borne le coût ; échantillon déterministe par seed).
CAP = max(1, int(os.environ.get("AGORA_OPINION_CAP", "150")))
MIN_CLAIMS = 8                   # sous ce seuil, signal trop faible → impur
MIN_ENGAGEMENT = 0.35            # garde-fou pureté : (fav+def)/n ≥ ce seuil sinon impur
OPPOSITION_CLIVANT = 0.15        # opposition ≥ ce seuil → 'clivant', sinon 'consensuel'
REP_FOR_TITLE = 8                # claims représentatifs pour le repli de titre
LLM_MAX_WORKERS = max(1, int(os.environ.get("AGORA_LLM_MAX_WORKERS", "4")))

ProgressFn = Callable[[str, int, int], None]


def _log(msg: str) -> None:
    print(f"[build_opinion] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Prompts — REPRIS TELS QUELS du proto validé (research/opinion_proto.py).
# --------------------------------------------------------------------------- #
CLEAVAGE_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne les MOTS-CLÉS et des "
    "CONTRIBUTIONS verbatim d'un THÈME. Identifie l'OBJET DE CLIVAGE central : la "
    "proposition ou mesure PRÉCISE sur laquelle des citoyens peuvent être POUR ou "
    "CONTRE. Formule-la comme une proposition polaire COURTE (≤12 mots), neutre et "
    "débattable, à l'infinitif ou nominale — ex. « instaurer le référendum d'initiative "
    "citoyenne », « rendre le vote obligatoire », « réduire le nombre d'élus », "
    "« tirer au sort des citoyens pour légiférer ». Si le thème mêle plusieurs "
    "propositions, choisis LA PLUS SAILLANTE. Réponds en JSON strict : "
    "{\"objet\":\"<proposition>\",\"justif\":\"<≤14 mots>\"}."
)

STANCE_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne UNE CIBLE (un objet de "
    "débat) et des CONTRIBUTIONS citoyennes verbatim. Pour chaque contribution, classe "
    "la PRISE DE POSITION de l'auteur ENVERS LA CIBLE en exactement une étiquette :\n"
    "  - \"favorable\"   : la contribution soutient, défend, réclame ou valorise la "
    "cible (elle est POUR) ;\n"
    "  - \"defavorable\" : la contribution rejette, critique, conteste ou veut empêcher "
    "la cible (elle est CONTRE) ;\n"
    "  - \"nuance\"      : position ambivalente, conditionnelle, hors-sujet, ou aucune "
    "position claire ENVERS LA CIBLE précise.\n"
    "Juge la position envers la CIBLE, pas la qualité de l'écriture. Si la contribution "
    "ne parle pas de la cible, c'est \"nuance\". Réponds en JSON strict : "
    "{\"results\":[{\"i\":<int>,\"stance\":\"favorable|defavorable|nuance\",\"justif\":"
    "\"<≤14 mots>\"}]}. Une entrée par contribution, dans l'ordre, rien d'autre."
)


# --------------------------------------------------------------------------- #
# Cleavage T2 — objet de clivage dérivé (1 appel LLM par feuille).
# --------------------------------------------------------------------------- #
def derive_cleavage(node: ThemeNode, sample_texts: list[str], *, model: str) -> dict:
    kw = ", ".join((node.keywords or [])[:10])
    contribs = "\n".join(f"- {t[:160]}" for t in sample_texts[:14])
    user = f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    messages = [{"role": "system", "content": CLEAVAGE_SYSTEM},
                {"role": "user", "content": user}]
    fallback = node.title or node.label
    try:
        raw = mistral_client.chat(messages, model=model, temperature=0.0,
                                  max_tokens=200, json_mode=True)
        data = json.loads(raw)
        objet = str(data.get("objet", "")).strip()
        return {"objet": objet or fallback,
                "justif": str(data.get("justif", "")).strip()}
    except (mistral_client.MistralError, json.JSONDecodeError):
        return {"objet": fallback, "justif": "(repli label)"}


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
        out[idx] = {"stance": stance, "justif": str(rec.get("justif", "")).strip()}
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
                    got[i] = {"stance": "nuance", "justif": "(échec LLM)"}
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


def _leaf_claims(node: ThemeNode, prepared) -> list[tuple[str, str]]:
    """(avis_id, claim_text) verbatim pour les claims du nœud feuille (text_clean ancré)."""
    out: list[tuple[str, str]] = []
    for i in node.members:
        t = (prepared.claim_texts[i] or "").strip()
        if len(t) >= 12:
            aid = prepared.avis[prepared.claim_owner[i]].id
            out.append((aid, t))
    return out


def analyse_leaf(node: ThemeNode, tree: ThemeTree, rng: random.Random, *, model: str) -> dict:
    """Dérive la cible T2 d'une feuille, classe les claims, agrège la répartition."""
    if not node.title:  # repli si le build d'analyse n'a pas (encore) titré ce nœud
        reps = [tree.prepared.claim_texts[i] for i in node.members[:REP_FOR_TITLE]]
        node.representative_claims = node.representative_claims or [r[:240] for r in reps]

    cl = _leaf_claims(node, tree.prepared)
    if len(cl) > CAP:
        cl = [cl[i] for i in sorted(rng.sample(range(len(cl)), CAP))]
    sample = [(j, txt) for j, (_aid, txt) in enumerate(cl)]

    cleavage = derive_cleavage(node, [t for _, t in sample], model=model)
    proposition = cleavage["objet"]

    st = run_stance(proposition, sample, model=model)
    counts = Counter(st[j]["stance"] for j, _ in sample if j in st)
    opinion = aggregate(node.id, proposition, counts, len(sample))
    opinion["title"] = node.title or node.label
    opinion["cleavage_justif"] = cleavage.get("justif", "")
    return opinion


# --------------------------------------------------------------------------- #
def build_opinion(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Construit l'arbre (mêmes paramètres que `build_analysis` → mêmes theme_id), dérive
    l'opinion par FEUILLE, persiste `opinion.json`, renvoie le payload.

    L'arbre est rebâti en mémoire à partir des caches claims/embeddings existants (zéro
    ré-extraction si déjà fait). On ne traite QUE les feuilles (1 feuille ≈ 1 proposition) ;
    une feuille avec trop peu de claims sort en 'impur' sans répartition.
    """
    t0 = perf_counter()
    dataset = ds.id
    model = model or MODEL
    rng = random.Random(seed)

    _log(f"{dataset} · construction de l'arbre (caché si déjà extrait)…")
    tree = build_theme_tree(ds, backend=backend, embedder=embedder,
                            resolution=resolution, seed=seed)

    leaves = [tree.nodes[nid] for nid in tree.order if not tree.nodes[nid].children]
    total = len(leaves)
    _log(f"{dataset} · {total} feuilles à traiter (cap {CAP} claims/feuille, modèle {model})")

    done = 0
    lock = threading.Lock()
    opinions: list[dict] = []

    def _work(node: ThemeNode) -> dict:
        return analyse_leaf(node, tree, rng, model=model)

    def _record(_k: int) -> None:
        if on_progress:
            on_progress("opinion", _k, total)
        if _k == total or _k % 5 == 0:
            _log(f"{dataset} · opinion {_k}/{total}")

    if LLM_MAX_WORKERS <= 1 or total <= 1:
        for node in leaves:
            opinions.append(_work(node))
            done += 1
            _record(done)
    else:
        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS,
                                thread_name_prefix="agora-opinion") as ex:
            futures = [ex.submit(_work, node) for node in leaves]
            for fut in as_completed(futures):
                opinions.append(fut.result())
                with lock:
                    done += 1
                    k = done
                _record(k)

    # Ordre STABLE (par theme_id dans l'ordre de l'arbre) — indépendant de l'ordonnancement.
    rank = {nid: i for i, nid in enumerate(tree.order)}
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
        "cleavage_prompt_system": CLEAVAGE_SYSTEM,
        "stance_prompt_system": STANCE_SYSTEM,
        "counts": {"clivant": n_clivant, "consensuel": n_consensuel, "impur": n_impur},
        "n_leaves": total,
        "took_seconds": took_s,
        "themes": opinions,
    }
    store.write_opinion(dataset, payload)
    _log(f"{dataset} · ✓ opinion.json écrit · {total} feuilles "
         f"({n_clivant} clivant / {n_consensuel} consensuel / {n_impur} impur) · {took_s}s")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bake la répartition d'opinion par thème feuille (objet de clivage T2 + stance).")
    ap.add_argument("--dataset", required=True, help="id du dataset (sous backend/cache/)")
    ap.add_argument("--backend", default=None, help="api (défaut) | mac | auto")
    ap.add_argument("--model", default=None, help=f"modèle cleavage+stance (défaut {MODEL})")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")

    ds = load_dataset(args.dataset)
    build_opinion(ds, backend=args.backend, model=args.model, embedder=args.embedder,
                  resolution=args.resolution, seed=args.seed)


if __name__ == "__main__":
    main()

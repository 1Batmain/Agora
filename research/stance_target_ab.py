"""R&D CIBLE DE STANCE a/b/c — la cible de clivage est-elle trop GÉNÉRIQUE, et peut-on
la rendre DISTINCTIVE sans casser le garde-fou « central > saillant » du cleavage-v2 ?

Symptôme (Bob) : les objets de clivage servis sont quasi-identiques d'un cluster à
l'autre (« réguler X » partout). Cause suspectée = la MÊME que pour les titres
génériques : anisotropie de l'embedding sur un corpus mono-sujet → les claims d'entrée
(médoïdes centraux, cf. `backend/develop.py` : le pur médoïde surface le COURT et
GÉNÉRIQUE) et donc la cible dérivée sont plats. Repro anisotropie : `research/cluster_merge_note.md`.

TENSION documentée à ne PAS piétiner : le cleavage-v2 (`research/cleavage_v2_note.md`,
docstrings `backend/build_opinion.py`) a REJETÉ la dérive vers le SAILLANT. Il conditionne
sur le titre + exige le CENTRAL. Rendre la cible « distinctive » = pousser vers le saillant
= risque de rejouer exactement ce que v2 a écarté. On MESURE ce compromis, on ne tranche pas.

Trois variantes de dérivation de la cible, par feuille (mêmes claims-source pour a et b) :
  (a) ACTUEL — échantillon CENTRAL (top cos↔centroïde) + prompt v2 de production
      (`cleavage_system`) + mots-clés c-TF-IDF « bruts ». Reproduit `build_opinion` fidèlement.
  (b) CENTRAL + MOTS-CLÉS DISTINCTIFS EN CONTEXTE — même échantillon central, mais les
      mots-clés c-TF-IDF sont présentés comme CE QUI DISTINGUE ce thème des voisins, et le
      prompt reçoit une consigne de contraste. Change UNIQUEMENT le cadrage du prompt.
  (c) CLAIMS DISTINCTIFS — même prompt que (a), mais l'échantillon-source est sélectionné
      par DISTINCTIVITÉ PURE (claims les plus riches en vocabulaire propre au cluster).
      Change UNIQUEMENT l'INPUT. C'est la variante la plus exposée au risque « saillant ».

Mesures (SANS re-bake, SANS toucher le pipeline) :
  - les 3 cibles côte à côte par feuille ;
  - `fit_titre` = cos(emb(cible), emb(titre)) — la métrique de représentativité ADOPTÉE
    par v2 (fit-centroïde rejeté, cf. note). Un fit_titre qui CHUTE en (b)/(c) = la cible
    s'éloigne du sujet déclaré = matérialisation du risque « saillant » de v2 ;
  - `fit_centroide` = cos(emb(cible), centroïde) — REFERENCE (métrique rejetée, pour info) ;
  - DISTINCTIVITÉ INTER-CIBLES : cos moyen entre les N cibles d'une même variante. BAS =
    cibles variées/distinctes ; HAUT = quasi-identiques (le symptôme de Bob). C'est LE test.

Sortie :
  - `research/stance_target_ab_results.json` (rows + summary) ;
  - `research/stance_target_ab_panel.jsonl` — PANEL AVEUGLE : paires anonymisées (X/Y) à
    juger, sans l'identité de variante, + contexte NEUTRE (titre + claims représentatives) ;
  - `research/stance_target_ab_panel_key.json` — la clé pair_id → {X,Y}=variante (HORS du
    fichier aveugle). Le juge est lancé SÉPARÉMENT.

Budget : ~30 appels Mistral (3/feuille × 10 feuilles), modèle cheap. Titres = cache HIT (0 LLM).

Lancement (racine du worktree) :
  MISTRAL_API_KEY=... PYTHONPATH=. \
  uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
  python research/stance_target_ab.py [--dry] [--dataset tiktok] [--leaves 10]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

from backend.analysis import DEFAULT_EMBEDDER, DEFAULT_SEED, build_theme_tree
from backend.build_analysis import EXTRACT_MODEL, load_dataset
from backend.build_opinion import cleavage_system  # prompt v2 de PRODUCTION (variante a)
from backend.develop import corpus_idf
from backend.titles import title_for_node
from pipeline.claims.pipeline import embed_claim_texts
from pipeline.cluster import mistral_client
from pipeline.cluster.naming import _tokenizer

# select_distinctive_claims est en cours de création par la lane titres-ancres
# (backend/develop.py). S'il atterrit, on le préférera ; ici il n'existe pas encore, donc
# on utilise le sélecteur distinctif LOCAL ci-dessous. On NE duplique PAS un helper backend :
# ceci est un sélecteur de recherche, à remplacer par le canonique quand son API est figée.
try:  # pragma: no cover - dépend d'une lane concurrente
    from backend.develop import select_distinctive_claims as _CANONICAL_DISTINCTIVE
except Exception:
    _CANONICAL_DISTINCTIVE = None

SEED = DEFAULT_SEED
RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = RESEARCH_DIR / "stance_target_ab_results.json"
PANEL_PATH = RESEARCH_DIR / "stance_target_ab_panel.jsonl"
PANEL_KEY_PATH = RESEARCH_DIR / "stance_target_ab_panel_key.json"
MODEL = os.environ.get("AGORA_OPINION_MODEL", "mistral-small-latest")

MIN_CLAIMS = 12          # feuilles trop petites = signal trop faible
SAMPLE_FOR_PROMPT = 14   # contributions montrées au prompt cleavage (comme build_opinion)
CAP = 60                 # claims échantillonnés / feuille avant sélection central/distinctif
TOP_KEYWORDS = 10        # mots-clés injectés (comme build_opinion : node.keywords[:10])
REP_FOR_TITLE = 8
DISTINCTIVE_VOCAB = 40   # taille du vocabulaire distinctif par cluster (variante c)
REP_FOR_PANEL = 3        # claims représentatives montrées au juge (contexte neutre)

# Question posée au PANEL AVEUGLE (le juge est lancé séparément) — stockée dans le fichier.
PANEL_QUESTION = (
    "Voici le TITRE d'un thème citoyen et quelques contributions. Deux propositions "
    "d'« objet de clivage » (X et Y) tentent de résumer la tension débattable AU CŒUR de "
    "ce thème. Laquelle capture le mieux le sujet CENTRAL du thème (pas une facette "
    "secondaire ni le détail le plus bruyant), tout en restant une proposition polaire "
    "claire sur laquelle on peut être POUR ou CONTRE ? Réponds \"X\", \"Y\" ou \"nul\" "
    "(équivalentes)."
)

# Suffixe de CONTRASTE ajouté au prompt v2 pour la variante (b) — pousse la cible à
# refléter ce qui DISTINGUE ce thème de ses voisins, sans abandonner l'exigence « central ».
CONTRAST_SUFFIX = (
    " Les MOTS-CLÉS fournis sont ce qui DISTINGUE ce thème des thèmes voisins de la même "
    "consultation : ta proposition doit refléter cette SPÉCIFICITÉ (éviter une formulation "
    "passe-partout qui conviendrait à n'importe quel thème), tout en restant CENTRALE au thème."
)


# --------------------------------------------------------------------------- #
# Sélection des claims-source : CENTRAL (a,b) vs DISTINCTIF (c).
# --------------------------------------------------------------------------- #
def _cache_model(dataset: str, default: str) -> str:
    """Modèle d'EXTRACTION épinglé depuis le cache claims → build = cache HIT, ZÉRO
    extraction LLM (fail-closed). Même pinning que `research/naming_contrastif.py`."""
    try:
        from backend.recluster import dataset_dir
        rec = json.loads((dataset_dir(dataset) / "claims.json").read_text(encoding="utf-8"))
        return rec.get("model") or default
    except Exception:
        return default


def central_order(members: list[int], vecs: np.ndarray, centroid: np.ndarray) -> list[int]:
    """Indices LOCAUX de `members` triés par centralité (cos↔centroïde) décroissante."""
    sims = vecs[members] @ centroid
    return list(np.argsort(-sims))


def _distinctive_vocab(member_texts: list[str], idf: dict[str, float],
                       corpus_df_frac: dict[str, float]) -> dict[str, float]:
    """Vocabulaire distinctif du cluster : tokens SUR-représentés ici vs corpus × idf.

    weight(t) = max(0, df_frac_cluster(t) − df_frac_corpus(t)) × idf(t). C'est la même
    logique c-TF-IDF que `pipeline.cluster.naming` (faire remonter le distinctif, écraser
    le commun), calculée localement sur le pool. Top `DISTINCTIVE_VOCAB` tokens.
    """
    n = len(member_texts)
    if not n:
        return {}
    df: dict[str, int] = {}
    for t in member_texts:
        for tok in set(_tokenizer(t)):
            df[tok] = df.get(tok, 0) + 1
    weights: dict[str, float] = {}
    for tok, c in df.items():
        over = (c / n) - corpus_df_frac.get(tok, 0.0)
        if over > 0:
            weights[tok] = over * idf.get(tok, 0.0)
    top = sorted(weights.items(), key=lambda kv: -kv[1])[:DISTINCTIVE_VOCAB]
    return dict(top)


def distinctive_order(member_texts: list[str], vocab_w: dict[str, float]) -> list[int]:
    """Indices LOCAUX triés par DISTINCTIVITÉ décroissante : densité de vocabulaire propre
    au cluster. score(claim) = Σ weight(tok) sur tokens uniques / (1 + log(#tokens)) —
    normalisé pour ne pas juste récompenser les claims longs."""
    scores = []
    for txt in member_texts:
        toks = set(_tokenizer(txt))
        if not toks:
            scores.append(0.0)
            continue
        s = sum(vocab_w.get(tok, 0.0) for tok in toks) / (1.0 + math.log(len(toks)))
        scores.append(s)
    return list(np.argsort(-np.array(scores)))


# --------------------------------------------------------------------------- #
# Dérivation de la cible (1 appel LLM) — user-message IDENTIQUE à build_opinion.
# --------------------------------------------------------------------------- #
def derive_target(system: str, keywords_label: str, keywords: list[str],
                  sample_texts: list[str], fallback: str) -> dict:
    kw = ", ".join((keywords or [])[:TOP_KEYWORDS])
    contribs = "\n".join(f"- {t[:160]}" for t in sample_texts[:SAMPLE_FOR_PROMPT])
    user = f"{keywords_label} : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        raw = mistral_client.chat(messages, model=MODEL, temperature=0.0,
                                  max_tokens=200, json_mode=True)
        data = json.loads(raw)
        objet = str(data.get("objet", "")).strip()
        return {"objet": objet or fallback, "justif": str(data.get("justif", "")).strip()}
    except (mistral_client.MistralError, json.JSONDecodeError):
        return {"objet": fallback, "justif": "(repli)"}


# --------------------------------------------------------------------------- #
def _leaf_members_texts(node, prepared) -> tuple[list[int], list[str]]:
    """(members filtrés ≥12 car, textes alignés) pour une feuille."""
    mem, txt = [], []
    for i in node.members:
        t = (prepared.claim_texts[i] or "").strip()
        if len(t) >= 12:
            mem.append(i)
            txt.append(t)
    return mem, txt


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _mean_pairwise_cos(mat: np.ndarray) -> float:
    """cos moyen entre toutes les paires de lignes (vecteurs L2-normalisés)."""
    n = len(mat)
    if n < 2:
        return float("nan")
    g = mat @ mat.T
    iu = np.triu_indices(n, k=1)
    return float(g[iu].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tiktok")
    ap.add_argument("--leaves", type=int, default=10)
    ap.add_argument("--dry", action="store_true",
                    help="construit l'arbre, choisit les feuilles, montre les échantillons "
                         "central/distinctif — ZÉRO appel LLM de cleavage.")
    args = ap.parse_args()

    if not args.dry and not mistral_client.available():
        sys.exit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")
    if _CANONICAL_DISTINCTIVE is not None:
        print("ℹ backend.develop.select_distinctive_claims DISPONIBLE — le sélecteur local "
              "reste utilisé (API canonique non figée) ; à réconcilier.", flush=True)

    rng = random.Random(SEED)
    print(f"Construction de l'arbre {args.dataset} (caches existants)…", flush=True)
    ds = load_dataset(args.dataset)
    model_pin = _cache_model(args.dataset, EXTRACT_MODEL)  # cache HIT, zéro extraction
    print(f"  modèle d'extraction épinglé (cache) : {model_pin}", flush=True)
    tree = build_theme_tree(ds, model=model_pin, embedder=DEFAULT_EMBEDDER,
                            resolution=1.0, seed=SEED)
    prepared = tree.prepared
    vecs = prepared.claim_vecs
    corpus_texts = list(prepared.claim_texts)
    idf = corpus_idf(corpus_texts)
    # Fractions de df corpus (un claim = un doc) — base du contraste distinctif.
    N = len(corpus_texts)
    corpus_df: dict[str, int] = {}
    for t in corpus_texts:
        for tok in set(_tokenizer(t)):
            corpus_df[tok] = corpus_df.get(tok, 0) + 1
    corpus_df_frac = {tok: c / N for tok, c in corpus_df.items()} if N else {}

    leaves = [tree.nodes[nid] for nid in tree.order if not tree.nodes[nid].children]
    leaves = [n for n in leaves if len(_leaf_members_texts(n, prepared)[0]) >= MIN_CLAIMS]
    leaves.sort(key=lambda n: -len(n.members))
    # Variété : dédup par titre pour éviter deux feuilles au même intitulé, puis top-N.
    picked, seen_titles = [], set()
    for node in leaves:
        if not node.representative_claims:
            reps = [prepared.claim_texts[i] for i in node.members[:REP_FOR_TITLE]]
            node.representative_claims = [r[:240] for r in reps]
        title = title_for_node(args.dataset, node) or node.label
        node.title = title
        key = title.strip().lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        picked.append(node)
        if len(picked) >= args.leaves:
            break
    print(f"  {len(picked)} feuilles retenues (n≥{MIN_CLAIMS}, titres distincts, top par taille)\n",
          flush=True)

    rows = []
    for node in picked:
        title = node.title
        mem, txt = _leaf_members_texts(node, prepared)
        # Sous-échantillon reproductible avant sélection (borne le coût de tri).
        if len(mem) > CAP:
            idxs = sorted(rng.sample(range(len(mem)), CAP))
            mem = [mem[i] for i in idxs]
            txt = [txt[i] for i in idxs]

        c_order = central_order(mem, vecs, node.centroid)
        vocab_w = _distinctive_vocab(txt, idf, corpus_df_frac)
        d_order = distinctive_order(txt, vocab_w)

        central_sample = [txt[i] for i in c_order[:SAMPLE_FOR_PROMPT]]
        distinctive_sample = [txt[i] for i in d_order[:SAMPLE_FOR_PROMPT]]
        keywords = (node.keywords or [])
        fallback = title or node.label

        print(f"=== {node.id} (n={len(node.members)}) — {title!r}", flush=True)
        print(f"    kw: {', '.join(keywords[:TOP_KEYWORDS])}", flush=True)
        print(f"    vocab distinctif: {', '.join(list(vocab_w)[:12])}", flush=True)
        print(f"    central[0]:    {central_sample[0][:110]!r}", flush=True)
        print(f"    distinctif[0]: {distinctive_sample[0][:110]!r}", flush=True)

        if args.dry:
            rows.append({"theme_id": node.id, "title": title,
                         "keywords": keywords[:TOP_KEYWORDS],
                         "distinctive_vocab": list(vocab_w)[:12],
                         "central_sample": central_sample[:5],
                         "distinctive_sample": distinctive_sample[:5]})
            print("", flush=True)
            continue

        # (a) ACTUEL : prompt v2 prod + échantillon central + mots-clés bruts.
        ta = derive_target(cleavage_system(title), "MOTS-CLÉS", keywords,
                           central_sample, fallback)
        # (b) CENTRAL + mots-clés distinctifs cadrés + consigne de contraste.
        tb = derive_target(cleavage_system(title) + CONTRAST_SUFFIX,
                           "MOTS-CLÉS DISTINCTIFS (ce qui SÉPARE ce thème des thèmes voisins)",
                           keywords, central_sample, fallback)
        # (c) CLAIMS DISTINCTIFS : prompt (a) identique, input = échantillon distinctif.
        tc = derive_target(cleavage_system(title), "MOTS-CLÉS", keywords,
                           distinctive_sample, fallback)

        row = {
            "theme_id": node.id, "n_members": len(node.members), "title": title,
            "label": node.label, "keywords": keywords[:TOP_KEYWORDS],
            "distinctive_vocab": list(vocab_w)[:12],
            "a_objet": ta["objet"], "a_justif": ta["justif"],
            "b_objet": tb["objet"], "b_justif": tb["justif"],
            "c_objet": tc["objet"], "c_justif": tc["justif"],
            "representative_claims": (node.representative_claims or [])[:REP_FOR_PANEL],
        }
        rows.append(row)
        print(f"    (a) {ta['objet']!r}", flush=True)
        print(f"    (b) {tb['objet']!r}", flush=True)
        print(f"    (c) {tc['objet']!r}\n", flush=True)

    if args.dry:
        RESULTS_PATH.with_suffix(".dry.json").write_text(
            json.dumps({"dataset": args.dataset, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"✓ dry-run écrit {RESULTS_PATH.with_suffix('.dry.json')}", flush=True)
        return

    # ------------------------------------------------------------------- #
    # Métriques : fit titre/centroïde par variante + distinctivité inter-cibles.
    # ------------------------------------------------------------------- #
    titles = [r["title"] for r in rows]
    title_vecs = embed_claim_texts(titles, embedder=DEFAULT_EMBEDDER)
    centroids = {r["theme_id"]: tree.nodes[r["theme_id"]].centroid for r in rows}
    variant_vecs = {}
    for v in ("a", "b", "c"):
        props = [r[f"{v}_objet"] for r in rows]
        variant_vecs[v] = embed_claim_texts(props, embedder=DEFAULT_EMBEDDER)

    for i, r in enumerate(rows):
        cen = centroids[r["theme_id"]]
        for v in ("a", "b", "c"):
            pv = variant_vecs[v][i]
            r[f"{v}_fit_title"] = round(max(0.0, _cos(pv, title_vecs[i])), 4)
            r[f"{v}_fit_centroid"] = round(max(0.0, _cos(pv, cen)), 4)

    summary = {"n_leaves": len(rows), "model": MODEL}
    for v in ("a", "b", "c"):
        ft = np.array([r[f"{v}_fit_title"] for r in rows])
        fc = np.array([r[f"{v}_fit_centroid"] for r in rows])
        summary[v] = {
            "mean_fit_title": round(float(ft.mean()), 4),
            "median_fit_title": round(float(np.median(ft)), 4),
            "mean_fit_centroid": round(float(fc.mean()), 4),
            # DISTINCTIVITÉ INTER-CIBLES : bas = cibles variées, haut = quasi-identiques.
            "inter_target_mean_cos": round(_mean_pairwise_cos(variant_vecs[v]), 4),
        }
    # Combien de cibles changent vs (a) ?
    summary["n_changed_b_vs_a"] = sum(1 for r in rows
                                      if r["b_objet"].strip().lower() != r["a_objet"].strip().lower())
    summary["n_changed_c_vs_a"] = sum(1 for r in rows
                                      if r["c_objet"].strip().lower() != r["a_objet"].strip().lower())

    print("================ RÉSUMÉ ================", flush=True)
    print(f"  n_leaves = {summary['n_leaves']}  modèle = {MODEL}", flush=True)
    for v in ("a", "b", "c"):
        s = summary[v]
        print(f"  ({v}) fit_titre μ={s['mean_fit_title']:.3f} méd={s['median_fit_title']:.3f} "
              f"| fit_centroïde μ={s['mean_fit_centroid']:.3f} "
              f"| INTER-CIBLES cos={s['inter_target_mean_cos']:.3f}", flush=True)
    print(f"  changées b/a: {summary['n_changed_b_vs_a']}/{len(rows)}  "
          f"c/a: {summary['n_changed_c_vs_a']}/{len(rows)}", flush=True)
    print("  (INTER-CIBLES bas = cibles distinctes ; haut = symptôme « quasi-identiques »)", flush=True)

    RESULTS_PATH.write_text(json.dumps(
        {"dataset": args.dataset, "seed": SEED, "model": MODEL, "cap": CAP,
         "embedder": DEFAULT_EMBEDDER, "sample_for_prompt": SAMPLE_FOR_PROMPT,
         "cleavage_system_a": cleavage_system("<TITRE>"),
         "cleavage_system_b_suffix": CONTRAST_SUFFIX,
         "summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ écrit {RESULTS_PATH}", flush=True)

    # ------------------------------------------------------------------- #
    # PANEL AVEUGLE : paires anonymisées X/Y (les 3 comparaisons a-b, a-c, b-c par feuille).
    # ------------------------------------------------------------------- #
    panel, key = [], {}
    pair_types = [("a", "b"), ("a", "c"), ("b", "c")]
    for r in rows:
        for v1, v2 in pair_types:
            o1, o2 = r[f"{v1}_objet"].strip(), r[f"{v2}_objet"].strip()
            if o1.lower() == o2.lower():
                continue  # cibles identiques : rien à juger
            pair_id = f"{r['theme_id']}__{v1}{v2}"
            # Anonymisation reproductible : qui est X, qui est Y (seedé).
            if rng.random() < 0.5:
                x_var, y_var, x_obj, y_obj = v1, v2, o1, o2
            else:
                x_var, y_var, x_obj, y_obj = v2, v1, o2, o1
            panel.append({
                "pair_id": pair_id,
                "title": r["title"],
                "context_claims": r["representative_claims"],
                "option_X": x_obj,
                "option_Y": y_obj,
                "question": PANEL_QUESTION,
            })
            key[pair_id] = {"X": x_var, "Y": y_var,
                            "theme_id": r["theme_id"], "title": r["title"]}

    with PANEL_PATH.open("w", encoding="utf-8") as fh:
        for item in panel:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    PANEL_KEY_PATH.write_text(json.dumps(
        {"dataset": args.dataset, "question": PANEL_QUESTION, "key": key},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ panel aveugle : {len(panel)} paires → {PANEL_PATH}", flush=True)
    print(f"✓ clé (hors aveugle) → {PANEL_KEY_PATH}", flush=True)


if __name__ == "__main__":
    main()

"""R&D CLEAVAGE v2 — la cible de clivage doit RÉSUMER le thème, pas une facette.

Problème (Bob) : `derive_cleavage` (backend/build_opinion.py) ne reçoit PAS le titre du
cluster et demande « la proposition la plus SAILLANTE » → la cible dérive vers une facette
bruyante. Ex. thème « Comprendre les réalités des Français » → cible v1 « cesser les
discours trompeurs… » (une facette, pas le centre).

v2 = 3 leviers :
  1. CONDITIONNER : passer le TITRE du cluster (`node.title`) au prompt et exiger une
     proposition qui capture le débat CENTRAL de CE thème.
  2. « central » > « saillant » : reformuler (résume le débat du thème, pas le plus bruyant).
  3. FIT : embedder la proposition (même encodeur que les claims, nomic-v2) → cosinus vs
     CENTROÏDE du cluster. `cleavage_fit ∈ [0,1]`. Fit bas → cible peu représentative.

Validation SANS re-bake : ~12-15 feuilles granddebat (dont « Comprendre les réalités… »).
On compare cible v1 vs v2 + le fit de chacune contre le MÊME centroïde. Sortie :
research/cleavage_v2_results.json + tableau imprimé. Verdict → research/cleavage_v2_note.md.

Lancement (racine du worktree) :
  MISTRAL_API_KEY=$(cat var/mistral.key) PYTHONPATH=. \
  uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
  python research/cleavage_v2.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from backend.analysis import DEFAULT_EMBEDDER, DEFAULT_SEED, build_theme_tree
from backend.build_analysis import load_dataset
from backend.titles import title_for_node
from pipeline.claims.pipeline import embed_claim_texts
from pipeline.cluster import mistral_client

SEED = DEFAULT_SEED
DATASET = "granddebat"
MODEL = os.environ.get("AGORA_OPINION_MODEL", "mistral-small-latest")
RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = RESEARCH_DIR / "cleavage_v2_results.json"
KEY_FALLBACK = Path.home() / "projects/Agora/var/mistral.key"

CAP = 60                # claims échantillonnés / feuille (mêmes pour v1 et v2)
MIN_CLAIMS = 12         # on ignore les feuilles trop petites (singletons résiduels)
N_LEAVES = 15           # nb de feuilles évaluées (les plus grosses)
SAMPLE_FOR_PROMPT = 14  # contributions montrées au prompt cleavage
REP_FOR_TITLE = 8


# --------------------------------------------------------------------------- #
# v1 — prompt ACTUEL (build_opinion.py), repris TEL QUEL (pas de titre, « saillante »).
# --------------------------------------------------------------------------- #
CLEAVAGE_SYSTEM_V1 = (
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


def _cleavage_system_v2(title: str) -> str:
    """v2 : conditionné sur le TITRE + « central » (résumé du thème) > « saillant »."""
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


def derive_cleavage(system: str, node, sample_texts: list[str]) -> dict:
    kw = ", ".join((node.keywords or [])[:10])
    contribs = "\n".join(f"- {t[:160]}" for t in sample_texts[:SAMPLE_FOR_PROMPT])
    user = f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    fallback = node.title or node.label
    try:
        raw = mistral_client.chat(messages, model=MODEL, temperature=0.0,
                                  max_tokens=200, json_mode=True)
        data = json.loads(raw)
        objet = str(data.get("objet", "")).strip()
        return {"objet": objet or fallback, "justif": str(data.get("justif", "")).strip()}
    except (mistral_client.MistralError, json.JSONDecodeError):
        return {"objet": fallback, "justif": "(repli)"}


# --------------------------------------------------------------------------- #
# Fit — cosinus entre la proposition embeddée (même encodeur) et le centroïde du cluster.
# --------------------------------------------------------------------------- #
def fit_to_centroid(proposition: str, centroid: np.ndarray) -> float:
    """cos(emb(proposition), centroïde). Encodeur PROD nomic-v2 (search_document:),
    même espace que les claims dont est dérivé le centroïde. → [-1,1], clampé à [0,1]."""
    if not proposition.strip():
        return 0.0
    v = embed_claim_texts([proposition], embedder=DEFAULT_EMBEDDER)[0]
    cos = float(np.dot(v, centroid))
    return round(max(0.0, cos), 4)


def _leaf_claim_texts(node, prepared) -> list[str]:
    out = []
    for i in node.members:
        t = (prepared.claim_texts[i] or "").strip()
        if len(t) >= 12:
            out.append(t)
    return out


def main() -> None:
    if not mistral_client.available() and KEY_FALLBACK.exists():
        os.environ["MISTRAL_API_KEY"] = KEY_FALLBACK.read_text(encoding="utf-8").strip()
    if not mistral_client.available():
        sys.exit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")

    rng = random.Random(SEED)
    print(f"Construction de l'arbre {DATASET} (caches existants)…", flush=True)
    ds = load_dataset(DATASET)
    tree = build_theme_tree(ds, embedder=DEFAULT_EMBEDDER, resolution=1.0, seed=SEED)

    leaves = [tree.nodes[nid] for nid in tree.order if not tree.nodes[nid].children]
    leaves = [n for n in leaves if len(_leaf_claim_texts(n, tree.prepared)) >= MIN_CLAIMS]
    leaves.sort(key=lambda n: -len(n.members))
    leaves = leaves[:N_LEAVES]
    print(f"  {len(leaves)} feuilles évaluées (n≥{MIN_CLAIMS}, top {N_LEAVES} par taille)\n",
          flush=True)

    rows = []
    for node in leaves:
        if not node.representative_claims:
            reps = [tree.prepared.claim_texts[i] for i in node.members[:REP_FOR_TITLE]]
            node.representative_claims = [r[:240] for r in reps]
        title = title_for_node(DATASET, node) or node.label
        node.title = title  # le titre conditionne v2

        cl = _leaf_claim_texts(node, tree.prepared)
        if len(cl) > CAP:
            cl = [cl[i] for i in sorted(rng.sample(range(len(cl)), CAP))]

        c1 = derive_cleavage(CLEAVAGE_SYSTEM_V1, node, cl)
        c2 = derive_cleavage(_cleavage_system_v2(title), node, cl)
        fit1 = fit_to_centroid(c1["objet"], node.centroid)
        fit2 = fit_to_centroid(c2["objet"], node.centroid)
        # Référence : à quel point le TITRE lui-même colle au centroïde (plafond pratique).
        fit_title = fit_to_centroid(title, node.centroid)

        row = {
            "theme_id": node.id, "n_members": len(node.members),
            "label": node.label, "title": title,
            "v1_objet": c1["objet"], "v1_justif": c1["justif"], "fit_v1": fit1,
            "v2_objet": c2["objet"], "v2_justif": c2["justif"], "fit_v2": fit2,
            "fit_title": fit_title,
            "keywords": (node.keywords or [])[:8],
        }
        rows.append(row)
        print(f"=== {node.id} (n={len(node.members)}) — {title!r}", flush=True)
        print(f"    v1 [fit {fit1:.3f}] {c1['objet']!r}", flush=True)
        print(f"    v2 [fit {fit2:.3f}] {c2['objet']!r}", flush=True)
        print(f"    (titre fit {fit_title:.3f})\n", flush=True)

    f1 = np.array([r["fit_v1"] for r in rows])
    f2 = np.array([r["fit_v2"] for r in rows])
    ft = np.array([r["fit_title"] for r in rows])
    n_v2_better = int(np.sum(f2 > f1 + 1e-6))
    n_changed = sum(1 for r in rows if r["v1_objet"].strip() != r["v2_objet"].strip())
    summary = {
        "n_leaves": len(rows),
        "mean_fit_v1": round(float(f1.mean()), 4),
        "mean_fit_v2": round(float(f2.mean()), 4),
        "mean_fit_title": round(float(ft.mean()), 4),
        "median_fit_v1": round(float(np.median(f1)), 4),
        "median_fit_v2": round(float(np.median(f2)), 4),
        "n_v2_fit_better": n_v2_better,
        "n_proposition_changed": n_changed,
        "fit_v1_range": [round(float(f1.min()), 4), round(float(f1.max()), 4)],
        "fit_v2_range": [round(float(f2.min()), 4), round(float(f2.max()), 4)],
    }
    print("================ RÉSUMÉ ================", flush=True)
    for k, v in summary.items():
        print(f"  {k:24} {v}", flush=True)

    RESULTS_PATH.write_text(json.dumps(
        {"dataset": DATASET, "seed": SEED, "model": MODEL, "cap": CAP,
         "embedder": DEFAULT_EMBEDDER,
         "cleavage_system_v1": CLEAVAGE_SYSTEM_V1,
         "cleavage_system_v2_template": _cleavage_system_v2("<TITRE>"),
         "summary": summary, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ écrit {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()

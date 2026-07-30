"""Bench de RÉGRESSION LLM — porter l'enrichissement sur de plus petits modèles, mesurer la perte.

L'extraction est déjà sur ministral-3b (le plus petit). Le coût € du pipeline est dans
l'ENRICHISSEMENT sur `mistral-large` (titres, accroches, synthèses, opinion, arguments — un
appel par thème/feuille). Ce bench mesure la PERTE DE QUALITÉ quand on descend le modèle
d'enrichissement : large → small → ministral-8b → ministral-3b.

Méthode (eval-as-truth, playbook #5) :
  - Entrées = de VRAIS thèmes d'un dataset bâti (mots-clés + claims représentatives).
  - Pour chaque modèle, on génère la sortie du rôle (ici : le TITRE) avec le MÊME prompt.
  - Un JUGE (mistral-large) note chaque sortie EN AVEUGLE (étiquettes A/B/C/D mélangées, noms de
    modèles cachés) sur une grille 0-5 (spécificité, fidélité, lisibilité).
  - Le € est le RATIO DE PRIX PUBLIÉ (large ≈ 10× small ≈ 50× ministral-3b) : le bench mesure la
    RÉTENTION de qualité ; le verdict = plus petit modèle qui garde la qualité.
Résilient : écrit au fur et à mesure dans research/llm_regression_results.json.

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \
      --extra faiss python -u research/llm_regression_bench.py [dataset] [N_thèmes]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.titles import _clean_title, _title_messages
from pipeline.cluster import mistral_client as MC

MODELS = ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
          "ministral-8b-latest", "ministral-3b-latest"]
# Prix RÉEL $/M tokens (output) — table du cost.json + tarifs publics ministral. Le verdict
# qualité/€ s'appuie dessus. (input/output moyennés côté output, dominant sur nos sorties.)
PRICE_OUT = {"mistral-large-latest": 6.0, "mistral-medium-latest": 2.0, "mistral-small-latest": 0.3,
             "ministral-8b-latest": 0.10, "ministral-3b-latest": 0.04}
PRICE_REL = {m: round(p / PRICE_OUT["ministral-3b-latest"], 1) for m, p in PRICE_OUT.items()}
# Juge = mistral-medium (flagship récent, neutre vs l'incumbent large). En aveugle ; medium est
# aussi CANDIDAT → léger biais d'auto-jugement possible (medium notant medium), signalé au verdict.
JUDGE = "mistral-medium-latest"
OUT = Path("research/llm_regression/llm_regression_results.json")


def _themes(dataset: str, n: int) -> list[dict]:
    """N thèmes FEUILLES (avec claims) d'un dataset bâti, ordre stable."""
    a = json.loads(Path(f"backend/cache/{dataset}/analysis/analysis.json").read_text())
    leaves = [t for t in a["themes"] if t.get("level") != 0 and (t.get("representative_claims"))]
    return leaves[:n]


def _gen_title(theme: dict, model: str) -> tuple[str, float]:
    node = SimpleNamespace(keywords=theme.get("keywords", []),
                           representative_claims=theme.get("representative_claims", []),
                           label=theme.get("label", ""), id=theme.get("id", "n?"))
    anchors = list(theme.get("representative_claims") or [])[:5]
    t0 = perf_counter()
    raw = MC.chat(_title_messages(node, anchors), model=model, max_tokens=32, temperature=0.2)
    return _clean_title(raw), round((perf_counter() - t0) * 1000)


def _judge_titles(theme: dict, labelled: list[tuple[str, str]]) -> dict[str, int]:
    """`labelled` = [(lettre, titre)] mélangé. Le juge note chaque titre 0-5 EN AVEUGLE."""
    kw = ", ".join((theme.get("keywords") or [])[:8])
    reps = "\n".join(f"- {c[:200]}" for c in (theme.get("representative_claims") or [])[:5])
    cands = "\n".join(f"{L}. {t}" for L, t in labelled)
    system = (
        "Tu es un JUGE de qualité de titres de thèmes citoyens. Tu notes chaque titre candidat "
        "de 0 à 5 selon : SPÉCIFICITÉ au sujet (pas générique), FIDÉLITÉ aux mots-clés/témoignages "
        "(rien d'inventé), LISIBILITÉ (phrase nominale propre, pas une salade de mots-clés). "
        "Tu ne sais PAS quel modèle a produit quel titre. Réponds STRICTEMENT en JSON "
        "{\"A\":note,\"B\":note,...} et rien d'autre."
    )
    user = f"Mots-clés du thème : {kw}\n\nTémoignages :\n{reps}\n\nTitres candidats :\n{cands}\n\nNotes JSON :"
    raw = MC.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                  model=JUDGE, max_tokens=120, temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return {L: int(v) for L, v in d.items() if isinstance(v, (int, float))}


def main(dataset: str = "tiktok", n: int = 12) -> None:
    themes = _themes(dataset, n)
    print(f"[bench] {dataset} · {len(themes)} thèmes · rôle=TITRE · modèles={len(MODELS)}", flush=True)
    results = {"dataset": dataset, "role": "title", "n_themes": len(themes),
               "models": MODELS, "per_theme": [], "scores": {m: [] for m in MODELS},
               "latency_ms": {m: [] for m in MODELS}}

    # ordre de mélange DÉTERMINISTE par thème (pas de Math.random ici) : rotation par index.
    letters = "ABCDEFGH"
    for ti, th in enumerate(themes):
        titles = {}
        for m in MODELS:
            title, ms = _gen_title(th, m)
            titles[m] = title
            results["latency_ms"][m].append(ms)
        # mélange déterministe : rotation des modèles par index de thème
        order = MODELS[ti % len(MODELS):] + MODELS[:ti % len(MODELS)]
        labelled = [(letters[i], titles[m]) for i, m in enumerate(order)]
        lab2model = {letters[i]: m for i, m in enumerate(order)}
        notes = _judge_titles(th, labelled)
        row = {"theme": th.get("id"), "keywords": (th.get("keywords") or [])[:5], "titles": {}}
        for L, model in lab2model.items():
            sc = notes.get(L)
            row["titles"][model] = {"title": titles[model], "score": sc}
            if sc is not None:
                results["scores"][model].append(sc)
        results["per_theme"].append(row)
        print(f"  [{ti+1}/{len(themes)}] {th.get('id')} noté : "
              + " · ".join(f"{m.split('-')[0][:5]}={notes.get(L,'?')}" for L, m in lab2model.items()), flush=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # Synthèse
    print("\n=== VERDICT (titre) — score moyen /5, latence, prix relatif ===")
    base = None
    summary = {}
    for m in MODELS:
        sc = results["scores"][m]
        avg = round(sum(sc) / len(sc), 2) if sc else None
        lat = round(sum(results["latency_ms"][m]) / max(len(results["latency_ms"][m]), 1))
        summary[m] = {"score_moyen": avg, "latence_ms": lat, "prix_relatif": PRICE_REL[m]}
        if m == "mistral-large-latest":
            base = avg
        ret = f"{round(100*avg/base)}%" if (avg and base) else "—"
        print(f"  {m:24} score={avg}/5  rétention={ret}  latence={lat}ms  prix×{PRICE_REL[m]}")
    results["summary"] = summary
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nRésultats → {OUT}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "tiktok"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    main(ds, n)

"""VALIDATION QUALITÉ RENFORCÉE — cible de clivage v1 vs v2, juge AVEUGLE, PANEL de 3.

Objectif (Bob) : prouver que la cible de clivage v2 (conditionnée sur le TITRE du cluster
+ « central » > « saillant ») capture mieux le débat CENTRAL d'un thème que la v1 (prompt
de prod : « la plus saillante », sans titre) — plus solidement que « 12/15 fit ».

Durcissement vs research/cleavage_v2.py :
  1. ~25 clusters (au lieu de 15).
  2. JUGE AVEUGLE PANEL de 3 : « laquelle des deux propositions (A/B anonymisées, ordre
     randomisé par juge) capture le MIEUX le débat CENTRAL du thème « <titre> » ? »,
     température 0.5 → diversité réelle. Majorité de 3.
  3. cleavage_fit = cos(emb(cible), emb(titre)) — le fit cible↔TITRE (memory : le fit
     vs centroïde est cassé/trompeur). On CORRÈLE ce fit avec le jugement du panel :
     le fit prédit-il quelle cible gagne ?

Sortie : research/v2_cleavage_cache/results.json + résumé. Rapport : research/v2_quality_note.md.

Lancement (racine du worktree) :
  MISTRAL_API_KEY=$(cat ~/projects/Analyse-des-consultations-citoyennes/var/mistral.key) \
  PYTHONPATH=. uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
  python research/v2_cleavage_quality.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from backend.analysis import DEFAULT_EMBEDDER, DEFAULT_SEED, build_theme_tree
from backend.build_analysis import load_dataset
from backend.titles import title_for_node
from pipeline.claims.ollama import parse_json_object
from pipeline.claims.pipeline import embed_claim_texts
from pipeline.cluster import mistral_client

SEED = DEFAULT_SEED
DATASET = "granddebat"
MODEL = os.environ.get("AGORA_OPINION_MODEL", "mistral-small-latest")
JUDGE_MODEL = os.environ.get("AGORA_V2_MODEL", "mistral-large-latest")
RESEARCH_DIR = Path(__file__).resolve().parent
OUT = RESEARCH_DIR / "v2_cleavage_cache"
OUT.mkdir(parents=True, exist_ok=True)
KEY_FALLBACK = Path.home() / "projects/Analyse-des-consultations-citoyennes/var/mistral.key"

CAP = 60
MIN_CLAIMS = 12
N_LEAVES = 25
SAMPLE_FOR_PROMPT = 14
REP_FOR_TITLE = 8
N_PANEL = 3
JUDGE_TEMP = 0.5

# v1 — prompt ACTUEL de prod (build_opinion.py), tel quel.
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


JUDGE_SYS = (
    "Tu es analyste NEUTRE et RIGOUREUX de consultations citoyennes. On te donne le TITRE "
    "d'un THÈME, ses mots-clés, un échantillon de CONTRIBUTIONS verbatim, puis DEUX "
    "propositions d'« objet de clivage » (A et B) — la phrase polaire censée résumer le "
    "débat CENTRAL du thème, ce sur quoi les citoyens se divisent POUR/CONTRE.\n\n"
    "Question : LAQUELLE des deux capture le MIEUX le débat CENTRAL de CE thème (ce dont "
    "parle le titre et la majorité des contributions), sans dériver vers une facette "
    "secondaire ni un détail bruyant ?\n"
    "Réponds « A », « B » ou « tie ». Ne dis « tie » que si elles sont vraiment "
    "équivalentes. Réponds STRICTEMENT en JSON : "
    "{\"meilleur\":\"A|B|tie\",\"justif\":\"une phrase courte\"}."
)


def derive_cleavage(system: str, node, sample_texts: list[str]) -> dict:
    kw = ", ".join((node.keywords or [])[:10])
    contribs = "\n".join(f"- {t[:160]}" for t in sample_texts[:SAMPLE_FOR_PROMPT])
    user = f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    fallback = node.title or node.label
    for attempt in range(5):
        try:
            raw = mistral_client.chat(messages, model=MODEL, temperature=0.0,
                                      max_tokens=200, json_mode=True, timeout=120)
            data = json.loads(raw)
            objet = str(data.get("objet", "")).strip()
            return {"objet": objet or fallback, "justif": str(data.get("justif", "")).strip()}
        except mistral_client.MistralError as exc:
            if exc.status in {0, 429, 500, 502, 503, 504} and attempt < 4:
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                continue
            return {"objet": fallback, "justif": "(repli)"}
        except json.JSONDecodeError:
            return {"objet": fallback, "justif": "(repli)"}
    return {"objet": fallback, "justif": "(repli)"}


def cos_fit(text: str, ref_vec: np.ndarray) -> float:
    """cos(emb(text), ref_vec), même encodeur (nomic-v2). Clampé à [0,1]."""
    if not text.strip():
        return 0.0
    v = embed_claim_texts([text], embedder=DEFAULT_EMBEDDER)[0]
    return round(max(0.0, float(np.dot(v, ref_vec))), 4)


def _leaf_claim_texts(node, prepared) -> list[str]:
    out = []
    for i in node.members:
        t = (prepared.claim_texts[i] or "").strip()
        if len(t) >= 12:
            out.append(t)
    return out


def run_panel(rows: list[dict]) -> None:
    """Panel de 3 juges aveugles par cluster. Ajoute aux rows : judge_winner (v1/v2/tie),
    judge_tally, et les votes. Ordre A/B randomisé indépendant par (cluster, juge)."""

    def judge_call(row, j):
        r = random.Random(f"{row['theme_id']}|{j}")
        v1_is_A = r.random() < 0.5
        propA = row["v1_objet"] if v1_is_A else row["v2_objet"]
        propB = row["v2_objet"] if v1_is_A else row["v1_objet"]
        kw = ", ".join(row["keywords"])
        contribs = "\n".join(f"- {t}" for t in row["sample_contribs"])
        user = (f"TITRE DU THÈME : « {row['title']} »\nMOTS-CLÉS : {kw}\n\n"
                f"CONTRIBUTIONS :\n{contribs}\n\n"
                f"PROPOSITION A : « {propA} »\nPROPOSITION B : « {propB} »")
        for attempt in range(5):
            try:
                raw = mistral_client.chat(
                    [{"role": "system", "content": JUDGE_SYS},
                     {"role": "user", "content": user}],
                    model=JUDGE_MODEL, temperature=JUDGE_TEMP,
                    max_tokens=200, json_mode=True, timeout=120)
                break
            except mistral_client.MistralError as exc:
                if exc.status in {0, 429, 500, 502, 503, 504} and attempt < 4:
                    time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                    continue
                raw = None
                break
        obj = parse_json_object(raw or "") or {}
        v = obj.get("meilleur")
        if v not in ("A", "B"):
            winner = "tie"
        else:
            winner_is_A = (v == "A")
            winner = "v1" if (winner_is_A == v1_is_A) else "v2"
        return {"judge": j, "v1_is_A": v1_is_A, "winner": winner,
                "justif": obj.get("justif", "")}

    jobs = [(row, j) for row in rows for j in range(N_PANEL)]
    by_id = {row["theme_id"]: row for row in rows}
    votes: dict[str, list] = {row["theme_id"]: [None] * N_PANEL for row in rows}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(judge_call, row, j): (row["theme_id"], j) for row, j in jobs}
        done = 0
        for fut in as_completed(futs):
            tid, j = futs[fut]
            votes[tid][j] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[panel] {done}/{len(jobs)} votes", flush=True)

    for row in rows:
        vs = votes[row["theme_id"]]
        tally = Counter(v["winner"] for v in vs)
        top, cnt = tally.most_common(1)[0]
        row["judge_winner"] = top if cnt > N_PANEL / 2 else "tie"
        row["judge_tally"] = dict(tally)
        row["votes"] = vs


def main() -> None:
    if not mistral_client.available() and KEY_FALLBACK.exists():
        os.environ["MISTRAL_API_KEY"] = KEY_FALLBACK.read_text(encoding="utf-8").strip()
    if not mistral_client.available():
        sys.exit("Pas de clé Mistral. Abandon.")

    rng = random.Random(SEED)
    print(f"Construction de l'arbre {DATASET}…", flush=True)
    ds = load_dataset(DATASET)
    tree = build_theme_tree(ds, embedder=DEFAULT_EMBEDDER, resolution=1.0, seed=SEED)

    leaves = [tree.nodes[nid] for nid in tree.order if not tree.nodes[nid].children]
    leaves = [n for n in leaves if len(_leaf_claim_texts(n, tree.prepared)) >= MIN_CLAIMS]
    leaves.sort(key=lambda n: -len(n.members))
    leaves = leaves[:N_LEAVES]
    print(f"  {len(leaves)} feuilles (n≥{MIN_CLAIMS}, top {N_LEAVES} par taille)\n",
          flush=True)

    results_path = OUT / "results.json"
    if (OUT / "derived.json").exists():
        rows = json.loads((OUT / "derived.json").read_text())
        print(f"[derive] cache existant ({len(rows)} clusters)")
    else:
        rows = []
        for node in leaves:
            if not node.representative_claims:
                reps = [tree.prepared.claim_texts[i] for i in node.members[:REP_FOR_TITLE]]
                node.representative_claims = [r[:240] for r in reps]
            title = title_for_node(DATASET, node) or node.label
            node.title = title

            cl = _leaf_claim_texts(node, tree.prepared)
            if len(cl) > CAP:
                cl = [cl[i] for i in sorted(rng.sample(range(len(cl)), CAP))]

            c1 = derive_cleavage(CLEAVAGE_SYSTEM_V1, node, cl)
            c2 = derive_cleavage(_cleavage_system_v2(title), node, cl)
            # fit cible↔TITRE (le fit retenu par la R&D ; vs centroïde = cassé).
            title_vec = embed_claim_texts([title], embedder=DEFAULT_EMBEDDER)[0]
            fit1 = cos_fit(c1["objet"], title_vec)
            fit2 = cos_fit(c2["objet"], title_vec)

            row = {
                "theme_id": node.id, "n_members": len(node.members),
                "label": node.label, "title": title,
                "v1_objet": c1["objet"], "v1_justif": c1["justif"], "fit_v1": fit1,
                "v2_objet": c2["objet"], "v2_justif": c2["justif"], "fit_v2": fit2,
                "keywords": (node.keywords or [])[:8],
                "sample_contribs": [t[:160] for t in cl[:SAMPLE_FOR_PROMPT]],
            }
            rows.append(row)
            print(f"=== {node.id} (n={len(node.members)}) — {title!r}", flush=True)
            print(f"    v1 [fit↔titre {fit1:.3f}] {c1['objet']!r}", flush=True)
            print(f"    v2 [fit↔titre {fit2:.3f}] {c2['objet']!r}\n", flush=True)
        (OUT / "derived.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    # Panel aveugle.
    print("\n--- PANEL aveugle (3 juges) ---", flush=True)
    run_panel(rows)

    # Agrégat jugement.
    jc = Counter(r["judge_winner"] for r in rows)
    n = len(rows)
    # Corrélation fit↔jugement : sur les clusters DÉCIDÉS, le fit de v2 prédit-il sa victoire ?
    f1 = np.array([r["fit_v1"] for r in rows])
    f2 = np.array([r["fit_v2"] for r in rows])
    n_fit_v2_better = int(np.sum(f2 > f1 + 1e-6))

    # Le fit prédit-il le vainqueur ? Pour chaque cluster décidé, "fit choisit v2" si
    # fit_v2>fit_v1. Concordance avec le panel.
    decided = [r for r in rows if r["judge_winner"] in ("v1", "v2")]
    concord = sum(1 for r in decided
                  if (r["fit_v2"] > r["fit_v1"]) == (r["judge_winner"] == "v2"))
    # Point-biserial : corrélation entre (fit_v2 - fit_v1) et (panel a choisi v2 = 1).
    if decided:
        dfit = np.array([r["fit_v2"] - r["fit_v1"] for r in decided])
        win2 = np.array([1.0 if r["judge_winner"] == "v2" else 0.0 for r in decided])
        if dfit.std() > 1e-9 and win2.std() > 1e-9:
            corr = float(np.corrcoef(dfit, win2)[0, 1])
        else:
            corr = float("nan")
    else:
        corr = float("nan")

    summary = {
        "n_clusters": n, "n_panel": N_PANEL,
        "judge_v2_wins": jc.get("v2", 0), "judge_v1_wins": jc.get("v1", 0),
        "judge_ties": jc.get("tie", 0),
        "v2_win_rate": round(jc.get("v2", 0) / n, 3),
        "v2_rate_decided": round(jc.get("v2", 0) / max(1, len(decided)), 3),
        "mean_fit_v1": round(float(f1.mean()), 4),
        "mean_fit_v2": round(float(f2.mean()), 4),
        "n_fit_v2_better": n_fit_v2_better,
        "fit_panel_concordance": round(concord / max(1, len(decided)), 3),
        "fit_delta_winner_corr": round(corr, 3) if corr == corr else None,
        "n_decided": len(decided),
    }
    out = {"dataset": DATASET, "seed": SEED, "derive_model": MODEL,
           "judge_model": JUDGE_MODEL, "fit": "cos(cible, titre)",
           "summary": summary, "rows": rows}
    results_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n================ RÉSUMÉ CLEAVAGE ================", flush=True)
    for k, v in summary.items():
        print(f"  {k:24} {v}", flush=True)
    print("\n--- cas où v2 PERD (panel choisit v1) ---")
    for r in rows:
        if r["judge_winner"] == "v1":
            print(f"  {r['theme_id']} « {r['title']} »")
            print(f"    v1 [fit {r['fit_v1']:.3f}] {r['v1_objet']!r}")
            print(f"    v2 [fit {r['fit_v2']:.3f}] {r['v2_objet']!r}")
    print(f"\n✓ écrit {results_path}", flush=True)


if __name__ == "__main__":
    main()

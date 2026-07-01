"""PROTO OPINION — quelle CIBLE mesure le mieux l'opinion citoyenne par cluster ?

Le livrable Agora : mesurer la RÉPARTITION d'opinion (favorable / défavorable /
nuancé) par thème, pas un seul côté. Dataset = GRAND DÉBAT (« démocratie et
citoyenneté ») : 3000 contributions à UNE question ouverte
  « Que faudrait-il faire pour renouer le lien entre les citoyens et les élus ? »

Le NŒUD DUR : trouver la BONNE CIBLE de stance. On compare 3 candidates, prompt et
échantillon de claims IDENTIQUES (seule la chaîne-cible change → design propre) :

  T1  titre du cluster          (sujet canonique, validé [[agora-stance-subject-verdict]])
  T2  objet de clivage dérivé   (1 passe LLM/cluster → proposition polaire débattable :
                                 « instaurer le RIC », « réduire le nombre d'élus »…)
  T3  question de consultation  (rejetée avant — re-testée ici sur clusters clivants)

Pour chaque (cluster × cible) : un LLM classe la stance de chaque claim ENVERS la cible
(favorable / defavorable / nuance + justif). On agrège, puis on mesure quelle cible est
la plus DISCRIMINANTE (sépare vraiment pour/contre) :
  - engagement = (fav+def)/n           → la cible fait-elle prendre position ? (1 - %nuance)
  - opposition = min(fav,def)/(fav+def) → révèle-t-elle l'opposition réelle ? (clivage)
  - discrimination = écart-type de la part favorable ENTRE clusters → sépare-t-elle
                     clivants et consensuels ?

Sorties : research/opinion_proto_results.json (chiffres) +
research/opinion_proto_annot.json (échantillon à annoter à la main pour le taux
d'erreur). Verdict rédigé : research/opinion_proto_note.md.

Lancement (racine du worktree) :
  MISTRAL_API_KEY=$(cat var/mistral.key) PYTHONPATH=. \
  uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
  python research/opinion_proto.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from backend.recluster import load_cache
from backend.live_cluster import build_live_tree
from backend.titles import title_for_node
from pipeline.cluster import mistral_client

SEED = 42
DATASET = "granddebat"
RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = RESEARCH_DIR / "opinion_proto_results.json"
ANNOT_PATH = RESEARCH_DIR / "opinion_proto_annot.json"

MODEL = os.environ.get("AGORA_MISTRAL_MODEL", "mistral-small-latest")
BATCH = 10               # claims par appel stance
CAP = 60                 # claims échantillonnés par cluster (mêmes pour les 3 cibles)
MIN_MEMBERS = 100        # on ignore les macros résiduelles (singletons)
REP_FOR_TITLE = 8
N_ANNOT = 45             # taille de l'échantillon dumpé pour annotation manuelle

# Worktree R&D : ni var/ ni claims.json embarqués → repli lecture seule dépôt principal.
MAIN_REPO = Path("~/projects/Analyse-des-consultations-citoyennes")
CLAIMS_JSON = MAIN_REPO / "backend" / "cache" / DATASET / "claims.json"
KEY_FALLBACK = MAIN_REPO / "var" / "mistral.key"

QUESTION = ("Que faudrait-il faire pour renouer le lien entre les citoyens et les élus "
            "qui les représentent ?")


# --------------------------------------------------------------------------- #
# Prompt STANCE — IDENTIQUE pour les 3 cibles (seule `cible` change).
# --------------------------------------------------------------------------- #
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


def stance_batch(cible: str, items: list[tuple[int, str]]) -> dict[int, dict]:
    lines = [f"[{i}] {text}" for i, text in items]
    user = (f"CIBLE : {cible}\n\n"
            f"CONTRIBUTIONS (réponds pour chaque [indice]) :\n" + "\n".join(lines))
    messages = [{"role": "system", "content": STANCE_SYSTEM},
                {"role": "user", "content": user}]
    raw = mistral_client.chat(messages, model=MODEL, temperature=0.0,
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


def run_stance(cible: str, items: list[tuple[int, str]]) -> dict[int, dict]:
    """Stance sur tous les `items` (batché, repli unitaire)."""
    results: dict[int, dict] = {}
    for start in range(0, len(items), BATCH):
        batch = items[start:start + BATCH]
        try:
            got = stance_batch(cible, batch)
        except (mistral_client.MistralError, json.JSONDecodeError):
            got = {}
        for i, text in batch:
            if i not in got:
                try:
                    got.update(stance_batch(cible, [(i, text)]))
                except (mistral_client.MistralError, json.JSONDecodeError):
                    got[i] = {"stance": "nuance", "justif": "(échec LLM)"}
        results.update(got)
        time.sleep(0.05)
    return results


# --------------------------------------------------------------------------- #
# T2 — objet de clivage dérivé (1 passe LLM par cluster).
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


def derive_cleavage(node, sample_texts: list[str]) -> dict:
    kw = ", ".join((node.keywords or [])[:10])
    contribs = "\n".join(f"- {t[:160]}" for t in sample_texts[:14])
    user = f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    messages = [{"role": "system", "content": CLEAVAGE_SYSTEM},
                {"role": "user", "content": user}]
    try:
        raw = mistral_client.chat(messages, model=MODEL, temperature=0.0,
                                  max_tokens=200, json_mode=True)
        data = json.loads(raw)
        objet = str(data.get("objet", "")).strip()
        return {"objet": objet or node.label, "justif": str(data.get("justif", "")).strip()}
    except (mistral_client.MistralError, json.JSONDecodeError):
        return {"objet": node.label, "justif": "(repli label)"}


# --------------------------------------------------------------------------- #
def claims_of_cluster(members: list[int], ideas, claims_by_avis) -> list[tuple[str, str]]:
    """(avis_id, claim_text) verbatim pour les avis membres."""
    out: list[tuple[str, str]] = []
    for i in members:
        aid = ideas[i].id
        for c in claims_by_avis.get(aid, []):
            t = (c.get("text") or "").strip()
            if len(t) >= 12:
                out.append((aid, t))
    return out


def metrics(counts: Counter, n: int) -> dict:
    fav, dfv, nu = counts.get("favorable", 0), counts.get("defavorable", 0), counts.get("nuance", 0)
    pol = fav + dfv
    return {
        "favorable": fav, "defavorable": dfv, "nuance": nu, "n": n,
        "engagement": round(pol / n, 3) if n else 0.0,         # 1 - %nuance
        "opposition": round(min(fav, dfv) / pol, 3) if pol else 0.0,
        "fav_share": round(fav / n, 3) if n else 0.0,
        "def_share": round(dfv / n, 3) if n else 0.0,
    }


def analyse_cluster(mid, tree, ideas, claims_by_avis, rng) -> dict:
    node = tree.nodes[mid]
    members = node.members
    reps = [(ideas[i].text_clean or ideas[i].text) for i in members[:REP_FOR_TITLE]]
    node.representative_claims = [r[:240] for r in reps]
    title = title_for_node(DATASET, node) or node.label

    cl = claims_of_cluster(members, ideas, claims_by_avis)
    if len(cl) > CAP:
        cl = [cl[i] for i in sorted(rng.sample(range(len(cl)), CAP))]
    sample = [(j, txt) for j, (_aid, txt) in enumerate(cl)]
    avis_ids = [aid for aid, _ in cl]

    cleavage = derive_cleavage(node, [t for _, t in sample])

    targets = {"title": title, "cleavage": cleavage["objet"], "question": QUESTION}
    print(f"\n=== {mid} (n={len(members)}, {len(sample)} claims) ===", flush=True)
    print(f"    T1 titre    : {title!r}", flush=True)
    print(f"    T2 clivage  : {cleavage['objet']!r}", flush=True)

    per_target = {}
    raw_stances = {}
    for tname, cible in targets.items():
        st = run_stance(cible, sample)
        counts = Counter(st[j]["stance"] for j, _ in sample if j in st)
        per_target[tname] = {"cible": cible, **metrics(counts, len(sample))}
        raw_stances[tname] = st
        m = per_target[tname]
        print(f"    [{tname:8}] fav={m['favorable']:3} def={m['defavorable']:3} "
              f"nu={m['nuance']:3} | engag={m['engagement']:.2f} oppo={m['opposition']:.2f}",
              flush=True)

    examples = []
    for j, txt in sample[:8]:
        examples.append({
            "claim": txt[:200],
            "title": raw_stances["title"].get(j, {}),
            "cleavage": raw_stances["cleavage"].get(j, {}),
            "question": raw_stances["question"].get(j, {}),
        })

    return {
        "node_id": mid, "n_members": len(members), "n_claims_sampled": len(sample),
        "keywords": (node.keywords or [])[:8],
        "title": title, "cleavage_object": cleavage,
        "targets": per_target, "examples": examples,
        "_sample": [(avis_ids[j], txt) for j, txt in sample],
        "_raw": {t: {str(j): raw_stances[t].get(j, {}) for j, _ in sample} for t in targets},
    }


def discrimination(clusters: list[dict], tname: str) -> dict:
    """Agrégat d'une cible SUR les clusters : moyennes + pouvoir discriminant."""
    eng = [c["targets"][tname]["engagement"] for c in clusters]
    opp = [c["targets"][tname]["opposition"] for c in clusters]
    favs = [c["targets"][tname]["fav_share"] for c in clusters]
    return {
        "mean_engagement": round(float(np.mean(eng)), 3),
        "mean_opposition": round(float(np.mean(opp)), 3),
        "fav_share_std": round(float(np.std(favs)), 3),       # discrimination inter-clusters
        "fav_share_range": [round(min(favs), 3), round(max(favs), 3)],
    }


def build_annotation_sample(clusters: list[dict], rng) -> list[dict]:
    """Échantillon (claim, cible=T2, stance LLM) à annoter à la main → taux d'erreur."""
    pool = []
    for c in clusters:
        cible = c["targets"]["cleavage"]["cible"]
        raw = c["_raw"]["cleavage"]
        for j, (aid, txt) in enumerate(c["_sample"]):
            st = raw.get(str(j), {})
            if st.get("stance"):
                pool.append({"node": c["node_id"], "avis_id": aid, "cible": cible,
                             "claim": txt, "llm_stance": st["stance"],
                             "llm_justif": st.get("justif", ""), "gold": None})
    rng.shuffle(pool)
    return pool[:N_ANNOT]


def main() -> None:
    if not mistral_client.available() and KEY_FALLBACK.exists():
        os.environ["MISTRAL_API_KEY"] = KEY_FALLBACK.read_text(encoding="utf-8").strip()
    if not mistral_client.available():
        sys.exit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")

    rng = random.Random(SEED)
    print("Chargement cache + clustering live granddebat…", flush=True)
    ideas, vecs, weights = load_cache(DATASET)
    tree = build_live_tree(ideas, vecs, weights, seed=SEED)
    claims_by_avis = json.loads(CLAIMS_JSON.read_text(encoding="utf-8"))["claims"]

    macros = [m for m in sorted(tree.macros, key=lambda m: -len(tree.nodes[m].members))
              if len(tree.nodes[m].members) >= MIN_MEMBERS]
    print(f"  {len(ideas)} avis, {len(tree.macros)} macros, "
          f"{len(macros)} analysées (n≥{MIN_MEMBERS})", flush=True)

    clusters = [analyse_cluster(mid, tree, ideas, claims_by_avis, rng) for mid in macros]

    summary = {t: discrimination(clusters, t) for t in ("title", "cleavage", "question")}
    print("\n================ DISCRIMINATION PAR CIBLE ================", flush=True)
    for t, s in summary.items():
        print(f"  {t:9} engag={s['mean_engagement']:.2f}  oppo={s['mean_opposition']:.2f}  "
              f"fav_std={s['fav_share_std']:.2f}  fav∈{s['fav_share_range']}", flush=True)

    annot = build_annotation_sample(clusters, rng)
    ANNOT_PATH.write_text(json.dumps(annot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ échantillon d'annotation ({len(annot)}) → {ANNOT_PATH}", flush=True)

    # Allège la sortie : on retire les structures internes volumineuses.
    for c in clusters:
        c.pop("_sample", None)
        c.pop("_raw", None)
    results = {
        "dataset": DATASET, "seed": SEED, "model": MODEL, "question": QUESTION,
        "cap_claims_per_cluster": CAP, "stance_prompt_system": STANCE_SYSTEM,
        "cleavage_prompt_system": CLEAVAGE_SYSTEM,
        "n_ideas": len(ideas), "n_macros_analyzed": len(macros),
        "summary_discrimination": summary, "clusters": clusters,
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ écrit {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()

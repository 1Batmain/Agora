"""ARGUMENT MINING VERBATIM — prototype R&D (VOLET 3, lane stance-argmining).

Contexte (cf. `.agent/inbox/stance-argmining-2026-07-07.md`) : l'argument mining servi
(`backend/build_arguments.py`) fait « synthèse puis re-sourçage » — le TEXTE d'argument
affiché est une phrase REFORMULÉE par le LLM → l'invariant verbatim d'Agora est cassé
(le front l'admet : `ArgumentsPanel.tsx:11`). Ce proto re-conçoit l'argument mining pour
que CHAQUE argument servi soit un SPAN VERBATIM de témoignage — comme un claim, jamais une
synthèse. Décision Bob : « verbatim SEUL » (pas de génération UX pour l'instant), et
BENCHER deux variantes :

  * V-CLUSTER  — ZÉRO LLM sur le texte servi. Par groupe (feuille × stance), medoïde glouton
                 sur les embeddings claims (déjà en cache) : 1 argument = 1 sous-cluster,
                 son TEXTE = le claim MÉDOÏDE (verbatim). Invariant garanti par construction.
  * V-SELECT   — le LLM JUGE (ne rédige pas) : il SÉLECTIONNE les indices des claims les plus
                 représentatifs d'idées distinctes. Le texte servi = le claim choisi
                 (verbatim), re-validé comme sous-chaîne d'avis. 1 appel LLM par groupe.

ISOLATION TOTALE : lit UNIQUEMENT les caches servis sous `cached_data/<ds>/` (claims +
embeddings + claim_stance + avis), n'écrit QUE sous `research/`. Ne touche NI la prod NI les
caches servis NI le pipeline `backend/`.

Groupes : partition POUR/CONTRE directement issue de `claim_stance.json` (baké par
`build_opinion`) — clé claim = `{avis_id}#{global_index}`, l'index global indexe aussi la
ligne d'embedding (vérifié : flatten insertion-order de claims.json ≡ index `#N`).

Usage :
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/argmine_verbatim.py --dataset lutte-contre-les-fausses-informations
    # V-CLUSTER seul (offline, zéro clé) :
    uv run python research/argmine_verbatim.py --dataset <ds> --methods vcluster
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.claims.span import as_claim
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHED = ROOT / "cached_data"

MIN_SUPPORT = int(os.environ.get("ARG_MIN_SUPPORT", "2"))   # claims min derrière un argument
SIM_THRESHOLD = float(os.environ.get("ARG_SIM_THRESHOLD", "0.55"))  # seuil de rattachement
MAX_K = int(os.environ.get("ARG_MAX_K", "5"))               # arguments max / groupe
SELECT_MODEL = os.environ.get("ARG_SELECT_MODEL", "mistral-large-latest")
_STANCES = ("favorable", "defavorable")
_LABEL = {"favorable": "pour", "defavorable": "contre"}


# --------------------------------------------------------------------------- #
# Chargement des caches servis (offline, read-only)
# --------------------------------------------------------------------------- #
def load_corpus(dataset: str):
    base = CACHED / dataset
    claims_raw = json.loads((base / "claims.json").read_text())["claims"]
    vecs = np.load(base / "claims_emb.npz")["vecs"].astype(np.float32)
    stance = json.loads((base / "analysis" / "claim_stance.json").read_text())
    avis = json.loads((base / "analysis" / "avis.json").read_text())

    # Flatten insertion-order → index GLOBAL (≡ clé `#N` de claim_stance ET ligne d'embedding).
    owner: list[str] = []
    texts: list[str] = []
    specs: list[dict] = []  # claim dict (text/spans/target) pour la validation verbatim
    for aid, lst in claims_raw.items():
        for c in lst:
            owner.append(aid)
            texts.append(c["text"])
            specs.append(c)
    assert len(texts) == vecs.shape[0], (len(texts), vecs.shape)
    return {"owner": owner, "texts": texts, "specs": specs, "vecs": vecs,
            "stance": stance, "avis": avis}


def build_groups(corpus) -> dict[tuple[str, str], list[int]]:
    """(theme_id, stance) → indices globaux des claims, pour stance ∈ {favorable, defavorable}."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key, rec in corpus["stance"].items():
        st = rec.get("stance")
        if st not in _STANCES:
            continue
        gi = int(key.rsplit("#", 1)[1])
        groups[(rec["theme_id"], st)].append(gi)
    return {k: sorted(v) for k, v in groups.items() if len(v) >= MIN_SUPPORT}


# --------------------------------------------------------------------------- #
# Assignation exclusive commune (support honnête, comptes disjoints)
# --------------------------------------------------------------------------- #
def _exclusive_support(seed_rows: list[int], gvecs: np.ndarray,
                       *, sim_threshold: float) -> dict[int, list[int]]:
    """Chaque claim du groupe → son SEED le plus proche (argmax cosinus, ≥ seuil).

    Renvoie {seed_row_local: [claim_row_local, ...]}. Comptes disjoints (un claim soutient
    AU PLUS un argument) — même garantie d'honnêteté que `back_match` du build servi.
    """
    seeds = gvecs[seed_rows]                       # (k, d)
    sims = gvecs @ seeds.T                          # (n, k)
    best = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(sims.shape[0]), best]
    support: dict[int, list[int]] = {r: [] for r in seed_rows}
    for j in range(gvecs.shape[0]):
        if best_sim[j] >= sim_threshold:
            support[seed_rows[int(best[j])]].append(j)
    return support


# --------------------------------------------------------------------------- #
# V-CLUSTER — medoïde glouton, ZÉRO LLM sur le texte servi
# --------------------------------------------------------------------------- #
def vcluster(gvecs: np.ndarray, *, sim_threshold: float, max_k: int) -> list[int]:
    """Indices LOCAUX des claims-medoïdes (seeds), gloutons par centralité décroissante.

    On sème le claim le plus central encore NON couvert, on couvre ses voisins (≥ seuil),
    on répète jusqu'à max_k ou couverture totale. Déterministe, aucun appel LLM.
    """
    n = gvecs.shape[0]
    centroid = gvecs.mean(axis=0)
    nrm = np.linalg.norm(centroid)
    if nrm > 0:
        centroid = centroid / nrm
    order = list(np.argsort(-(gvecs @ centroid)))   # plus central d'abord
    covered = np.zeros(n, dtype=bool)
    seeds: list[int] = []
    for i in order:
        if covered[i]:
            continue
        seeds.append(int(i))
        covered |= (gvecs @ gvecs[i]) >= sim_threshold
        if len(seeds) >= max_k or covered.all():
            break
    return seeds


# --------------------------------------------------------------------------- #
# V-SELECT — le LLM SÉLECTIONNE des claims (ne rédige jamais)
# --------------------------------------------------------------------------- #
_SELECT_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne des CONTRIBUTIONS citoyennes "
    "NUMÉROTÉES, toutes du même camp (toutes POUR, ou toutes CONTRE une proposition). Ta "
    "tâche : SÉLECTIONNER les contributions qui représentent le MIEUX les arguments PRINCIPAUX "
    "et DISTINCTS du groupe — une par idée distincte, la plus claire et représentative. Tu ne "
    "RÉDIGES rien, tu ne reformules rien : tu renvoies UNIQUEMENT les NUMÉROS des contributions "
    "choisies, au plus {k}, de la plus représentative à la moins. Ignore les redites. "
    'Réponds en JSON strict : {{"selected":[<int>, ...]}} — rien d\'autre.'
)


def vselect(texts: list[str], *, model: str, max_k: int) -> list[int]:
    """Indices LOCAUX sélectionnés par le LLM (repli : [] → V-CLUSTER prendra le relais)."""
    numbered = "\n".join(f"[{i}] {t[:240]}" for i, t in enumerate(texts))
    messages = [
        {"role": "system", "content": _SELECT_SYSTEM.format(k=max_k)},
        {"role": "user", "content": f"CONTRIBUTIONS :\n{numbered}"},
    ]
    try:
        raw = mistral_client.chat(messages, model=model, temperature=0.0,
                                  max_tokens=200, json_mode=True)
        sel = json.loads(raw).get("selected", [])
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return []
    out: list[int] = []
    for v in sel:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= iv < len(texts) and iv not in out:
            out.append(iv)
        if len(out) >= max_k:
            break
    return out


# --------------------------------------------------------------------------- #
# Mine un groupe avec une méthode donnée → arguments verbatim
# --------------------------------------------------------------------------- #
def mine_group(rows: list[int], corpus, *, method: str, model: str) -> list[dict]:
    gvecs = corpus["vecs"][rows]                     # (n, d), déjà L2-normalisé
    texts = [corpus["texts"][r] for r in rows]
    if method == "vcluster":
        seed_local = vcluster(gvecs, sim_threshold=SIM_THRESHOLD, max_k=MAX_K)
    elif method == "vselect":
        seed_local = vselect(texts, model=model, max_k=MAX_K)
        if not seed_local:                            # repli gracieux : V-CLUSTER
            seed_local = vcluster(gvecs, sim_threshold=SIM_THRESHOLD, max_k=MAX_K)
    else:
        raise ValueError(method)

    support = _exclusive_support(seed_local, gvecs, sim_threshold=SIM_THRESHOLD)
    args: list[dict] = []
    for s in seed_local:
        members = support[s]
        if len(members) < MIN_SUPPORT:
            continue
        gi = rows[s]                                  # index GLOBAL du medoïde
        args.append({
            "argument": corpus["texts"][gi],          # VERBATIM (claim médoïde)
            "avis_id": corpus["owner"][gi],
            "claim_id": f"{corpus['owner'][gi]}#{gi}",
            "n_support": len(members),
            "sources": [{"claim_id": f"{corpus['owner'][rows[m]]}#{rows[m]}",
                         "text": corpus["texts"][rows[m]]} for m in members[:5]],
        })
    args.sort(key=lambda a: -a["n_support"])
    return args


# --------------------------------------------------------------------------- #
# Contrôle DUR de l'invariant : chaque texte servi est VERBATIM (sous-chaîne d'avis)
# --------------------------------------------------------------------------- #
def verbatim_audit(themes: list[dict], corpus) -> dict:
    """100 % des textes d'arguments servis doivent être verbatim (claim ancré dans son avis).

    Un claim peut être multi-spans (join ` … ` de sous-chaînes non-contiguës) — on valide via
    `Claim.is_verbatim` contre le texte d'avis, exactement comme le gate d'extraction.
    """
    total = ok = 0
    failures: list[str] = []
    for th in themes:
        for a in th["arguments"]:
            total += 1
            avis_text = corpus["avis"].get(a["avis_id"], {}).get("text", "")
            gi = int(a["claim_id"].rsplit("#", 1)[1])
            claim = as_claim(corpus["specs"][gi], avis_text=avis_text)
            if claim.is_verbatim(avis_text) and a["argument"] == claim.text:
                ok += 1
            else:
                failures.append(a["claim_id"])
    return {"total": total, "verbatim_ok": ok,
            "rate": round(ok / total, 4) if total else 1.0,
            "failures": failures[:10]}


# --------------------------------------------------------------------------- #
def run(dataset: str, methods: list[str]) -> dict:
    corpus = load_corpus(dataset)
    groups = build_groups(corpus)
    print(f"[argmine] {dataset} · {len(groups)} groupes (feuille×stance) ≥{MIN_SUPPORT} claims")

    out: dict = {"dataset": dataset, "params": {
        "min_support": MIN_SUPPORT, "sim_threshold": SIM_THRESHOLD, "max_k": MAX_K,
        "select_model": SELECT_MODEL}, "methods": {}}

    for method in methods:
        if method == "vselect" and not mistral_client.available():
            print("[argmine] pas de clé Mistral → V-SELECT sauté")
            continue
        by_theme: dict[str, dict] = {}
        n_args = 0
        for (theme_id, stance), rows in sorted(groups.items()):
            args = mine_group(rows, corpus, method=method, model=SELECT_MODEL)
            for a in args:
                a["stance"] = _LABEL[stance]
            if not args:
                continue
            th = by_theme.setdefault(theme_id, {"theme_id": theme_id, "arguments": []})
            th["arguments"].extend(args)
            n_args += len(args)
        themes = sorted(by_theme.values(), key=lambda t: t["theme_id"])
        audit = verbatim_audit(themes, corpus)
        # Couverture : part des claims de groupe rattachés à un argument gardé.
        covered = sum(sum(a["n_support"] for a in th["arguments"]) for th in themes)
        total_claims = sum(len(r) for r in groups.values())
        out["methods"][method] = {
            "n_themes": len(themes), "n_arguments": n_args,
            "verbatim": audit,
            "coverage": round(covered / total_claims, 3) if total_claims else 0.0,
            "themes": themes,
        }
        print(f"[argmine] {method}: {n_args} args / {len(themes)} thèmes · "
              f"verbatim {audit['verbatim_ok']}/{audit['total']} "
              f"({audit['rate']*100:.0f}%) · couverture {out['methods'][method]['coverage']*100:.0f}%")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prototype argument mining VERBATIM (R&D).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default="vcluster,vselect",
                    help="liste séparée par des virgules : vcluster,vselect")
    ap.add_argument("--out", default=str(Path(__file__).parent / "argmine_verbatim_results.json"))
    args = ap.parse_args()
    result = run(args.dataset, [m.strip() for m in args.methods.split(",") if m.strip()])
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[argmine] → {args.out}")


if __name__ == "__main__":
    main()

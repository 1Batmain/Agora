"""ÉMERGENCE RAFFINÉE — densité (primitif thème) + fusion des sur-découpes + filtre LLM.

Corrige les deux manques identifiés sur republique-numerique (`emerge_note.md`) :
  1. **Sur-découpe** : le primitif thème FORCE ≥2 communautés même sur une feuille homogène.
     → on FUSIONNE les communautés dont les centroïdes sont trop proches (cos ≥ MERGE_THR) :
     un vrai clivage a des pôles séparés, pas deux paraphrases du même point.
  2. **Substance** : « dense ≠ argument » (méta-commentaires, hors-sujet). → un jugement LLM
     LÉGER par communauté qui (a) VALIDE que c'est un argument sur le fond et (b) DÉSIGNE le
     membre le plus NET (le LLM ne rédige RIEN : il filtre et pointe un membre verbatim).
     L'anti-biais est préservé : la communauté a émergé de la DONNÉE avant tout LLM.

Affichage = le membre choisi (verbatim), pas le médoïde fade. Fail-closed : une feuille sans
aucun argument substantiel = « positions pas assez développées ».

Lit `research/emerge_cache/<ds>/` (via `emerge_build.py`). N'écrit que sous `research/`.
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/emerge_refined.py --dataset tiktok
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from research.emerge_proto import _load, emerge_leaf, dispersion, medoid
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
MERGE_THR = float(os.environ.get("EMERGE_MERGE_THR", "0.85"))   # centroïdes ≥ → même argument
JUDGE_MODEL = os.environ.get("EMERGE_JUDGE_MODEL", "mistral-large-latest")
MIN_SUPPORT = int(os.environ.get("EMERGE_MIN_SUPPORT", "3"))


def _centroid(gis: list[int], vecs: np.ndarray) -> np.ndarray:
    s = vecs[gis].sum(axis=0)
    n = np.linalg.norm(s)
    return s / n if n > 0 else s


def merge_oversplit(communities: list[list[int]], vecs: np.ndarray,
                    *, thr: float) -> list[list[int]]:
    """Fusionne gloutonnement les communautés à centroïdes trop proches (sur-découpe)."""
    cents = [_centroid(c, vecs) for c in communities]
    merged: list[list[int]] = []
    mcents: list[np.ndarray] = []
    for c, ce in sorted(zip(communities, cents), key=lambda x: -len(x[0])):
        home = next((i for i, mc in enumerate(mcents) if float(np.dot(mc, ce)) >= thr), None)
        if home is None:
            merged.append(list(c)); mcents.append(ce)
        else:
            merged[home].extend(c); mcents[home] = _centroid(merged[home], vecs)
    return merged


_JUDGE_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne le TITRE d'un thème et un GROUPE "
    "de contributions citoyennes NUMÉROTÉES regroupées automatiquement car proches. "
    "Deux tâches, SANS rien rédiger :\n"
    "1. is_argument : ce groupe défend-il une POSITION SUR LE FOND du sujet — une idée, une "
    "demande, une proposition, ou une raison pour/contre quelque chose ? Une position même "
    "CONSENSUELLE compte (ex. « l'anonymisation doit être garantie », « publier les sources »). "
    "Réponds false UNIQUEMENT si c'est : un commentaire sur LA CONSULTATION ou LE TEXTE lui-même "
    "(« hors sujet », « bien rédigé », « à réécrire », « pas à sa place »), une simple QUESTION, "
    "ou du bavardage sans contenu de fond.\n"
    "2. best : le NUMÉRO de la contribution qui exprime cette position le plus CLAIREMENT (celle "
    "qu'on afficherait). Si is_argument=false, mets best=-1.\n"
    'Réponds en JSON strict : {"is_argument":<bool>,"best":<int>} — rien d\'autre.'
)


def judge_community(title: str, texts: list[str], *, model: str) -> tuple[bool, int]:
    numbered = "\n".join(f"[{i}] {t[:200]}" for i, t in enumerate(texts))
    messages = [{"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": f"TITRE : {title}\n\nCONTRIBUTIONS :\n{numbered}"}]
    try:
        raw = mistral_client.chat(messages, model=model, temperature=0.0,
                                  max_tokens=40, json_mode=True)
        data = json.loads(raw)
        is_arg = bool(data.get("is_argument", False))
        best = int(data.get("best", -1))
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError, ValueError):
        return False, -1
    if not is_arg or not (0 <= best < len(texts)):
        return (False, -1) if not is_arg else (True, 0)
    return True, best


def run(dataset: str, *, judge: bool) -> dict:
    vecs, claims, leaves = _load(dataset)
    texts = [c["text"] for c in claims]

    per_leaf, n_arg, n_meta = [], 0, 0
    for lf in leaves:
        gis = lf["member_gis"]
        if len(gis) < MIN_SUPPORT:
            continue
        comms = emerge_leaf(gis, vecs, None)
        comms = merge_oversplit(comms, vecs, thr=MERGE_THR)
        comms = [c for c in comms if len(c) >= MIN_SUPPORT]
        args = []
        for members_gi in sorted(comms, key=len, reverse=True):
            # membres triés par centralité (pour numéroter le LLM et capper)
            sub = vecs[members_gi]
            central = list(np.argsort(-(sub @ _centroid(members_gi, vecs))))
            ordered = [members_gi[i] for i in central]
            cand_texts = [texts[g] for g in ordered[:15]]
            if judge:
                is_arg, best = judge_community(lf["title"], cand_texts, model=JUDGE_MODEL)
            else:
                is_arg, best = True, 0    # sans LLM : médoïde (central) comme repli
            if not is_arg:
                n_meta += 1
                continue
            display_gi = ordered[best]
            args.append({"n_support": len(members_gi),
                         "dispersion": round(dispersion(members_gi, vecs), 3),
                         "argument": texts[display_gi], "display_gi": display_gi})
            n_arg += 1
        per_leaf.append({"theme_id": lf["theme_id"], "title": lf["title"],
                         "n_claims": len(gis), "n_arguments": len(args), "arguments": args})

    n_clivage = sum(1 for l in per_leaf if l["n_arguments"] >= 2)
    n_consensus = sum(1 for l in per_leaf if l["n_arguments"] == 1)
    n_underdev = sum(1 for l in per_leaf if l["n_arguments"] == 0)
    summary = {
        "dataset": dataset, "judged": judge, "merge_thr": MERGE_THR,
        "n_leaves": len(per_leaf), "n_arguments": n_arg,
        "communities_filtered_as_meta": n_meta,
        "leaves_clivage": n_clivage, "leaves_consensus": n_consensus,
        "leaves_underdeveloped": n_underdev,
    }
    print(f"[refined] {dataset} · {len(per_leaf)} feuilles · {n_arg} arguments substantiels · "
          f"{n_meta} communautés filtrées (méta/hors-sujet)")
    print(f"[refined] clivage {n_clivage} · consensus {n_consensus} · sous-développé {n_underdev}")
    return {"summary": summary, "leaves": per_leaf}


def main() -> None:
    ap = argparse.ArgumentParser(description="Émergence raffinée (merge + filtre LLM) — R&D.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--no-judge", action="store_true", help="sans le filtre LLM (baseline densité)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    judge = not args.no_judge
    if judge and not mistral_client.available():
        raise SystemExit("Pas de clé Mistral (ou utilise --no-judge).")
    out = args.out or str(Path(__file__).parent / f"emerge_refined_{args.dataset}.json")
    result = run(args.dataset, judge=judge)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[refined] → {out}")


if __name__ == "__main__":
    main()

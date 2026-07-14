"""Construit le cache d'embeddings nomic-v2 d'UN dataset (un SEUL appel torch).

Cache **par dataset** : `backend/cache/<dataset>/{embeddings.npy, ideas.jsonl,
meta.json}`. Le serveur live (`backend/server.py`) re-clusterise à partir de ce
cache ; il ne charge JAMAIS le modèle torch.

GÉNÉRIQUE (zéro littéral de corpus) : un dataset = un **descripteur**
(`pipeline/ingest/descriptors/<id>.json`) lu par le `read_generic` de la lane
ingest, + des options de sous-échantillonnage déclaratives (cap, équilibrage par
champ, min_chars, dédup exacte). Aucun nom de corpus n'est codé en dur.

Pipeline : `read_generic(desc)` → `to_idea` (nettoyage + langue + anonymisation,
réutilise la lane ingest) → subset (min_chars → dédup exacte → échantillon
ÉQUILIBRÉ par champ, cap) → embed nomic-v2 → cache aligné.

Usage :
    # superset complet (défaut tiktok, rétro-compat)
    uv run --extra embed-contender python -m backend.build_cache --dataset tiktok
    # échantillon multilingue équilibré, plafonné (vitrine x-stance)
    uv run --extra embed-contender python -m backend.build_cache \
        --dataset xstance --balance lang --cap 3000 --min-chars 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np

from backend.recluster import CACHE_DIR, EMB_NAME, IDEAS_NAME, META_NAME
from pipeline.embed.embedder import Embedder
from pipeline.ingest import download
from pipeline.ingest.build import to_idea
from pipeline.ingest.config import DESCRIPTORS_DIR
from pipeline.ingest.sources import SourceDescriptor, read_generic


def resolve_descriptor(dataset: str, explicit: str | None) -> SourceDescriptor:
    """Descripteur d'un dataset : chemin explicite, sinon `descriptors/<id>.json`."""
    path = Path(explicit) if explicit else DESCRIPTORS_DIR / f"{dataset}.json"
    if not path.exists():
        raise SystemExit(
            f"Descripteur introuvable : {path}\n"
            f"Dépose un descripteur `{dataset}.json` dans {DESCRIPTORS_DIR} "
            "(cf. pipeline/ingest/README.md)."
        )
    return SourceDescriptor.from_json(path)


def _idea_lang(idea: dict) -> str:
    return idea["props"].get("lang", "") or ""


def _idea_clean(idea: dict) -> str:
    return idea["props"].get("text_clean") or idea["props"].get("text") or ""


def subset(
    ideas: list[dict],
    *,
    min_chars: int = 1,
    dedup_exact: bool = True,
    balance: str | None = None,
    cap: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sous-échantillonne les ideas (générique, déclaratif).

    1. `min_chars` : retire les avis trop courts.
    2. `dedup_exact` : collapse les textes IDENTIQUES (cumule le poids — aucune
       voix perdue), pour ne pas embedder deux fois la même chaîne.
    3. `balance` (ex. "lang") + `cap` : échantillon ÉQUILIBRÉ par valeur du champ
       jusqu'à `cap` (quota ≈ cap / nb_valeurs, round-robin pour le reliquat).
       Sans `balance`, simple troncature aléatoire à `cap`. Déterministe (seed).
    """
    rng = random.Random(seed)

    # 1) min_chars
    if min_chars:
        ideas = [i for i in ideas if len(_idea_clean(i).strip()) >= min_chars]

    # 2) dédup exacte (cumule le poids du doublon sur le représentant gardé)
    if dedup_exact:
        seen: dict[str, dict] = {}
        for i in ideas:
            key = _idea_clean(i).strip()
            if key in seen:
                seen[key]["props"]["weight"] = seen[key]["props"].get("weight", 1.0) + 1.0
            else:
                seen[key] = i
        ideas = list(seen.values())

    # 3) échantillon (équilibré par champ, puis cap)
    if balance:
        groups: dict[str, list[dict]] = {}
        for i in ideas:
            groups.setdefault(i["props"].get(balance, "") or "?", []).append(i)
        for g in groups.values():
            rng.shuffle(g)
        if cap is None:
            ideas = [i for g in groups.values() for i in g]
        else:
            # Round-robin entre les groupes → équilibre par valeur, jusqu'au cap.
            order = sorted(groups)  # ordre stable
            picked: list[dict] = []
            cursors = {k: 0 for k in order}
            while len(picked) < cap and any(cursors[k] < len(groups[k]) for k in order):
                for k in order:
                    if cursors[k] < len(groups[k]):
                        picked.append(groups[k][cursors[k]])
                        cursors[k] += 1
                        if len(picked) >= cap:
                            break
            ideas = picked
    elif cap is not None and len(ideas) > cap:
        ideas = rng.sample(ideas, cap)

    return ideas


def build_cache(
    dataset: str = "tiktok",
    *,
    descriptor: str | None = None,
    model: str = "nomic-v2",
    min_chars: int = 1,
    dedup_exact: bool = True,
    balance: str | None = None,
    cap: int | None = None,
    label: str | None = None,
    seed: int = 42,
) -> dict:
    _t0_build = perf_counter()
    desc = resolve_descriptor(dataset, descriptor)

    # Télécharge la source si absente (idempotent ; ne bloque pas si offline).
    if not desc.resolved_path().exists() and desc.url:
        print(f"Source {dataset} absente — tentative de téléchargement…")
        download.main(["--only", desc.name])

    # Lecture générique → Idea canonique (nettoyage + langue + anonymisation).
    raw = list(read_generic(desc))
    n_loaded = len(raw)
    ideas = [idea for rec in raw if (idea := to_idea(rec)) is not None]
    if not ideas:
        raise SystemExit(
            f"Aucun avis lisible pour {dataset} (source absente ou descripteur ?)."
        )

    # Deux temps : filtres de VALIDITÉ (min_chars + dédup) d'abord → n_responses = voix
    # réelles à la question dans le CORPUS ENTIER (sémantique stable, capé ou non) ; puis
    # l'échantillonnage (balance/cap) qui ne change pas ce dénominateur.
    ideas = subset(ideas, min_chars=min_chars, dedup_exact=dedup_exact,
                   balance=None, cap=None, seed=seed)
    n_responses = int(round(sum(i["props"].get("weight", 1.0) or 1.0 for i in ideas)))
    ideas = subset(ideas, min_chars=0, dedup_exact=False,
                   balance=balance, cap=cap, seed=seed)
    if not ideas:
        raise SystemExit("Aucun avis après sous-échantillonnage (filtres trop stricts ?).")

    # SEUL appel au modèle torch de tout le système live.
    embedder = Embedder(model_id=model)
    texts = [_idea_clean(i) for i in ideas]
    print(f"Embedding {len(texts)} avis [{dataset}] avec {embedder.model_id} (~1 min)…")
    vecs = embedder.embed(texts).astype(np.float32)

    out_dir = CACHE_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / EMB_NAME, vecs)
    with open(out_dir / IDEAS_NAME, "w", encoding="utf-8") as fh:
        for idea in ideas:
            fh.write(json.dumps(idea, ensure_ascii=False) + "\n")

    # meta.json : ce que `GET /datasets` expose (langues/n/source), dérivé.
    from collections import Counter

    lang_counts = Counter(_idea_lang(i) for i in ideas if _idea_lang(i))
    src_counts = Counter(i["props"].get("source", dataset) for i in ideas)

    # LIBELLÉ : ne JAMAIS rétrograder celui d'un dataset déjà servi. Le repli sur l'id est
    # légitime pour un dataset neuf, mais un re-ingest sans `--label` avait ainsi écrasé
    # « Consultation TikTok (FR) » par « tiktok » — une régression muette, invisible tant
    # qu'on ne regarde pas l'UI. On relit donc le meta existant avant de replier.
    meta_path = out_dir / META_NAME
    previous = {}
    if meta_path.exists():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    inherited = previous.get("label")
    if inherited == dataset:      # un repli d'un run précédent n'est pas un libellé à garder
        inherited = None

    meta = {
        "id": dataset,
        "label": label or desc.extra.get("label") or inherited or dataset,
        "status": desc.status,
        "n_nodes": len(ideas),
        # Voix réelles à la question dans le CORPUS (pré-échantillonnage) — l'affichage
        # distingue participants (n_loaded) / réponses (n_responses) / textes analysés.
        "n_responses": n_responses,
        "languages": [lg for lg, _ in lang_counts.most_common()],
        "lang_counts": dict(lang_counts.most_common()),
        "source": src_counts.most_common(1)[0][0] if src_counts else dataset,
        "model_id": embedder.model_id,
        "dim": int(vecs.shape[1]),
        "n_loaded": n_loaded,
        "built_with": {
            "min_chars": min_chars,
            "dedup_exact": dedup_exact,
            "balance": balance,
            "cap": cap,
            "seed": seed,
        },
    }
    # QUESTION posée + CONTEXTE : recopiés du DESCRIPTEUR (source de vérité, générique).
    # Sans ça, ils n'existaient que comme modifs manuelles de meta.json, perdues à chaque
    # rebuild/reset (bug : la question cadre l'extraction v2 ET le titre de la page).
    for _k in ("question", "context", "official_baseline", "official_url"):
        if desc.extra.get(_k):
            meta[_k] = desc.extra[_k]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Durée réelle de l'ingestion+embedding (0 token LLM — c'est du calcul local, mais elle
    # compte dans la durée de traitement affichée honnêtement par l'overview).
    try:
        from backend import cost as _cost
        _cost.record_phase(dataset, "ingest_embed",
                           {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}},
                           duration_seconds=perf_counter() - _t0_build)
    except Exception:
        pass
    print(f"✓ {out_dir / EMB_NAME}  ({vecs.shape[0]}×{vecs.shape[1]} float32)")
    print(f"✓ {out_dir / IDEAS_NAME}  ({len(ideas)} avis)")
    print(f"✓ {out_dir / META_NAME}")
    print(f"  langues : {meta['lang_counts']}")
    print(f"  subset  : {n_loaded}→{len(ideas)} "
          f"(min_chars≥{min_chars}, dedup_exact={dedup_exact}, "
          f"balance={balance}, cap={cap})")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache d'embeddings nomic-v2 par dataset.")
    ap.add_argument("--dataset", default="tiktok", help="id du dataset = nom du descripteur")
    ap.add_argument("--descriptor", default=None, help="chemin descripteur explicite (override)")
    ap.add_argument("--model", default="nomic-v2", help="alias/model_id (défaut nomic-v2)")
    ap.add_argument("--min-chars", type=int, default=1, help="filtre avis trop courts")
    ap.add_argument("--no-dedup-exact", action="store_true", help="garder les textes identiques")
    ap.add_argument("--balance", default=None,
                    help="champ d'équilibrage de l'échantillon (ex. 'lang')")
    ap.add_argument("--cap", type=int, default=None, help="plafond d'avis (rendu fluide)")
    ap.add_argument("--label", default=None, help="libellé d'affichage (UI)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_cache(
        dataset=args.dataset,
        descriptor=args.descriptor,
        model=args.model,
        min_chars=args.min_chars,
        dedup_exact=not args.no_dedup_exact,
        balance=args.balance,
        cap=args.cap,
        label=args.label,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

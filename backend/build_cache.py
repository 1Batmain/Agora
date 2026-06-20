"""Construit le cache d'embeddings nomic-v2 (UN SEUL appel au modèle torch).

Le serveur live (`backend/server.py`) re-clusterise à partir de ce cache ; il ne
charge JAMAIS le modèle torch. On embedde donc ici, une fois, le **superset** des
avis TikTok/FR (`source=tiktok`, `lang=fr`, `min_chars≥1`) et on sauvegarde :

  - `backend/cache/embeddings.npy` : matrice (n, d) float32, L2-normalisée
  - `backend/cache/ideas.jsonl`    : un Idea par ligne, ALIGNÉ sur les vecteurs

À l'exécution, le serveur applique les filtres live (`min_chars`, `dedup`) sur ce
superset caché — sans jamais ré-embedder.

Usage :
    uv run --extra embed-contender python -m backend.build_cache [--model nomic-v2]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pipeline.cluster import io
from pipeline.embed.embedder import Embedder

CACHE_DIR = Path(__file__).resolve().parent / "cache"
EMB_PATH = CACHE_DIR / "embeddings.npy"
IDEAS_PATH = CACHE_DIR / "ideas.jsonl"


def build_cache(
    model: str = "nomic-v2",
    source: str = "tiktok",
    lang: str = "fr",
    min_chars: int = 1,
    input_path: str | None = None,
) -> dict:
    ideas = io.load_ideas(input_path)
    n_loaded = len(ideas)

    # Superset : on garde large (min_chars≥1) ; les filtres fins sont live.
    if source:
        ideas = [i for i in ideas if i.source == source]
    if lang:
        ideas = [i for i in ideas if i.lang == lang]
    if min_chars:
        ideas = [i for i in ideas if len((i.text_clean or i.text).strip()) >= min_chars]

    if not ideas:
        raise SystemExit("Aucun avis dans le superset (filtres trop stricts ?).")

    # SEUL appel au modèle torch de tout le système live.
    embedder = Embedder(model_id=model)
    texts = [i.text_clean or i.text for i in ideas]
    print(f"Embedding {len(texts)} avis avec {embedder.model_id} (peut prendre ~1 min)…")
    vecs = embedder.embed(texts).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, vecs)
    with open(IDEAS_PATH, "w", encoding="utf-8") as fh:
        for idea in ideas:
            fh.write(json.dumps(asdict(idea), ensure_ascii=False) + "\n")

    meta = {
        "model_id": embedder.model_id,
        "n_loaded": n_loaded,
        "n_cached": len(ideas),
        "dim": int(vecs.shape[1]),
        "source": source,
        "lang": lang,
        "min_chars": min_chars,
    }
    print(f"✓ {EMB_PATH}  ({vecs.shape[0]}×{vecs.shape[1]} float32)")
    print(f"✓ {IDEAS_PATH}  ({len(ideas)} avis)")
    print(f"  model_id : {meta['model_id']}")
    print(f"  superset : source={source} lang={lang} min_chars≥{min_chars} "
          f"({n_loaded}→{len(ideas)})")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache d'embeddings nomic-v2 (one-shot).")
    ap.add_argument("--model", default="nomic-v2", help="alias/model_id (défaut nomic-v2)")
    ap.add_argument("--source", default="tiktok")
    ap.add_argument("--lang", default="fr")
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--input", default=None, help="ideas.jsonl (sinon auto-résolu)")
    args = ap.parse_args()
    build_cache(
        model=args.model,
        source=args.source,
        lang=args.lang,
        min_chars=args.min_chars,
        input_path=args.input,
    )


if __name__ == "__main__":
    main()

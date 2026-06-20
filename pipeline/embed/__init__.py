"""Lane NLP — service d'embeddings in-process (sentence-transformers)."""

from pipeline.embed.embedder import Embedder, DEFAULT_MODEL_ID, embed

__all__ = ["Embedder", "DEFAULT_MODEL_ID", "embed"]

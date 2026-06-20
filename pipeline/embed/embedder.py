"""T-N1 · Service d'embeddings in-process.

Embeddings sémantiques robustes au paraphrasing via sentence-transformers,
modèle léger multilingue (FR) `intfloat/multilingual-e5-small`, CPU only.

API : `Embedder.embed(texts) -> np.ndarray` (batch ou single), vecteurs
L2-normalisés (le cosinus se calcule alors par produit scalaire). Le `model_id`
est traçable pour remplir le contrat `Embedding{idea_id, vector[d], model_id}`.

Les modèles e5 attendent un préfixe d'instruction : "query: " pour une requête,
"passage: " pour un document à indexer. Ici on encode des avis (documents) →
préfixe "passage:" par défaut.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np

DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
_E5_DOC_PREFIX = "passage: "
_E5_QUERY_PREFIX = "query: "


class Embedder:
    """Encodeur in-process, lazy-loadé (le modèle ST n'est chargé qu'au 1er appel)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 32,
        e5_prefix: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.e5_prefix = e5_prefix
        self._model = None  # chargé à la demande

    @property
    def model(self):
        if self._model is None:
            # Import paresseux : pas de coût torch tant qu'on n'encode rien.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def _prep(self, texts: list[str], prefix: str) -> list[str]:
        if not self.e5_prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def embed(
        self,
        texts: str | Iterable[str],
        is_query: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode un texte ou une liste de textes → matrice (n, d) float32.

        Vecteurs L2-normalisés par défaut (cosine = dot product).
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)

        prefix = _E5_QUERY_PREFIX if is_query else _E5_DOC_PREFIX
        prepared = self._prep(items, prefix)
        vecs = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float32)
        return vecs[0] if single else vecs

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def benchmark(self, texts: list[str]) -> dict:
        """Mesure latence/throughput sur un échantillon (accept T-N1)."""
        t0 = time.perf_counter()
        vecs = self.embed(texts)
        dt = time.perf_counter() - t0
        n = len(texts)
        return {
            "model_id": self.model_id,
            "n": n,
            "dim": int(vecs.shape[1]) if n else self.dim,
            "seconds": round(dt, 4),
            "throughput_per_s": round(n / dt, 2) if dt > 0 else None,
            "latency_ms_per_text": round(1000 * dt / n, 3) if n else None,
        }


def embed(texts: str | Iterable[str], model_id: str = DEFAULT_MODEL_ID) -> np.ndarray:
    """Helper one-shot : `embed(texts) -> vectors`."""
    return Embedder(model_id=model_id).embed(texts)


if __name__ == "__main__":
    # Petit smoke-test / benchmark CLI : python -m pipeline.embed.embedder
    sample = [
        "Développer les pistes cyclables en ville.",
        "Investir dans les énergies renouvelables.",
        "Réduire les effectifs par classe à l'école.",
    ]
    emb = Embedder()
    bench = emb.benchmark(sample)
    print("benchmark:", bench)

"""Client d'embeddings pour requêter des API compatibles OpenAI (LMStudio, Nvidia NIM, etc.)."""

from __future__ import annotations

import time
from typing import Iterable

import httpx
import numpy as np


class APIEmbedder:
    """Encodeur distant via requêtes HTTP POST (compatible OpenAI /v1/embeddings)."""

    def __init__(
        self,
        url: str,
        model_path: str,
        api_key: str = "",
        batch_size: int = 32,
    ) -> None:
        """Initialise le client API pour le modèle d'embedding.
        
        Args:
            url: Endpoint de l'API (ex: 'http://localhost:1234/v1/embeddings').
            model_path: Nom ou identifiant du modèle côté serveur.
            api_key: Clé API si requise (vide pour LMStudio local).
            batch_size: Taille du batch de requêtes envoyées à l'API.
        """
        self.url = url
        self.model_path = model_path
        self.api_key = api_key
        self.batch_size = batch_size
        self._dim = None

    def embed(
        self,
        texts: str | Iterable[str],
        is_query: bool = False,
        normalize: bool | None = None,
    ) -> np.ndarray:
        """Encode un texte ou une liste de textes via l'API → matrice (n, d) float32."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)

        do_normalize = True if normalize is None else normalize

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        all_vecs = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            payload = {
                "input": batch,
                "model": self.model_path,
            }
            try:
                resp = httpx.post(self.url, json=payload, headers=headers, timeout=120.0)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"Erreur API ({self.url}): {e}")

            data = resp.json()
            if "data" not in data:
                raise RuntimeError(f"Réponse invalide de l'API: {data}")

            # OpenAI specification says embeddings are returned in 'data' array
            # We sort by 'index' to ensure matching order.
            sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
            vecs = [x["embedding"] for x in sorted_data]
            all_vecs.extend(vecs)

        arr = np.array(all_vecs, dtype=np.float32)
        
        if do_normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            # Avoid division by zero
            arr = np.divide(arr, norms, out=np.zeros_like(arr), where=norms != 0)

        return arr[0] if single else arr

    @property
    def dim(self) -> int:
        """Retourne la dimension des vecteurs générés par ce modèle."""
        if self._dim is None:
            # Fait un appel minimal pour obtenir la dimension
            vec = self.embed("dimension check", normalize=False)
            self._dim = vec.shape[0]
        return self._dim

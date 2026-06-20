"""T-N1 · Service d'embeddings in-process, MULTILINGUE et pluggable.

Embeddings sémantiques robustes au paraphrasing via sentence-transformers, CPU.
Trois modèles multilingues derrière UNE interface, chacun avec SA convention de
préfixe (cf. `pipeline.embed.registry`) :

  - `intfloat/multilingual-e5-small` (défaut) : "passage: " / "query: "
  - `nomic-ai/nomic-embed-text-v2-moe`       : "search_document: " / "search_query: "
  - `BAAI/bge-m3`                            : aucun préfixe

API (inchangée, utilisée par cluster/eval) :
  `Embedder(model_id).embed(texts, is_query=False) -> np.ndarray`
Vecteurs L2-normalisés par défaut (cosine = produit scalaire). Le `model_id` est
traçable pour remplir le contrat `Embedding{idea_id, vector[d], model_id}`.

Chargement paresseux (aucun coût torch tant qu'on n'encode rien), CPU, un seul
modèle chargé par instance — n'instanciez pas les 3 modèles simultanément
(RAM ~7 Gi partagée).
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from pipeline.embed.registry import ModelSpec, get_spec, resolve_model_id

DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"


class Embedder:
    """Encodeur in-process, lazy-loadé (le modèle ST n'est chargé qu'au 1er appel).

    La convention de préfixe et les flags de chargement viennent du registre,
    sélectionnés par `model_id`. `use_prefix=False` désactive tout préfixe
    (comportement legacy / debug).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 32,
        use_prefix: bool = True,
        e5_prefix: bool | None = None,  # alias rétro-compat de use_prefix
    ) -> None:
        self.model_id = resolve_model_id(model_id)
        self.spec: ModelSpec = get_spec(model_id)
        self.device = device
        self.batch_size = batch_size
        # `e5_prefix` (ancien nom) reste accepté ; il pilote `use_prefix`.
        self.use_prefix = e5_prefix if e5_prefix is not None else use_prefix
        self._model = None  # chargé à la demande

    @property
    def model(self):
        if self._model is None:
            # Import paresseux : pas de coût torch tant qu'on n'encode rien.
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": self.device}
            if self.spec.trust_remote_code:
                kwargs["trust_remote_code"] = True
            self._model = SentenceTransformer(self.model_id, **kwargs)
        return self._model

    def _prep(self, texts: list[str], prefix: str) -> list[str]:
        if not self.use_prefix or not prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def embed(
        self,
        texts: str | Iterable[str],
        is_query: bool = False,
        normalize: bool | None = None,
    ) -> np.ndarray:
        """Encode un texte ou une liste de textes → matrice (n, d) float32.

        `is_query` choisit le préfixe (requête vs document) selon le modèle.
        Vecteurs L2-normalisés par défaut (cosine = produit scalaire) ; passez
        `normalize=False` pour des vecteurs bruts.
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.empty((0, self.dim), dtype=np.float32)

        do_normalize = self.spec.normalize if normalize is None else normalize
        prepared = self._prep(items, self.spec.prefix(is_query))
        vecs = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=do_normalize,
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


# --------------------------------------------------------------------------- #
# Smoke-test multilingue : cosinus cross-lingue de paraphrases (FR/DE/EN).
# --------------------------------------------------------------------------- #

# Mêmes idées exprimées en 3 langues — colonnes = paraphrases inter-langues.
_PARALLEL = {
    "bike_lanes": {
        "fr": "il faut plus de pistes cyclables",
        "de": "wir brauchen mehr Radwege",
        "en": "we need more bike lanes",
    },
    "renewables": {
        "fr": "investir dans les énergies renouvelables",
        "de": "in erneuerbare Energien investieren",
        "en": "invest in renewable energy",
    },
    "class_size": {
        "fr": "réduire le nombre d'élèves par classe",
        "de": "die Klassengröße verringern",
        "en": "reduce the number of students per class",
    },
}


def _smoke(model_id: str) -> None:
    """Encode l'échantillon parallèle multilingue et imprime dim + cosinus
    cross-lingue moyen des paraphrases (haut = bon multilingue) vs cosinus
    inter-thèmes (bas = bonne séparation). C'est le signal qui nous intéresse.
    """
    emb = Embedder(model_id=model_id)
    concepts = list(_PARALLEL.keys())
    langs = ("fr", "de", "en")

    texts, idx = [], {}
    for c in concepts:
        for lg in langs:
            idx[(c, lg)] = len(texts)
            texts.append(_PARALLEL[c][lg])

    t0 = time.perf_counter()
    vecs = emb.embed(texts)  # documents, L2-normalisés → cosine = dot
    dt = time.perf_counter() - t0

    def cos(a, b):
        return float(np.dot(vecs[a], vecs[b]))

    # Cosinus cross-lingue : paraphrases du MÊME concept, langues différentes.
    same, cross = [], []
    pairs = [("fr", "de"), ("fr", "en"), ("de", "en")]
    for c in concepts:
        for la, lb in pairs:
            same.append(cos(idx[(c, la)], idx[(c, lb)]))
    # Cosinus inter-thèmes : concepts différents (toutes langues confondues).
    for i, ci in enumerate(concepts):
        for cj in concepts[i + 1 :]:
            for la in langs:
                for lb in langs:
                    cross.append(cos(idx[(ci, la)], idx[(cj, lb)]))

    print(f"model_id           : {emb.model_id}")
    print(f"spec               : doc_prefix={emb.spec.doc_prefix!r} "
          f"query_prefix={emb.spec.query_prefix!r} "
          f"trust_remote_code={emb.spec.trust_remote_code}")
    print(f"dim                : {int(vecs.shape[1])}")
    print(f"n_textes           : {len(texts)} ({len(concepts)} concepts × {len(langs)} langues)")
    print(f"encode             : {dt:.3f}s ({1000*dt/len(texts):.1f} ms/texte)")
    print(f"cos cross-lingue   : {np.mean(same):.3f}  (paraphrases — VEUT être ÉLEVÉ)")
    print(f"cos inter-thèmes   : {np.mean(cross):.3f}  (concepts ≠ — VEUT être bas)")
    print(f"marge (séparation) : {np.mean(same) - np.mean(cross):+.3f}  (positif = bon)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedder multilingue — smoke & bench.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help="model_id ou alias (e5 | nomic | bge-m3). Défaut: e5-small.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke multilingue : dim + cosinus cross-lingue des paraphrases FR/DE/EN.",
    )
    args = parser.parse_args()

    if args.smoke:
        _smoke(args.model)
    else:
        sample = [
            "Développer les pistes cyclables en ville.",
            "Investir dans les énergies renouvelables.",
            "Réduire les effectifs par classe à l'école.",
        ]
        emb = Embedder(model_id=args.model)
        print("benchmark:", emb.benchmark(sample))

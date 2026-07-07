"""Registre des modèles d'embeddings multilingues pluggables.

Le multilingue est une contrainte de 1er ordre : on veut regrouper par THÈME,
pas par langue. Chaque modèle a SA convention de préfixe — un mauvais préfixe
détruit silencieusement la qualité. Ce registre encapsule, par `model_id`, la
convention (préfixe doc/query), les flags de chargement (`trust_remote_code`)
et la normalisation par défaut.

Ajouter un contender = ajouter une `ModelSpec` ici (aucun autre changement).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Convention d'un modèle d'embedding.

    - `doc_prefix` / `query_prefix` : préfixes d'instruction prepend-és aux
      textes selon qu'on encode un document (à indexer) ou une requête.
      Chaîne vide = aucun préfixe (ex. bge-m3).
    - `trust_remote_code` : nécessaire pour les modèles à code custom (nomic).
    - `revision` : commit/branche HF ÉPINGLÉ. OBLIGATOIRE en pratique quand
      `trust_remote_code=True` (sécurité : on exécute du code distant au chargement) —
      sans pin, un push amont change le code exécuté sous nos pieds. `None` = `main`.
    - `normalize` : L2-normalisation par défaut (cosine = produit scalaire).
    """

    model_id: str
    doc_prefix: str = ""
    query_prefix: str = ""
    trust_remote_code: bool = False
    revision: str | None = None
    normalize: bool = True
    note: str = ""

    def prefix(self, is_query: bool) -> str:
        return self.query_prefix if is_query else self.doc_prefix


# Convention de préfixe par modèle — CŒUR du registre. Cf. README.
REGISTRY: dict[str, ModelSpec] = {
    # Baseline déjà en place. e5 EXIGE les préfixes "passage:"/"query:".
    "intfloat/multilingual-e5-small": ModelSpec(
        model_id="intfloat/multilingual-e5-small",
        doc_prefix="passage: ",
        query_prefix="query: ",
        trust_remote_code=False,
        normalize=True,
        note="baseline légère, multilingue (FR/DE/EN…), CPU rapide.",
    ),
    # MoE multilingue ; code custom => trust_remote_code + dépend de `einops`.
    # Matryoshka : on garde la dim native (pas de troncature).
    "nomic-ai/nomic-embed-text-v2-moe": ModelSpec(
        model_id="nomic-ai/nomic-embed-text-v2-moe",
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
        trust_remote_code=True,
        # Révision ÉPINGLÉE : le modèle charge du code custom (trust_remote_code) →
        # on fige le commit exécuté. C'est exactement le snapshot déjà en cache/prod
        # (refs/main au moment du build), donc embeddings INCHANGÉS, mais plus aucun
        # push amont ne peut altérer le code exécuté chez nous.
        revision="1066b6599d099fbb93dfcb64f9c37a7c9e503e85",
        normalize=True,
        note="MoE multilingue ; requiert einops ; matryoshka (dim native gardée).",
    ),
    # Contender R&D (bench JINA, 2026-07-07). jina-embeddings-v2-base-de :
    # Apache-2.0 (déployable) MAIS BILINGUE DE-EN seulement — pas de FR/IT natif.
    # Le flagship multilingue de Jina (jina-embeddings-v3, FR/DE/IT) est CC-BY-NC-4.0
    # (NON-COMMERCIAL) → rédhibitoire pour Agora, donc NON benché. Code custom
    # (JinaBERT/ALiBi) => trust_remote_code + révision épinglée. Aucun préfixe.
    "jinaai/jina-embeddings-v2-base-de": ModelSpec(
        model_id="jinaai/jina-embeddings-v2-base-de",
        doc_prefix="",
        query_prefix="",
        trust_remote_code=True,
        revision="3f9eede875721714945b6a99a3198299243cf2be",
        normalize=True,
        note="Apache-2.0 mais bilingue DE-EN (pas FR/IT) ; JinaBERT/ALiBi, trust_remote_code.",
    ),
    # Multilingue fort. AUCUN préfixe — en ajouter dégraderait la qualité.
    "BAAI/bge-m3": ModelSpec(
        model_id="BAAI/bge-m3",
        doc_prefix="",
        query_prefix="",
        trust_remote_code=False,
        normalize=True,
        note="dense vectors via sentence-transformers ; pas de préfixe.",
    ),
}

# Alias courts pratiques pour la CLI / la sélection.
ALIASES: dict[str, str] = {
    "e5": "intfloat/multilingual-e5-small",
    "e5-small": "intfloat/multilingual-e5-small",
    "nomic": "nomic-ai/nomic-embed-text-v2-moe",
    "nomic-v2": "nomic-ai/nomic-embed-text-v2-moe",
    "bge-m3": "BAAI/bge-m3",
    "bge": "BAAI/bge-m3",
    "jina": "jinaai/jina-embeddings-v2-base-de",
    "jina-v2-de": "jinaai/jina-embeddings-v2-base-de",
}


def resolve_model_id(name: str) -> str:
    """Résout un alias court vers un `model_id` canonique (ou le renvoie tel quel)."""
    return ALIASES.get(name, name)


def get_spec(model_id: str) -> ModelSpec:
    """Spec du modèle. Modèle inconnu → spec sûre SANS préfixe.

    On préfère « aucun préfixe » à « mauvais préfixe » : un préfixe erroné
    dégrade silencieusement, tandis que l'absence de préfixe est neutre.
    """
    model_id = resolve_model_id(model_id)
    spec = REGISTRY.get(model_id)
    if spec is not None:
        return spec
    return ModelSpec(model_id=model_id, note="inconnu du registre — aucun préfixe.")


def list_models() -> list[ModelSpec]:
    return list(REGISTRY.values())

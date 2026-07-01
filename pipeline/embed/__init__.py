"""Lane NLP — service d'embeddings in-process multilingue (sentence-transformers).

Registre de modèles pluggables, chacun avec sa convention de préfixe :
voir `pipeline.embed.registry` et `pipeline/embed/README.md`.
"""

from pipeline.embed.embedder import Embedder, DEFAULT_MODEL_ID, embed
from pipeline.embed.registry import (
    REGISTRY,
    ModelSpec,
    get_spec,
    list_models,
    resolve_model_id,
)

__all__ = [
    "Embedder",
    "DEFAULT_MODEL_ID",
    "embed",
    "REGISTRY",
    "ModelSpec",
    "get_spec",
    "list_models",
    "resolve_model_id",
]

"""`pipeline.claims` — pipeline ouvert avis → claims atomiques → thèmes ÉMERGENTS.

Aucune taxonomie imposée : les thèmes émergent du clustering des claims (style
TalkToTheCity). L'extraction a plusieurs backends (`pipeline.claims.backend`) :
API Mistral `ministral-3b-latest` par DÉFAUT, Mac Ollama `ministral-3` souverain en
opt-in (`mac`/`auto`). Réutilise `pipeline.embed` / `pipeline.cluster`.
"""

from pipeline.claims.backend import (
    ApiBackend,
    BackendUnavailable,
    ClaimBackend,
    MacBackend,
    resolve_backend,
)
from pipeline.claims.extract import (
    batch_claim_prompt,
    claim_prompt,
    extract_claims,
    parse_batch_claims,
    parse_claims,
)
from pipeline.claims.ollama import OllamaClient, OllamaStats
from pipeline.claims.pipeline import (
    Avis,
    Theme,
    as_avis,
    cluster_claims,
    embed_claim_texts,
    run_claims,
)

__all__ = [
    "ApiBackend",
    "Avis",
    "BackendUnavailable",
    "ClaimBackend",
    "MacBackend",
    "OllamaClient",
    "OllamaStats",
    "Theme",
    "as_avis",
    "batch_claim_prompt",
    "claim_prompt",
    "cluster_claims",
    "embed_claim_texts",
    "extract_claims",
    "parse_batch_claims",
    "parse_claims",
    "resolve_backend",
    "run_claims",
]

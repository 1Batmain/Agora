"""`pipeline.claims` — pipeline ouvert avis → claims atomiques → thèmes ÉMERGENTS.

Aucune taxonomie imposée : les thèmes émergent du clustering des claims (style
TalkToTheCity). Souverain : l'extraction tourne sur un LLM LOCAL (ministral sur
le Mac via `AGORA_OLLAMA_URL`). Réutilise `pipeline.embed` / `pipeline.cluster`.
"""

from pipeline.claims.extract import claim_prompt, extract_claims, parse_claims
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
    "Avis",
    "Theme",
    "OllamaClient",
    "OllamaStats",
    "as_avis",
    "claim_prompt",
    "cluster_claims",
    "embed_claim_texts",
    "extract_claims",
    "parse_claims",
    "run_claims",
]

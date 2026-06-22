"""Extraction des CLAIMS atomiques d'un avis (style TalkToTheCity).

Pour CHAQUE avis, un LLM local (ministral sur le Mac) extrait ses idées
distinctes — préoccupations / opinions / propositions autoportantes —, chacune
reformulée en UNE assertion atomique en vocabulaire LIBRE. Aucune taxonomie, rien
de codé en dur : tout émerge des données (généricité).

C'est l'étape LENTE (~2 s/avis) du pipeline. Elle est isolée ici pour que le
backend puisse la CACHER par dataset et rejouer le clustering/résolution sans
ré-extraire.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pipeline.claims.ollama import OllamaStats, parse_json_object

if TYPE_CHECKING:
    from pipeline.claims.backend import ClaimBackend

# Prompt OUVERT : on demande des assertions atomiques, sans catégorie ni étiquette.
CLAIM_SYS = (
    "Tu es un analyste d'avis citoyens, multilingue (FR, DE, IT, EN…). On te donne UN "
    "avis. Extrais ses IDÉES distinctes : chaque préoccupation, opinion ou proposition "
    "autoportante, reformulée en UNE assertion atomique, concise et compréhensible HORS "
    "contexte. Une idée = une assertion. N'invente rien, n'ajoute aucune catégorie ni "
    "étiquette ; reste fidèle à l'avis. Si l'avis ne porte qu'une idée, renvoie une seule "
    'assertion. Réponds STRICTEMENT en JSON : {"claims": ["assertion 1", "assertion 2", …]}.'
)


def claim_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": CLAIM_SYS},
            {"role": "user", "content": "Avis :\n" + text}]


def parse_claims(raw: str | None) -> list[str]:
    """Parse la réponse LLM → liste de claims (tolère une clé renommée)."""
    obj = parse_json_object(raw or "")
    if obj is None:
        return []
    val = obj.get("claims")
    if not isinstance(val, list):                 # le petit modèle a renommé la clé
        for v in obj.values():
            if isinstance(v, list):
                val = v
                break
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if str(x).strip()]


def extract_claims(
    avis: list,
    *,
    backend: "ClaimBackend",
    stats: OllamaStats | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[str]]:
    """Extrait les claims de chaque avis → ``{avis_id: [claim, ...]}``.

    `avis` : liste d'objets portant ``.id`` et ``.text`` (cf. `pipeline.claims.pipeline.Avis`).
    `backend` : un `ClaimBackend` (API Mistral, Mac Ollama…) — le prompt et le parsing
    sont IDENTIQUES quel que soit le backend, donc des claims au format identique. Repli :
    un avis dont l'extraction échoue devient 1 claim = son texte entier (jamais perdu).
    `progress(i, n)` est appelé pour le suivi.
    """
    stats = stats if stats is not None else OllamaStats()
    out: dict[str, list[str]] = {}
    n = len(avis)
    for i, a in enumerate(avis):
        raw = backend.complete(claim_prompt(a.text), stats=stats)
        out[a.id] = parse_claims(raw) or [a.text]
        if progress is not None:
            progress(i + 1, n)
    return out

"""Extraction EXTRACTIVE des CLAIMS d'un avis (verbatim, zéro hallucination).

Pour CHAQUE avis, un LLM **SÉLECTIONNE** ses idées distinctes — préoccupations /
opinions / propositions — en RECOPIANT des **portions verbatim** de l'avis (jamais
de reformulation). Chaque portion est ensuite VALIDÉE comme sous-chaîne exacte de
l'avis (`pipeline.claims.span`) et ancrée par ses offsets → `Claim{text,start,end}`.
Garantie : aucun mot hors de l'avis ne peut apparaître dans un claim. Aucune
taxonomie, rien de codé en dur : tout émerge des données (généricité).

C'est l'étape LENTE (~2 s/avis) du pipeline. Elle est isolée ici pour que le
backend puisse la CACHER par dataset et rejouer le clustering/résolution sans
ré-extraire.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pipeline.claims.ollama import OllamaStats, parse_json_object
from pipeline.claims.span import Claim, align_spans, whole_avis_claim

if TYPE_CHECKING:
    from pipeline.claims.backend import ClaimBackend

# Prompt EXTRACTIF : on demande de RECOPIER des portions de l'avis, mot pour mot.
# Aucune reformulation, aucune catégorie : juste sélectionner et coller des extraits.
# Quatre exigences (sélectivité · regroupement · sujet+position · verbatim strict),
# illustrées par des few-shots tirés de cas réels (cf. queue C1/C2). Rien de codé en
# dur sur un corpus : les exemples illustrent des PRINCIPES, pas des thèmes attendus.
CLAIM_SYS = (
    "Tu es un analyste d'avis citoyens, multilingue (FR, DE, IT, EN…). On te donne UN "
    "avis. Tu en extrais les CLAIMS : ses idées de FOND distinctes — chaque grief, "
    "opinion ou proposition du citoyen. Tu RECOPIES chaque claim MOT POUR MOT depuis "
    "l'avis (sous-chaîne EXACTE : mêmes mots, même orthographe, même ponctuation, fautes "
    "comprises) ; tu ne reformules RIEN, n'ajoutes RIEN, ne corriges RIEN. Chaque claim "
    "est un extrait CONTIGU de l'avis.\n"
    "\n"
    "RÈGLES :\n"
    "1. SÉLECTIVITÉ — n'extrais que la SUBSTANCE. Laisse de côté le cadrage, le narratif "
    "et les annonces qui ne portent aucune idée par eux-mêmes (« pour illustrer… », « mes "
    "doléances sont triples : », « je voudrais dire que… », politesses, anecdote de "
    "contexte). Si un passage n'exprime ni grief, ni opinion, ni proposition, ne "
    "l'extrais pas.\n"
    "2. REGROUPEMENT — ne FRAGMENTE pas une même idée. Restent DANS UN SEUL claim : un "
    "contraste (« X et non Y »), une justification (« … parce que … »), une condition "
    "(« si …, alors … ») et une énumération qui DÉTAILLE une seule idée (« que ce soit X "
    "comme Y sur Z »). Ne sépare que des idées RÉELLEMENT distinctes.\n"
    "3. SUJET + POSITION — chaque claim doit, à lui seul, dire SUR QUOI porte l'idée ET "
    "la POSITION du citoyen dessus. Choisis la portion qui contient les DEUX ; un "
    "fragment qui ampute le sujet ou la position est inutilisable.\n"
    "4. VERBATIM — chaque claim est une sous-chaîne EXACTE de l'avis. En cas de doute, "
    "recopie un peu plus de contexte plutôt que d'altérer le texte.\n"
    "\n"
    "EXEMPLES (ils illustrent le REGROUPEMENT et le critère SUJET+POSITION) :\n"
    "• « Avoir des élus qui représentent l'intérêt des citoyens et non l'intérêt de ceux "
    "qui ont financé leur campagne » → UN seul claim (le contraste « … et non … » est UNE "
    "idée, on ne la coupe pas).\n"
    "• « Plus de respect, d'honnêteté de la part des élus » → UN seul claim (l'énumération "
    "détaille une même demande).\n"
    "• Une énumération « que ce soit X comme Y sur Z » qui précise une seule demande → UN "
    "seul claim.\n"
    "\n"
    "Si l'avis ne porte qu'une idée, renvoie un seul claim. S'il n'en porte AUCUNE (pur "
    "narratif/cadrage), renvoie une liste vide. Réponds STRICTEMENT en JSON : "
    '{"claims": ["extrait verbatim 1", "extrait verbatim 2", …]}.'
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
) -> dict[str, list[Claim]]:
    """Extrait les claims VERBATIM de chaque avis → ``{avis_id: [Claim, ...]}``.

    `avis` : liste d'objets portant ``.id`` et ``.text`` (cf. `pipeline.claims.pipeline.Avis`).
    `backend` : un `ClaimBackend` (API Mistral, Mac Ollama…) — le prompt et le parsing
    sont IDENTIQUES quel que soit le backend, donc des claims au format identique.

    Chaque portion renvoyée par le LLM est ANCRÉE comme sous-chaîne exacte de l'avis
    (`align_spans`) : les portions non retrouvées sont rejetées (zéro hallucination).
    Repli : un avis dont AUCUNE portion ne s'ancre devient 1 claim = son texte entier
    (jamais perdu, et trivialement verbatim). `progress(i, n)` est appelé pour le suivi.
    """
    stats = stats if stats is not None else OllamaStats()
    out: dict[str, list[Claim]] = {}
    n = len(avis)
    for i, a in enumerate(avis):
        raw = backend.complete(claim_prompt(a.text), stats=stats)
        claims = align_spans(a.text, parse_claims(raw))
        out[a.id] = claims or [whole_avis_claim(a.text)]
        if progress is not None:
            progress(i + 1, n)
    return out

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
    "opinion ou proposition du citoyen. Tu RECOPIES chaque portion MOT POUR MOT depuis "
    "l'avis (sous-chaîne EXACTE : mêmes mots, même orthographe, même ponctuation, fautes "
    "comprises) ; tu ne reformules RIEN, n'ajoutes RIEN, ne corriges RIEN.\n"
    "\n"
    "Chaque claim a DEUX champs :\n"
    "• `parts` : la/les portion(s) verbatim qui PORTENT l'idée. En général UNE seule "
    "portion contiguë. Mais si l'idée est répartie sur des passages NON-CONTIGUS de "
    "l'avis (p.ex. la phrase qui pose l'idée + la fin d'une phrase plus loin qui s'y "
    "réfère), mets CHAQUE morceau verbatim dans `parts` → ils forment UN seul claim. "
    "N'utilise PLUSIEURS parts QUE si les morceaux appartiennent vraiment à la même idée.\n"
    "• `target` : la CIBLE de l'idée — l'aspect précis dont parle le claim — recopiée "
    "VERBATIM depuis l'avis (une courte portion : « temps passé sur l'écran », « les "
    "vidéos », « la modération »…). C'est une portion DE L'AVIS, pas une étiquette que "
    "tu inventes ou normalises. Si aucune cible nette ne se dégage, mets `target` à null.\n"
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
    "3. SUJET + POSITION — chaque claim doit, à lui seul, dire SUR QUOI porte l'idée "
    "(la `target`) ET la POSITION du citoyen dessus. Choisis des `parts` qui contiennent "
    "les DEUX ; un fragment qui ampute le sujet ou la position est inutilisable.\n"
    "4. VERBATIM — chaque part ET la target sont des sous-chaînes EXACTES de l'avis. En "
    "cas de doute, recopie un peu plus de contexte plutôt que d'altérer le texte.\n"
    "\n"
    "EXEMPLES (REGROUPEMENT, SUJET+POSITION, MULTI-PARTS, TARGET) :\n"
    "• « Avoir des élus qui représentent l'intérêt des citoyens et non l'intérêt de ceux "
    "qui ont financé leur campagne » → UN claim, parts=[toute la portion], target=« les "
    "élus » (le contraste « … et non … » est UNE idée, on ne la coupe pas).\n"
    "• « J'adore les vidéos courtes, elles me détendent » → parts=[« J'adore les vidéos "
    "courtes, elles me détendent »], target=« les vidéos courtes ».\n"
    "• « Le temps passé sur l'écran est trop long. […] et ça, ça me dégoûte » → si « ça » "
    "renvoie au temps d'écran : UN claim, parts=[« Le temps passé sur l'écran est trop "
    "long », « ça me dégoûte »], target=« temps passé sur l'écran ».\n"
    "\n"
    "Si l'avis ne porte qu'une idée, renvoie un seul claim. S'il n'en porte AUCUNE (pur "
    "narratif/cadrage), renvoie une liste vide. Réponds STRICTEMENT en JSON : "
    '{"claims": [{"parts": ["extrait verbatim 1", "extrait verbatim 2"], '
    '"target": "cible verbatim"}, …]}.'
)


def claim_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": CLAIM_SYS},
            {"role": "user", "content": "Avis :\n" + text}]


def parse_claims(raw: str | None) -> list[dict]:
    """Parse la réponse LLM → liste de specs `{parts:[...], target:str|None}`.

    Tolérant : clé `claims` renommée ; claim donné comme simple chaîne (legacy → une
    part, sans cible) ; `parts` donné comme chaîne unique ; clés alternatives
    (`text`/`claim`/`verbatim`) pour la portion. Une spec sans aucune part est écartée.
    """
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

    specs: list[dict] = []
    for item in val:
        if isinstance(item, str):                 # legacy : claim = chaîne unique
            t = item.strip()
            if t:
                specs.append({"parts": [t], "target": None})
            continue
        if not isinstance(item, dict):
            continue
        parts = item.get("parts")
        if isinstance(parts, str):
            parts = [parts]
        if not isinstance(parts, list):           # repli : clé alternative pour la portion
            single = item.get("text") or item.get("claim") or item.get("verbatim")
            parts = [single] if isinstance(single, str) else []
        parts = [str(p).strip() for p in parts if str(p).strip()]
        if not parts:
            continue
        target = item.get("target")
        target = str(target).strip() if isinstance(target, str) and str(target).strip() else None
        specs.append({"parts": parts, "target": target})
    return specs


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

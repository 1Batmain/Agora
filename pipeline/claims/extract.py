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

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from pipeline.claims.ollama import OllamaStats, parse_json_object
from pipeline.claims.span import Claim, align_spans, whole_avis_claim

if TYPE_CHECKING:
    from pipeline.claims.backend import ClaimBackend

# BATCHING : N avis par appel LLM (vitesse/coût). 1 avis/appel = 1604 appels sur tiktok
# (lent). On en groupe plusieurs, réponse JSON CLÉE par numéro d'avis, remap → claims par
# avis_id. La validation `align_spans` reste PAR AVIS (verbatim strict, zéro contamination
# inter-avis). Réglable par env ; 0/1 → chemin mono-avis historique.
BATCH_SIZE = int(os.environ.get("AGORA_CLAIMS_BATCH", "8"))
# Budget de sortie par avis dans un batch (claims d'un avis tiennent largement). Le total
# du batch = BATCH_TOKENS_PER_AVIS × taille, borné, pour ne pas tronquer le JOSN.
BATCH_TOKENS_PER_AVIS = int(os.environ.get("AGORA_CLAIMS_BATCH_TOKENS", "400"))
BATCH_TOKENS_CAP = int(os.environ.get("AGORA_CLAIMS_BATCH_TOKENS_CAP", "8192"))

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
    "• `target` : la CIBLE DE LA POSITION — l'OBJET / l'aspect PRÉCIS sur lequel le "
    "citoyen prend PARTI, ce sur quoi on peut être POUR ou CONTRE (« les vidéos », « le "
    "temps passé sur l'écran », « la modération », « l'algorithme »…). Recopiée VERBATIM "
    "depuis l'avis (courte portion DE L'AVIS, pas une étiquette que tu inventes ou "
    "normalises). CHAQUE claim DOIT porter une cible : un claim, c'est une POSITION SUR "
    "UN OBJET. Si tu n'arrives pas à pointer l'objet d'une prise de position claire, "
    "c'est que le passage est trop vague/narratif → NE L'EXTRAIS PAS (mieux vaut moins "
    "de claims que des claims sans cible). Ne mets `target` à null QU'en tout dernier "
    "recours, pour un grief réel mais dont l'objet reste implicite.\n"
    "\n"
    "RÈGLES :\n"
    "1. SÉLECTIVITÉ — n'extrais que la SUBSTANCE : une PRISE DE POSITION (grief, opinion, "
    "proposition) sur un objet identifiable. Laisse de côté le cadrage, le narratif et les "
    "annonces qui ne portent aucune position par eux-mêmes (« pour illustrer… », « mes "
    "doléances sont triples : », « je voudrais dire que… », politesses, anecdote de "
    "contexte). Si un passage n'exprime ni grief, ni opinion, ni proposition CIBLÉE, ne "
    "l'extrais pas.\n"
    "2. REGROUPEMENT — ne FRAGMENTE pas une même idée. Restent DANS UN SEUL claim : un "
    "contraste (« X et non Y »), une justification (« … parce que … »), une condition "
    "(« si …, alors … ») et une énumération qui DÉTAILLE une seule idée (« que ce soit X "
    "comme Y sur Z »). Ne sépare que des idées RÉELLEMENT distinctes.\n"
    "3. OBJET + POSITION — chaque claim doit, à lui seul, dire SUR QUOI le citoyen prend "
    "parti (la `target`) ET sa POSITION dessus (pour/contre, aime/déteste, demande/refuse). "
    "Choisis des `parts` qui contiennent les DEUX ; un fragment qui ampute l'objet ou la "
    "position est inutilisable.\n"
    "4. VERBATIM — chaque part ET la target sont des sous-chaînes EXACTES de l'avis. En "
    "cas de doute, recopie un peu plus de contexte plutôt que d'altérer le texte.\n"
    "\n"
    "EXEMPLES (POSITION SUR UN OBJET, REGROUPEMENT, MULTI-PARTS, TARGET) :\n"
    "• « j'aime les vidéos parce qu'elles me font rire » → UN claim, parts=[toute la "
    "portion], target=« les vidéos » (position POUR son objet « les vidéos »).\n"
    "• « le temps passé sur l'écran me dégoûte » → parts=[toute la portion], target=« le "
    "temps passé sur l'écran » (position CONTRE).\n"
    "• « Avoir des élus qui représentent l'intérêt des citoyens et non l'intérêt de ceux "
    "qui ont financé leur campagne » → UN claim, parts=[toute la portion], target=« les "
    "élus » (le contraste « … et non … » est UNE idée, on ne la coupe pas).\n"
    "• « Le temps passé sur l'écran est trop long. […] et ça, ça me dégoûte » → si « ça » "
    "renvoie au temps d'écran : UN claim, parts=[« Le temps passé sur l'écran est trop "
    "long », « ça me dégoûte »], target=« temps passé sur l'écran ».\n"
    "\n"
    "Si l'avis ne porte qu'une position, renvoie un seul claim. S'il n'en porte AUCUNE "
    "(pur narratif/cadrage, ou rien de ciblable), renvoie une liste vide. Réponds "
    'STRICTEMENT en JSON : {"claims": [{"parts": ["extrait verbatim 1", "extrait '
    'verbatim 2"], "target": "cible verbatim"}, …]}.'
)


def claim_prompt(text: str) -> list[dict]:
    return [{"role": "system", "content": CLAIM_SYS},
            {"role": "user", "content": "Avis :\n" + text}]


# Consigne BATCH ajoutée au system : plusieurs avis numérotés, réponse CLÉE par numéro.
# Les RÈGLES (sélectivité, regroupement, objet+position, verbatim) restent identiques —
# elles s'appliquent à CHAQUE avis indépendamment. Le verbatim est PAR AVIS : ne recopie
# que des portions de l'avis du numéro courant (jamais d'un autre).
BATCH_SYS_SUFFIX = (
    "\n\nMODE LOT : on te donne PLUSIEURS avis, numérotés (#1, #2, …). Traite CHACUN "
    "INDÉPENDAMMENT avec les règles ci-dessus. Pour chaque avis, les `parts` et la "
    "`target` doivent être des sous-chaînes EXACTES de CET avis-LÀ (jamais d'un autre). "
    "Réponds STRICTEMENT en JSON, un objet dont les CLÉS sont les NUMÉROS d'avis (chaîne) "
    'et les valeurs `{"claims": [...]}` : '
    '{"1": {"claims": [{"parts": ["…"], "target": "…"}]}, "2": {"claims": []}, …}. '
    "INCLUS une entrée pour CHAQUE numéro (liste vide si l'avis ne porte aucune position)."
)


def batch_claim_prompt(texts: list[str]) -> list[dict]:
    """Prompt pour un LOT d'avis : avis numérotés, réponse JSON clée par numéro (#1..#N)."""
    blocks = [f"=== AVIS #{i} ===\n{t}" for i, t in enumerate(texts, 1)]
    user = (
        f"Voici {len(texts)} avis numérotés. Extrais les claims de CHAQUE avis "
        "séparément, et réponds avec un objet clé par numéro.\n\n" + "\n\n".join(blocks)
    )
    return [{"role": "system", "content": CLAIM_SYS + BATCH_SYS_SUFFIX},
            {"role": "user", "content": user}]


def parse_batch_claims(raw: str | None, n: int) -> list[list[dict] | None]:
    """Parse la réponse d'un LOT → specs par avis (index 0..n-1), ou `None` si absent.

    Attendu : `{"1": {"claims":[...]}, "2": {...}, …}`. Tolérant aux variantes : valeur
    donnée directement comme liste de claims ; objet enveloppé sous une clé unique
    (`avis`/`results`/…) ; liste POSITIONNELLE `[{...}, {...}]`. Un avis absent ou non
    parsable reste `None` → l'appelant le repli en mono-avis (robustesse).
    """
    out: list[list[dict] | None] = [None] * n
    obj = parse_json_object(raw or "")
    if obj is None:
        return out

    # Déballe un éventuel conteneur : {"results": {...}} / {"avis": [...]} → la map/liste.
    keyed: dict | None = obj
    if not any(str(i) in obj for i in range(1, n + 1)):
        for v in obj.values():
            if isinstance(v, dict) and any(str(i) in v for i in range(1, n + 1)):
                keyed = v
                break
            if isinstance(v, list):               # liste positionnelle enveloppée
                keyed = {str(i): item for i, item in enumerate(v, 1)}
                break

    for i in range(1, n + 1):
        value = keyed.get(str(i)) if isinstance(keyed, dict) else None
        if value is None:
            continue
        if isinstance(value, dict):               # {"claims":[...]} (clé renommée tolérée)
            value = _claims_list(value)
        if isinstance(value, list):
            out[i - 1] = _normalize_specs(value)
    return out


def _normalize_specs(val) -> list[dict]:
    """Liste d'items LLM → specs `{parts:[...], target:str|None}` (cœur du parsing).

    Tolérant : claim donné comme simple chaîne (legacy → une part, sans cible) ; `parts`
    donné comme chaîne unique ; clés alternatives (`text`/`claim`/`verbatim`) pour la
    portion. Une spec sans aucune part est écartée.
    """
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


def _claims_list(obj: dict | None):
    """Extrait la liste de claims d'un objet `{claims:[...]}` (clé renommée tolérée)."""
    if not isinstance(obj, dict):
        return None
    val = obj.get("claims")
    if isinstance(val, list):
        return val
    for v in obj.values():                        # le petit modèle a renommé la clé
        if isinstance(v, list):
            return v
    return None


def parse_claims(raw: str | None) -> list[dict]:
    """Parse la réponse LLM (UN avis) → liste de specs `{parts:[...], target:str|None}`.

    Tolérant : clé `claims` renommée ; voir `_normalize_specs` pour les replis par item.
    """
    val = _claims_list(parse_json_object(raw or ""))
    return _normalize_specs(val) if val is not None else []


def _anchor(a, specs: list[dict]) -> list[Claim]:
    """Ancre des specs sur l'avis `a` → claims verbatim (repli : avis entier)."""
    claims = align_spans(a.text, specs)
    return claims or [whole_avis_claim(a.text)]


def _extract_single(a, *, backend: "ClaimBackend", stats: OllamaStats) -> list[Claim]:
    """Extraction MONO-AVIS (chemin historique + repli d'un avis raté en lot)."""
    raw = backend.complete(claim_prompt(a.text), stats=stats)
    return _anchor(a, parse_claims(raw))


def extract_claims(
    avis: list,
    *,
    backend: "ClaimBackend",
    stats: OllamaStats | None = None,
    progress: Callable[[int, int], None] | None = None,
    batch_size: int | None = None,
) -> dict[str, list[Claim]]:
    """Extrait les claims VERBATIM de chaque avis → ``{avis_id: [Claim, ...]}``.

    `avis` : liste d'objets portant ``.id`` et ``.text`` (cf. `pipeline.claims.pipeline.Avis`).
    `backend` : un `ClaimBackend` (API Mistral, Mac Ollama…) — le prompt et le parsing
    sont IDENTIQUES quel que soit le backend, donc des claims au format identique.

    BATCHING : on groupe `batch_size` avis par appel LLM (défaut `BATCH_SIZE`) — réponse
    JSON clée par numéro d'avis, remappée → claims par avis_id. `batch_size<=1` rejoue le
    chemin mono-avis. Robustesse : un avis ABSENT/mal formé dans la réponse d'un lot est
    ré-extrait SEUL (repli mono-avis) ; on ne perd jamais d'avis.

    Chaque portion renvoyée par le LLM est ANCRÉE comme sous-chaîne exacte de l'avis
    (`align_spans`, PAR AVIS) : les portions non retrouvées sont rejetées (zéro
    hallucination, zéro contamination inter-avis). Repli ultime : un avis dont AUCUNE
    portion ne s'ancre devient 1 claim = son texte entier. `progress(i, n)` suit l'avancée.
    """
    stats = stats if stats is not None else OllamaStats()
    bs = BATCH_SIZE if batch_size is None else batch_size
    out: dict[str, list[Claim]] = {}
    n = len(avis)
    done = 0

    for start in range(0, n, max(1, bs)):
        batch = avis[start:start + max(1, bs)]
        if bs <= 1:
            specs_by_idx: list[list[dict] | None] = [None] * len(batch)
        else:
            raw = backend.complete(
                batch_claim_prompt([a.text for a in batch]), stats=stats,
                max_tokens=min(BATCH_TOKENS_CAP, BATCH_TOKENS_PER_AVIS * len(batch)),
            )
            specs_by_idx = parse_batch_claims(raw, len(batch))

        for j, a in enumerate(batch):
            specs = specs_by_idx[j]
            # Avis absent / non parsable dans la réponse du lot → repli mono-avis robuste.
            out[a.id] = (_anchor(a, specs) if specs is not None
                         else _extract_single(a, backend=backend, stats=stats))
            done += 1
            if progress is not None:
                progress(done, n)
    return out

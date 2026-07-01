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
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# CONCURRENCE BORNÉE des lots : plusieurs appels LLM d'extraction en parallèle, jusqu'à
# cette borne (même env que l'enrichissement). La borne respecte le RPM (pas de tempête
# 429 ; le backoff par appel reste dans le backend). Les lots sont INDÉPENDANTS (verbatim
# PAR AVIS, remap par avis_id) → résultats IDENTIQUES au sériel, ordre indifférent.
LLM_MAX_WORKERS = max(1, int(os.environ.get("AGORA_LLM_MAX_WORKERS", "4")))

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
    "• `target` : un INDICE OPTIONNEL — l'OBJET / l'aspect sur lequel porte la position "
    "(« les vidéos », « le temps d'écran », « la fiscalité locale », « le mille-feuille "
    "administratif »…), recopié VERBATIM depuis l'avis. Mets-la SI une courte portion de "
    "l'avis pointe l'objet sans effort ; sinon `target=null`. NE JAMAIS écarter un claim "
    "de fond au prétexte que sa cible est diffuse ou implicite : une position réelle se "
    "garde toujours, cible ou pas. La cible n'est qu'un indice secondaire, pas un filtre.\n"
    "\n"
    "RÈGLES :\n"
    "1. COMPLÉTUDE (priorité) — un avis citoyen, surtout long, ARGUMENTE souvent sur "
    "PLUSIEURS thèmes distincts (p.ex. fiscalité ET démocratie ET services publics). "
    "Capture CHAQUE prise de position distincte de l'avis : n'en oublie AUCUNE, ne t'arrête "
    "pas à la première. Sépare les thèmes RÉELLEMENT distincts en claims distincts. "
    "Balaie l'avis du début à la fin.\n"
    "2. SÉLECTIVITÉ — n'extrais que la SUBSTANCE : une PRISE DE POSITION (grief, opinion, "
    "proposition). Laisse de côté le pur cadrage, le narratif et les annonces qui ne "
    "portent aucune position par eux-mêmes (« pour illustrer… », « mes doléances sont "
    "triples : », politesses, anecdote de contexte). Pas de bruit, pas de redite.\n"
    "3. REGROUPEMENT (anti-fragmentation, PRIORITAIRE) — ne FRAGMENTE pas une même prise "
    "de position. Restent DANS UN SEUL claim : un contraste (« X et non Y »), une "
    "justification (« … parce que … »), une condition (« si …, alors … »), une énumération "
    "qui DÉTAILLE une seule idée, ET SURTOUT :\n"
    "   – l'ÉNONCÉ D'UN PROBLÈME ET LA SOLUTION proposée pour le résoudre = UN SEUL claim "
    "(ne sépare JAMAIS le constat de la mesure qui y répond) ;\n"
    "   – PLUSIEURS phrases qui visent le MÊME sujet (p.ex. plusieurs mesures contre un même "
    "travers) = UN SEUL claim. Ne multiplie pas les claims sur une même prise de position.\n"
    "Sépare uniquement les sujets RÉELLEMENT distincts ; en cas de doute, REGROUPE.\n"
    "4. VERBATIM — chaque part ET la target sont des sous-chaînes EXACTES de l'avis. En "
    "cas de doute, recopie un peu plus de contexte plutôt que d'altérer le texte.\n"
    "\n"
    "EXEMPLES :\n"
    "• « j'aime les vidéos parce qu'elles me font rire » → UN claim, parts=[toute la "
    "portion], target=« les vidéos ».\n"
    "• « Avoir des élus qui représentent l'intérêt des citoyens et non l'intérêt de ceux "
    "qui ont financé leur campagne » → UN claim (le contraste « … et non … » est UNE idée), "
    "target=« les élus ».\n"
    "• « Que les élus se préoccupent plus de leurs administrés que de leur situation "
    "personnelle : le mandat unique est une réponse » → UN SEUL claim (problème + solution = "
    "une seule prise de position), JAMAIS deux.\n"
    "• « Devoir de présence aux débats pour les élus. […] Et sanction financière en cas "
    "d'absentéisme aux débats. » → UN SEUL claim (plusieurs phrases visent le MÊME sujet : "
    "l'absentéisme des élus), pas une par phrase.\n"
    "• Avis multi-thèmes « Il faut baisser les impôts. Par ailleurs trop d'élus, supprimons "
    "le Sénat. Et les services publics ruraux disparaissent. » → TROIS claims distincts "
    "(fiscalité / nombre d'élus / services publics ruraux), un par thème.\n"
    "• « Le temps passé sur l'écran est trop long. […] et ça, ça me dégoûte » → si « ça » "
    "renvoie au temps d'écran : UN claim, parts=[« Le temps passé sur l'écran est trop "
    "long », « ça me dégoûte »], target=« temps passé sur l'écran ».\n"
    "\n"
    "Si l'avis ne porte AUCUNE position (pur narratif/cadrage), renvoie une liste vide. "
    "Réponds STRICTEMENT en JSON : {\"claims\": [{\"parts\": [\"extrait verbatim 1\"], "
    "\"target\": \"cible verbatim ou null\"}, …]}."
)


def _question_frame(question: str | None) -> str:
    """Bloc CONTEXTE injectant la question globale de la consultation (cadre la granularité).

    Vide si pas de question (généricité : un dataset sans `question` retombe sur le prompt
    nu). La question dit au LLM ce que TOUS les avis tentent de répondre → des sous-points
    qui répondent à la MÊME facette de la question = UN claim (anti-sur-segmentation)."""
    q = (question or "").strip()
    if not q:
        return ""
    return (
        "\n\nCONTEXTE DE LA CONSULTATION — tous les avis répondent à UNE même question : "
        f"« {q} ». Cette question CADRE la granularité attendue : des sous-points qui "
        "répondent tous à la MÊME facette de cette question forment UN SEUL claim. N'ouvre "
        "un claim distinct que pour une facette RÉELLEMENT différente de la question."
    )


def claim_sys(question: str | None = None) -> str:
    """Prompt système d'extraction, éventuellement cadré par la question de consultation."""
    return CLAIM_SYS + _question_frame(question)


def claim_prompt(text: str, question: str | None = None) -> list[dict]:
    return [{"role": "system", "content": claim_sys(question)},
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


def batch_claim_prompt(texts: list[str], question: str | None = None) -> list[dict]:
    """Prompt pour un LOT d'avis : avis numérotés, réponse JSON clée par numéro (#1..#N)."""
    blocks = [f"=== AVIS #{i} ===\n{t}" for i, t in enumerate(texts, 1)]
    user = (
        f"Voici {len(texts)} avis numérotés. Extrais les claims de CHAQUE avis "
        "séparément, et réponds avec un objet clé par numéro.\n\n" + "\n\n".join(blocks)
    )
    return [{"role": "system", "content": claim_sys(question) + BATCH_SYS_SUFFIX},
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


def _extract_single(a, *, backend: "ClaimBackend", stats: OllamaStats,
                    question: str | None = None) -> list[Claim]:
    """Extraction MONO-AVIS (chemin historique + repli d'un avis raté en lot)."""
    raw = backend.complete(claim_prompt(a.text, question), stats=stats)
    return _anchor(a, parse_claims(raw))


def extract_claims(
    avis: list,
    *,
    backend: "ClaimBackend",
    stats: OllamaStats | None = None,
    progress: Callable[[int, int], None] | None = None,
    batch_size: int | None = None,
    question: str | None = None,
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
    step = max(1, bs)
    batches = [avis[start:start + step] for start in range(0, n, step)]

    def _process_batch(batch: list) -> dict[str, list[Claim]]:
        """Extrait les claims d'UN lot → ``{avis_id: [Claim, ...]}`` (verbatim PAR AVIS).

        Sans état partagé hors `stats`/`backend` (tous deux thread-safe en lecture/appel) :
        l'ancrage `align_spans` reste par avis et le repli mono-avis est intact.
        """
        if bs <= 1:
            specs_by_idx: list[list[dict] | None] = [None] * len(batch)
        else:
            raw = backend.complete(
                batch_claim_prompt([a.text for a in batch], question), stats=stats,
                max_tokens=min(BATCH_TOKENS_CAP, BATCH_TOKENS_PER_AVIS * len(batch)),
            )
            specs_by_idx = parse_batch_claims(raw, len(batch))

        local: dict[str, list[Claim]] = {}
        for j, a in enumerate(batch):
            specs = specs_by_idx[j]
            # Avis absent / non parsable dans la réponse du lot → repli mono-avis robuste.
            local[a.id] = (_anchor(a, specs) if specs is not None
                           else _extract_single(a, backend=backend, stats=stats,
                                                 question=question))
        return local

    # Séquentiel si un seul worker / un seul lot : chemin historique, déterminisme garanti.
    if LLM_MAX_WORKERS <= 1 or len(batches) <= 1:
        done = 0
        for batch in batches:
            for aid, claims in _process_batch(batch).items():
                out[aid] = claims
            done += len(batch)
            if progress is not None:
                progress(done, n)
        return out

    # CONCURRENCE BORNÉE : lots indépendants traités en parallèle (RPM respecté par la
    # borne + le backoff par appel). Le remap par avis_id rend l'ordre indifférent.
    done = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS,
                            thread_name_prefix="agora-extract") as ex:
        futures = {ex.submit(_process_batch, b): b for b in batches}
        for fut in as_completed(futures):
            local = fut.result()  # propage toute exception du worker
            with lock:
                out.update(local)
                done += len(futures[fut])
                d = done
            if progress is not None:
                progress(d, n)
    return out

"""Stance et objet de clivage — la SEULE voie vers les positions.

Extrait de `backend/build_opinion.py` pour être partageable entre le pipeline batch et le
pipeline live, sans que `live/` ait à importer `backend/` (isolation). Les prompts sont
repris **tels quels** : ils portent des calibrages validés au banc, qu'on ne réécrit pas au
passage (chaque consigne est annotée de son verdict).

## Pourquoi ce module existe

Mesuré sur x-stance : le clustering d'embeddings recouvre le clivage FAVOR/AGAINST à
**NMI ≈ 0,04–0,06**. *L'embedding capte le SUJET, pas la POSITION* — deux personnes en
désaccord frontal sont voisines dans l'espace. Toute position affichée par Agora passe donc
par un jugement LLM, jamais par la géométrie.
"""

from __future__ import annotations

import json
import time

from pipeline.cluster import mistral_client

# Claims par appel de stance. Compromis coût/robustesse : au-delà, une réponse tronquée
# fait retomber tout le lot en repli unitaire.
BATCH = 10

# Niveaux de confiance valides (auto-évaluation du modèle). Toute valeur absente/inconnue
# est normalisée en repli prudent `low` (on n'invente pas de certitude).
CONFIDENCE_LEVELS = {"high", "medium", "low"}

STANCES = {"favorable", "defavorable", "nuance"}


def cleavage_system(title: str) -> str:
    """Prompt cleavage CONDITIONNÉ sur le TITRE du thème (v2).

    v1 (sans titre, « la PLUS SAILLANTE ») faisait dériver la cible vers une FACETTE
    bruyante au lieu du centre du thème. v2 = deux leviers validés
    (research/cleavage_v2_note.md) :
      1. CONDITIONNER sur le titre — la proposition doit capturer le sujet de CE thème ;
      2. « CENTRAL » > « saillant » — résumer le débat du thème, pas le détail le plus bruyant.
    """
    return (
        "Tu es analyste de consultations citoyennes. On te donne le TITRE d'un THÈME, ses "
        "MOTS-CLÉS et des CONTRIBUTIONS verbatim. Identifie l'OBJET DE CLIVAGE qui RÉSUME "
        f"le débat CENTRAL de CE thème, intitulé « {title} » : la proposition ou mesure "
        "PRÉCISE, au cœur du thème, sur laquelle des citoyens peuvent être POUR ou CONTRE. "
        "Elle doit capturer le SUJET CENTRAL du thème (ce dont parle le titre), PAS une "
        "facette secondaire ni le détail le plus bruyant. Formule-la comme une proposition "
        "polaire COURTE (≤12 mots), neutre et débattable, à l'infinitif ou nominale — ex. "
        "« instaurer le référendum d'initiative citoyenne », « rendre le vote obligatoire », "
        "« réduire le nombre d'élus », « tirer au sort des citoyens pour légiférer ». "
        "Réponds en JSON strict : {\"objet\":\"<proposition>\",\"justif\":\"<≤14 mots>\"}."
    )


STANCE_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne UNE CIBLE — une PROPOSITION "
    "D'ACTION débattable (p. ex. « réguler l'usage d'un service », « instaurer une mesure ») "
    "— et des CONTRIBUTIONS citoyennes verbatim. Pour chaque contribution, classe si son "
    "auteur SOUTIENT ou S'OPPOSE À CETTE ACTION (et NON son sentiment envers le sujet) :\n"
    "  - \"favorable\"   : la contribution VA DANS LE SENS de l'action — elle la réclame, OU "
    "elle décrit un PROBLÈME/méfait que cette action viserait à corriger (décrire les dangers "
    "d'un sujet = soutenir une action pour le réguler/limiter) ;\n"
    "  - \"defavorable\" : la contribution S'OPPOSE à l'action — elle défend le sujet tel quel, "
    "juge l'action inutile/excessive/nuisible, ou refuse toute intervention ;\n"
    "  - \"nuance\"      : position ambivalente/conditionnelle, ou aucune position claire sur "
    "l'ACTION elle-même.\n"
    "ATTENTION — le piège à éviter : ne confonds JAMAIS un sentiment négatif ENVERS LE SUJET "
    "avec une opposition à l'action. Quelqu'un qui critique ou subit un problème est FAVORABLE "
    "à une action qui vise à le corriger. Juge la position sur l'ACTION, pas la tonalité.\n"
    # PERTINENCE (pré-filtre SOFT) — VALIDÉ research/relevance_calibration_note.md.
    "PERTINENCE (à vérifier D'ABORD) : ne classe \"favorable\"/\"defavorable\" QUE si la "
    "contribution PORTE sur CETTE action précise. Si elle vise une action CLAIREMENT DIFFÉRENTE "
    "(même thème général), ou reste purement descriptive/générale SANS implication pour cette "
    "action, classe \"nuance\" — ne lui prête pas une position par simple proximité de sujet.\n"
    # large_noabst — VALIDÉ research/stance_large_bench.md.
    "PRIORITÉ : parmi les contributions QUI PORTENT sur l'action, ne réserve \"nuance\" qu'à "
    "celles vraiment sans position — si une lecture raisonnable permet de trancher, TRANCHE.\n"
    "Pour CHAQUE contribution, indique aussi ta CONFIANCE : \"high\" (position explicite et "
    "nette), \"medium\" (probable mais indirecte), \"low\" (ambigu/hors-sujet — tu hésites). "
    "Réponds en JSON strict : {\"results\":[{\"i\":<int>,\"stance\":\"favorable|defavorable|"
    "nuance\",\"confidence\":\"high|medium|low\",\"justif\":\"<≤14 mots>\"}]}. Une entrée par "
    "contribution, dans l'ordre, rien d'autre."
)

# ⚠️ DISCOURS RAPPORTÉ — consigne AJOUTÉE pour le registre parlementaire, où citer la thèse
# adverse pour la réfuter est un procédé constant. Observé à la sonde : une ministre opposée
# à un texte s'est vu extraire « nous serions des voleurs de vie » — la thèse qu'elle
# démolissait — ce qui, classé tel quel, INVERSE sa position.
# ⚠️ NON VALIDÉE PAR UN BANC : c'est une réponse plausible, pas un verdict. Ne pas publier de
# position par orateur en s'appuyant dessus tant qu'elle n'est pas mesurée.
REPORTED_SPEECH_GUARD = (
    "\nDISCOURS RAPPORTÉ : si la contribution CITE la thèse d'un adversaire pour la contester "
    "(« on nous dit que… », « certains prétendent que… », citation suivie d'une réfutation), "
    "la position à classer est celle de L'AUTEUR, pas celle qu'il cite. Dans le doute sur "
    "l'attribution, classe \"nuance\" avec une confiance \"low\"."
)


def stance_system(*, reported_speech_guard: bool = False) -> str:
    """Prompt de stance, éventuellement durci contre le discours rapporté."""
    return STANCE_SYSTEM + (REPORTED_SPEECH_GUARD if reported_speech_guard else "")


def norm_confidence(value) -> str:
    c = str(value or "").strip().lower()
    return c if c in CONFIDENCE_LEVELS else "low"


def chat_retry(messages: list[dict], *, model: str, max_tokens: int) -> str:
    """`mistral_client.chat` avec BACKOFF exponentiel sur 429/5xx/réseau.

    Leçon du bench stance-large : sans retry, les 429 retombent en repli « nuance/(échec
    LLM) » → fausse abstention massive et SILENCIEUSE. 4 tentatives, 2→16 s.
    """
    delay = 2.0
    for attempt in range(4):
        try:
            return mistral_client.chat(messages, model=model, temperature=0.0,
                                       max_tokens=max_tokens, json_mode=True)
        except mistral_client.MistralError as exc:
            if exc.status in (429, 500, 502, 503, 504, 0) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise mistral_client.MistralError(0, "retries_exhausted")


def derive_cleavage_from(title: str, keywords: list[str], sample_texts: list[str],
                         *, model: str) -> dict:
    """Objet de clivage d'un thème, décrit par son titre/mots-clés/extraits représentatifs.

    Signature GÉNÉRIQUE (pas de `ThemeNode`) pour être appelable depuis le pipeline live
    comme depuis le batch. Repli gracieux sur le titre : mieux vaut une cible faible qu'une
    exception au milieu d'un build.
    """
    kw = ", ".join((keywords or [])[:10])
    contribs = "\n".join(f"- {t[:160]}" for t in (sample_texts or [])[:14])
    messages = [{"role": "system", "content": cleavage_system(title)},
                {"role": "user", "content": f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"}]
    try:
        data = json.loads(chat_retry(messages, model=model, max_tokens=200))
        objet = str(data.get("objet", "")).strip()
        return {"objet": objet or title, "justif": str(data.get("justif", "")).strip()}
    except (mistral_client.MistralError, json.JSONDecodeError):
        return {"objet": title, "justif": "(repli titre)"}


def stance_batch(cible: str, items: list[tuple[int, str]], *, model: str,
                 reported_speech_guard: bool = False) -> dict[int, dict]:
    lines = [f"[{i}] {text}" for i, text in items]
    user = (f"CIBLE : {cible}\n\n"
            f"CONTRIBUTIONS (réponds pour chaque [indice]) :\n" + "\n".join(lines))
    messages = [{"role": "system", "content": stance_system(
                    reported_speech_guard=reported_speech_guard)},
                {"role": "user", "content": user}]
    data = json.loads(chat_retry(messages, model=model, max_tokens=1500))
    out: dict[int, dict] = {}
    for rec in data.get("results", []):
        try:
            idx = int(rec["i"])
        except (KeyError, ValueError, TypeError):
            continue
        stance = str(rec.get("stance", "")).strip().lower()
        if stance not in STANCES:
            stance = "nuance"
        out[idx] = {"stance": stance,
                    "confidence": norm_confidence(rec.get("confidence")),
                    "justif": str(rec.get("justif", "")).strip()}
    return out


def run_stance(cible: str, items: list[tuple[int, str]], *, model: str,
               reported_speech_guard: bool = False) -> dict[int, dict]:
    """Stance de chaque `(indice, texte)` envers `cible`. Batché, avec repli UNITAIRE.

    Un item absent de la réponse d'un lot est rejoué SEUL : on ne perd jamais un claim en
    silence. Échec définitif → `nuance/low/(échec LLM)`, explicitement marqué comme tel.
    """
    results: dict[int, dict] = {}
    for start in range(0, len(items), BATCH):
        batch = items[start:start + BATCH]
        try:
            got = stance_batch(cible, batch, model=model,
                               reported_speech_guard=reported_speech_guard)
        except (mistral_client.MistralError, json.JSONDecodeError):
            got = {}
        for i, text in batch:
            if i not in got:
                try:
                    got.update(stance_batch(cible, [(i, text)], model=model,
                                            reported_speech_guard=reported_speech_guard))
                except (mistral_client.MistralError, json.JSONDecodeError):
                    got[i] = {"stance": "nuance", "confidence": "low",
                              "justif": "(échec LLM)"}
        results.update(got)
        time.sleep(0.02)
    return results


__all__ = [
    "BATCH", "CONFIDENCE_LEVELS", "STANCES", "STANCE_SYSTEM", "REPORTED_SPEECH_GUARD",
    "cleavage_system", "stance_system", "norm_confidence", "chat_retry",
    "derive_cleavage_from", "stance_batch", "run_stance",
]

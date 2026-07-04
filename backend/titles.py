"""Titres courts de thèmes générés par LLM (3-7 mots) — précalculés au BUILD, CACHÉS.

Chaque thème porte déjà un `label` = 3 mots-clés c-TF-IDF : parfait pour le hover, mais
ce n'est pas une phrase lisible. Ici on ajoute un VRAI **titre court**, neutre et
descriptif du sujet du cluster, rédigé par LLM à partir de ses **claims d'ancrage** (les
plus denses en vocabulaire c-TF-IDF distinctif) + ses **mots-clés** donnés comme ANCRES.

On ANCRE, on ne CONTRASTE pas : chaque titre est tiré vers le vocabulaire PROPRE de son
thème — sans jamais mentionner ni voir ses voisins. Sous l'anisotropie de l'embedding,
les claims proches du centroïde sont génériques et font tomber le titrage LLM sur des
quasi-synonymes entre thèmes pourtant distincts (« addiction » / « temps perdu »
répétés) ; en montrant plutôt les contributions RICHES en termes caractéristiques et en
donnant les mots-clés distinctifs comme ancres, deux thèmes voisins divergent
NATURELLEMENT (cf. `research/cluster_merge_note.md`, §5). Précalculé au BUILD (comme les
insights) et exposé dans `/analysis` (champ `title` par thème). `keywords`/`label`
restent intacts (hover).

**Caché par CONTENU** : la clé est un hash de (dataset, theme_id, modèle, MÉTHODE de
sélection, label, mots-clés, claims d'ancrage). Un rebuild qui réutilise `claims.json`
retrouve le même contenu → cache HIT → zéro appel LLM (idempotence de l'acceptance). Si
le contenu d'un thème change (re-clustering) OU si la méthode de sélection évolue
(`ANCHOR_METHOD`), le hash change → re-génération ciblée, jamais un titre périmé.

Repli gracieux : sans clé Mistral ou sur erreur API, on retombe sur le `label` mots-clés
— jamais un crash, jamais un titre vide. Langue-agnostique : rédigé dans la langue
dominante des contributions (le LLM la déduit des claims fournies).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from backend import analysis_store as store
from backend.develop import select_distinctive_claims
from backend.llm_cache import cached_llm
from pipeline.cluster import mistral_client

TITLES_DIRNAME = "titles"
REP_PER_THEME = 5          # claims d'ancrage montrées au LLM
MAX_KEYWORDS = 8           # mots-clés distinctifs montrés au LLM (ancres du titre)
CLAIM_MAX_CHARS = 240      # borne la longueur d'une claim d'ancrage montrée au LLM
TITLE_MAX_TOKENS = 32      # un titre est court : on borne la génération
TITLE_MAX_WORDS = 10       # garde-fou : on tronque un titre bavard
TITLE_TEMPERATURE = 0.2    # descriptif et stable, pas créatif

# Méthode de sélection des claims d'ANCRAGE — versionnée DANS la clé de cache. La faire
# évoluer (changer le critère de sélection, le tokenizer, la borne) invalide les titres
# cachés → re-génération ciblée, jamais un titre servi depuis un cache obsolète.
ANCHOR_METHOD = "ctfidf-distinctive-v1"

# Cache MÉMOIRE : key_hash -> title. Évite de relire le disque dans une même session.
_MEM_CACHE: dict[str, str] = {}


def _titles_dir(dataset: str) -> Path:
    return store.analysis_dir(dataset) / TITLES_DIRNAME


def _disk_path(dataset: str, key_hash: str) -> Path:
    return _titles_dir(dataset) / f"{key_hash}.json"


def _anchor_claims(node, member_texts: list[str] | None,
                   idf: dict[str, float] | None) -> list[str]:
    """Claims d'ANCRAGE montrées au LLM : les ≤`REP_PER_THEME` contributions les plus
    DENSES dans le vocabulaire c-TF-IDF distinctif du thème.

    Motivation (cf. `research/cluster_merge_note.md`, §5) : sous l'anisotropie de
    l'embedding, les claims proches du centroïde sont GÉNÉRIQUES et font tomber le titrage
    LLM sur des quasi-synonymes entre thèmes DISTINCTS. On ANCRE plutôt le titrage dans
    les claims riches en vocabulaire caractéristique — sélection déterministe, sans LLM
    (`develop.select_distinctive_claims`).

    Deux régimes, le second toujours disponible (aucun crash, aucun titre vide) :
      • `member_texts` fournis (+ `idf` corpus des claims) — l'appelant qui POSSÈDE l'arbre
        les passe : on sélectionne par distinctivité parmi TOUTES les claims du nœud →
        ancrage plein sur les contributions caractéristiques ;
      • absents (repli autonome) : on garde les `representative_claims` déjà bakées, dans
        leur ordre (déjà re-rankées « développement » par `backend.analysis`). On NE les
        re-sélectionne PAS par distinctivité : sur un pool de 4-8 claims l'idf local est
        quasi uniforme (chaque terme n'apparaît qu'une fois) → « distinctivité » dégénère
        en préférence pour la répétition, ce qui casserait le bon ordre. En repli, c'est
        donc le PROMPT (mots-clés donnés comme ancres) qui porte l'ancrage.

    Dédoublonnage littéral (contributions récurrentes) et bornage de longueur.
    """
    if member_texts:
        idx = select_distinctive_claims(member_texts, idf or {}, k=REP_PER_THEME)
        candidates = [member_texts[i] for i in idx]
    else:
        candidates = list(node.representative_claims or [])

    out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        t = (raw or "").strip()[:CLAIM_MAX_CHARS].strip()
        low = t.lower()
        if not t or low in seen:
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= REP_PER_THEME:
            break
    return out


def _content_key(dataset: str, node, model: str, anchors: list[str]) -> str:
    """Hash STABLE du contenu d'un thème → clé de cache idempotente.

    Inclut tout ce qui change le titre : id, modèle, MÉTHODE de sélection, label,
    mots-clés et claims d'ANCRAGE montrées. Même contenu + même méthode ⇒ même hash ⇒
    cache HIT (zéro LLM au rebuild) ; changer `ANCHOR_METHOD` ou les ancres ⇒ nouveau
    hash ⇒ re-génération ciblée (jamais un titre périmé servi).
    """
    parts = [
        dataset, node.id, model, ANCHOR_METHOD, node.label or "",
        "|".join((node.keywords or [])[:MAX_KEYWORDS]),
        "|".join(anchors[:REP_PER_THEME]),
    ]
    raw = "\x00".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _title_messages(node, anchors: list[str]) -> list[dict]:
    kw = ", ".join((node.keywords or [])[:MAX_KEYWORDS])
    reps = "\n".join(f"- {r}" for r in anchors[:REP_PER_THEME])
    system = (
        "Tu nommes des thèmes issus d'un regroupement automatique de contributions "
        "citoyennes. Tu produis un TITRE COURT, neutre et descriptif du SUJET du thème "
        "— pas une phrase complète, sans ponctuation finale, sans guillemets, sans "
        "préfixe (« Thème : »…). Tu n'inventes rien hors des éléments fournis."
    )
    user = (
        "Voici les éléments d'un thème : ses MOTS-CLÉS DISTINCTIFS (les ancres qui le "
        "singularisent) et des contributions représentatives RICHES en ces termes. "
        "Rédige UN SEUL titre de 3 à 7 mots qui ANCRE le sujet sur ces mots-clés "
        "distinctifs : préfère les termes SPÉCIFIQUES (les ancres) aux formulations "
        "génériques, pour un titre PROPRE à ce thème. Langue dominante des "
        "contributions. Réponds UNIQUEMENT par le titre, rien d'autre.\n\n"
        f"Mots-clés distinctifs (ancres) : {kw or '(aucun)'}\n\n"
        f"Contributions représentatives :\n{reps or '(aucune)'}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _clean_title(raw: str) -> str:
    """Nettoie la sortie LLM en un titre court propre (1 ligne, sans guillemets/puces)."""
    t = (raw or "").strip()
    if not t:
        return ""
    t = t.splitlines()[0].strip()                     # 1re ligne seulement
    t = re.sub(r"^[-*•\d.)\s]+", "", t)               # puces/numérotation de tête
    t = t.strip().strip("\"'«»“”").strip()            # guillemets entourants
    t = re.sub(r"^(?:th[èe]me|titre|sujet)\s*[:\-–]\s*", "", t, flags=re.IGNORECASE)
    t = t.rstrip(" .;:,–-").strip()                   # ponctuation finale
    words = t.split()
    if len(words) > TITLE_MAX_WORDS:                  # garde-fou anti-bavard
        t = " ".join(words[:TITLE_MAX_WORDS])
    return t.strip()


def _fallback(node) -> str:
    """Titre de repli (jamais vide) : le label mots-clés, ou un libellé générique."""
    return (node.label or "").strip() or f"thème {node.id}"


def title_for_node(dataset: str, node, *, model: str | None = None,
                   refresh: bool = False,
                   member_texts: list[str] | None = None,
                   idf: dict[str, float] | None = None) -> str:
    """Renvoie le titre court d'un thème, CACHÉ par contenu (mémoire → disque → LLM).

    ANCRÉ : les claims montrées au LLM sont sélectionnées par densité c-TF-IDF distinctive
    (`_anchor_claims`) et ses mots-clés distinctifs lui sont donnés comme ANCRES du titre —
    pour que deux thèmes voisins reçoivent des titres divergents SANS jamais se voir (cf.
    `research/cluster_merge_note.md`). `member_texts` (textes des claims du nœud) + `idf`
    corpus, s'ils sont passés par l'appelant qui possède l'arbre, activent l'ancrage plein
    (sélection sur toutes les claims du nœud) ; absents, on ré-ordonne les représentatives
    déjà bakées — jamais de crash.

    Idempotent : tant que le contenu du thème (méthode, label, mots-clés, ancres) ne change
    pas, le titre est servi du cache sans rappeler le LLM. Repli sur le `label` si pas de
    clé Mistral ou erreur API. Ne lève jamais.
    """
    synth_model = model or mistral_client.NAMING_MODEL
    anchors = _anchor_claims(node, member_texts, idf)
    key_hash = _content_key(dataset, node, synth_model, anchors)

    title, _ = cached_llm(
        mem_cache=_MEM_CACHE,
        key=key_hash,
        disk_path=_disk_path(dataset, key_hash),
        build_messages=lambda: _title_messages(node, anchors),
        fallback_fn=lambda *_: _fallback(node),       # repli sur le label (jamais vide)
        model=synth_model,
        max_tokens=TITLE_MAX_TOKENS,
        temperature=TITLE_TEMPERATURE,
        decode=lambda data: (data.get("title") or "").strip(),
        encode=lambda t: {"id": node.id, "title": t, "model": synth_model},
        postprocess=_clean_title,
        refresh=refresh,
        cache_fallback=False,  # repli JAMAIS caché (429 transitoire ≠ vérité — bug des 257 titres-labels)
    )
    return title

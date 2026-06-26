"""Titres courts de thèmes générés par LLM (3-7 mots) — précalculés au BUILD, CACHÉS.

Chaque thème porte déjà un `label` = 3 mots-clés c-TF-IDF : parfait pour le hover, mais
ce n'est pas une phrase lisible. Ici on ajoute un VRAI **titre court**, neutre et
descriptif du sujet du cluster, rédigé par LLM à partir de ses **claims représentatives**
+ ses **mots-clés**. Précalculé au BUILD (comme les insights) et exposé dans `/analysis`
(champ `title` par thème). `keywords`/`label` restent intacts (hover).

**Caché par CONTENU** : la clé est un hash de (dataset, theme_id, modèle, label,
mots-clés, claims représentatives). Un rebuild qui réutilise `claims.json` retrouve le
même contenu → cache HIT → zéro appel LLM (idempotence de l'acceptance). Si le contenu
d'un thème change (re-clustering), le hash change → re-génération ciblée.

Repli gracieux : sans clé Mistral ou sur erreur API, on retombe sur le `label` mots-clés
— jamais un crash, jamais un titre vide. Langue-agnostique : rédigé dans la langue
dominante des contributions (le LLM la déduit des claims fournies).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from backend import analysis_store as store
from backend.llm_cache import cached_llm
from pipeline.cluster import mistral_client

TITLES_DIRNAME = "titles"
REP_PER_THEME = 5          # claims représentatives montrées au LLM
MAX_KEYWORDS = 8           # mots-clés distinctifs montrés au LLM
TITLE_MAX_TOKENS = 32      # un titre est court : on borne la génération
TITLE_MAX_WORDS = 10       # garde-fou : on tronque un titre bavard
TITLE_TEMPERATURE = 0.2    # descriptif et stable, pas créatif

# Cache MÉMOIRE : key_hash -> title. Évite de relire le disque dans une même session.
_MEM_CACHE: dict[str, str] = {}


def _titles_dir(dataset: str) -> Path:
    return store.analysis_dir(dataset) / TITLES_DIRNAME


def _disk_path(dataset: str, key_hash: str) -> Path:
    return _titles_dir(dataset) / f"{key_hash}.json"


def _content_key(dataset: str, node, model: str) -> str:
    """Hash STABLE du contenu d'un thème → clé de cache idempotente.

    Inclut tout ce qui change le titre : id, modèle, label, mots-clés et claims
    représentatives. Même contenu ⇒ même hash ⇒ cache HIT (zéro LLM au rebuild).
    """
    parts = [
        dataset, node.id, model, node.label or "",
        "|".join((node.keywords or [])[:MAX_KEYWORDS]),
        "|".join((node.representative_claims or [])[:REP_PER_THEME]),
    ]
    raw = "\x00".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _title_messages(node) -> list[dict]:
    kw = ", ".join((node.keywords or [])[:MAX_KEYWORDS])
    reps = "\n".join(f"- {r}" for r in (node.representative_claims or [])[:REP_PER_THEME])
    system = (
        "Tu nommes des thèmes issus d'un regroupement automatique de contributions "
        "citoyennes. Tu produis un TITRE COURT, neutre et descriptif du SUJET du thème "
        "— pas une phrase complète, sans ponctuation finale, sans guillemets, sans "
        "préfixe (« Thème : »…). Tu n'inventes rien hors des éléments fournis."
    )
    user = (
        "Voici les éléments d'un thème : ses mots-clés distinctifs et quelques "
        "contributions représentatives. Donne UN SEUL titre de 3 à 7 mots qui résume "
        "le sujet du thème, dans la langue dominante des contributions. Réponds "
        "UNIQUEMENT par le titre, rien d'autre.\n\n"
        f"Mots-clés : {kw or '(aucun)'}\n\n"
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
                   refresh: bool = False) -> str:
    """Renvoie le titre court d'un thème, CACHÉ par contenu (mémoire → disque → LLM).

    Idempotent : tant que le contenu du thème (label, mots-clés, claims représentatives)
    ne change pas, le titre est servi du cache sans rappeler le LLM. Repli sur le `label`
    si pas de clé Mistral ou erreur API. Ne lève jamais.
    """
    synth_model = model or mistral_client.NAMING_MODEL
    key_hash = _content_key(dataset, node, synth_model)

    title, _ = cached_llm(
        mem_cache=_MEM_CACHE,
        key=key_hash,
        disk_path=_disk_path(dataset, key_hash),
        build_messages=lambda: _title_messages(node),
        fallback_fn=lambda *_: _fallback(node),       # repli sur le label (jamais vide)
        model=synth_model,
        max_tokens=TITLE_MAX_TOKENS,
        temperature=TITLE_TEMPERATURE,
        decode=lambda data: (data.get("title") or "").strip(),
        encode=lambda t: {"id": node.id, "title": t, "model": synth_model},
        postprocess=_clean_title,
        refresh=refresh,
    )
    return title

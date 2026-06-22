"""Accroche (`hook`) + description Markdown (`description`) de thème, par LLM — CACHÉES.

Compagnon de `backend.titles` : MÊME infrastructure LLM (`pipeline.cluster.mistral_client`)
et MÊME stratégie de cache PAR CONTENU. À côté du `title` (3-7 mots) et du `label`
(mots-clés c-TF-IDF), chaque thème reçoit :

  - **`hook`** : une courte PHRASE D'ACCROCHE, accrocheuse mais fidèle au contenu —
    de quoi donner envie de cliquer sur le thème (1 phrase, pas un titre).
  - **`description`** : une courte DESCRIPTION en Markdown qui synthétise le thème en
    RELAYANT ses mots-clés dans une phrase descriptive (1-2 phrases).

Précalculé au BUILD (cf. `backend.build_analysis`), exposé dans `/analysis` (champs
`hook` / `description` par thème). **Caché par CONTENU** : la clé hash (dataset,
theme_id, KIND, modèle, label, mots-clés, claims représentatives). Un rebuild qui
réutilise `claims.json` retrouve le même contenu → cache HIT → zéro appel LLM
(idempotence de l'acceptance). Si le contenu d'un thème change, le hash change →
re-génération ciblée.

Repli gracieux : sans clé Mistral ou sur erreur API, on retombe sur un texte dérivé
des mots-clés/label — jamais un crash, jamais un champ vide. Langue-agnostique : rédigé
dans la langue dominante des contributions (le LLM la déduit des claims fournies).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend import analysis_store as store
from pipeline.cluster import mistral_client

REP_PER_THEME = 5          # claims représentatives montrées au LLM
MAX_KEYWORDS = 8           # mots-clés distinctifs montrés au LLM
HOOK_MAX_TOKENS = 64       # une accroche = une phrase courte
DESC_MAX_TOKENS = 160      # une description = 1-2 phrases
TEMPERATURE = 0.3          # descriptif et stable, légèrement vivant pour l'accroche

# Cache MÉMOIRE par KIND : key_hash -> texte. Évite de relire le disque dans une session.
_MEM_CACHE: dict[str, str] = {}


def _dir(dataset: str, kind: str) -> Path:
    return store.analysis_dir(dataset) / kind


def _disk_path(dataset: str, kind: str, key_hash: str) -> Path:
    return _dir(dataset, kind) / f"{key_hash}.json"


def _content_key(dataset: str, node, kind: str, model: str) -> str:
    """Hash STABLE du contenu d'un thème → clé de cache idempotente (par KIND)."""
    parts = [
        dataset, node.id, kind, model, node.label or "",
        "|".join((node.keywords or [])[:MAX_KEYWORDS]),
        "|".join((node.representative_claims or [])[:REP_PER_THEME]),
    ]
    raw = "\x00".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _elements(node) -> tuple[str, str]:
    kw = ", ".join((node.keywords or [])[:MAX_KEYWORDS])
    reps = "\n".join(f"- {r}" for r in (node.representative_claims or [])[:REP_PER_THEME])
    return kw or "(aucun)", reps or "(aucune)"


def _hook_messages(node) -> list[dict]:
    kw, reps = _elements(node)
    system = (
        "Tu rédiges des accroches pour des thèmes issus d'un regroupement automatique "
        "de contributions citoyennes. Une accroche est UNE phrase courte, vivante et "
        "FIDÈLE au contenu (jamais sensationnaliste, jamais inventée), qui donne envie "
        "d'explorer le thème. Sans guillemets, sans préfixe, une seule phrase."
    )
    user = (
        "Voici les éléments d'un thème : ses mots-clés distinctifs et quelques "
        "contributions représentatives. Rédige UNE accroche d'une phrase (≤ 18 mots) "
        "qui capte l'essence du thème, dans la langue dominante des contributions. "
        "Réponds UNIQUEMENT par l'accroche.\n\n"
        f"Mots-clés : {kw}\n\n"
        f"Contributions représentatives :\n{reps}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _desc_messages(node) -> list[dict]:
    kw, reps = _elements(node)
    system = (
        "Tu rédiges de courtes descriptions de thèmes issus d'un regroupement "
        "automatique de contributions citoyennes. La description est neutre, factuelle "
        "et RELAIE les mots-clés du thème dans une phrase descriptive — elle dit de quoi "
        "parle le thème. Tu n'inventes rien hors des éléments fournis."
    )
    user = (
        "Voici les éléments d'un thème : ses mots-clés distinctifs et quelques "
        "contributions représentatives. Rédige une DESCRIPTION en Markdown COURTE "
        "(1-2 phrases) qui synthétise le thème en intégrant naturellement ses mots-clés "
        "dans la phrase. Mets en **gras** 2-3 termes-clés. Rédige dans la langue "
        "dominante des contributions. Réponds UNIQUEMENT par la description.\n\n"
        f"Mots-clés : {kw}\n\n"
        f"Contributions représentatives :\n{reps}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip().strip("\"'«»“”").strip()


def _hook_fallback(node) -> str:
    """Accroche de repli (jamais vide) : les premiers mots-clés en une amorce neutre."""
    kw = ", ".join((node.keywords or [])[:4])
    return kw or (node.label or "").strip() or f"thème {node.id}"


def _desc_fallback(node) -> str:
    """Description de repli (jamais vide) : phrase descriptive bâtie sur les mots-clés."""
    kws = (node.keywords or [])[:5]
    if kws:
        bold = ", ".join(f"**{k}**" for k in kws)
        return f"Thème articulé autour de {bold}."
    return (node.label or "").strip() or f"thème {node.id}"


_KINDS = {
    "hooks": (_hook_messages, _hook_fallback, HOOK_MAX_TOKENS),
    "descriptions": (_desc_messages, _desc_fallback, DESC_MAX_TOKENS),
}


def _field_for_node(dataset: str, node, kind: str, *, model: str | None,
                    refresh: bool) -> str:
    build_messages, fallback, max_tokens = _KINDS[kind]
    synth_model = model or mistral_client.NAMING_MODEL
    key_hash = _content_key(dataset, node, kind, synth_model)

    if not refresh:
        cached = _MEM_CACHE.get(key_hash)
        if cached:
            return cached
        disk = _disk_path(dataset, kind, key_hash)
        if disk.exists():
            try:
                data = json.loads(disk.read_text(encoding="utf-8"))
                text = (data.get("text") or "").strip()
                if text:
                    _MEM_CACHE[key_hash] = text
                    return text
            except (json.JSONDecodeError, OSError):
                pass

    text = fallback(node)
    if mistral_client.available():
        try:
            content = mistral_client.chat(
                build_messages(node), model=synth_model,
                temperature=TEMPERATURE, max_tokens=max_tokens,
            )
            cleaned = _strip(content)
            if cleaned:
                text = cleaned
        except mistral_client.MistralError:
            pass  # repli silencieux (champ jamais vide)

    _MEM_CACHE[key_hash] = text
    try:
        disk = _disk_path(dataset, kind, key_hash)
        disk.parent.mkdir(parents=True, exist_ok=True)
        disk.write_text(
            json.dumps({"id": node.id, "kind": kind, "text": text, "model": synth_model},
                       ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass
    return text


def hook_for_node(dataset: str, node, *, model: str | None = None,
                  refresh: bool = False) -> str:
    """Accroche LLM d'un thème, CACHÉE par contenu (mémoire → disque → LLM). Ne lève jamais."""
    return _field_for_node(dataset, node, "hooks", model=model, refresh=refresh)


def description_for_node(dataset: str, node, *, model: str | None = None,
                         refresh: bool = False) -> str:
    """Description Markdown LLM d'un thème, CACHÉE par contenu. Ne lève jamais."""
    return _field_for_node(dataset, node, "descriptions", model=model, refresh=refresh)

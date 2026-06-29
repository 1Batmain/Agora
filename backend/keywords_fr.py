"""BUILD — traduit en FRANÇAIS les TERMES de mots-clés non-FR de l'arbre de thèmes.

Sur une consultation multilingue (x-stance = DE/FR/IT), les mots-clés distinctifs
(c-TF-IDF, `node.keywords`) restent dans la langue d'origine (`iniziativa`, `schweiz`,
`lavoro`, `frauen`…). Le front les affiche **en français** : on les traduit AU BUILD,
juste après `build_theme_tree` (mots-clés remplis) et AVANT que les titres/accroches/
descriptions LLM et le payload persisté ne les lisent → tout l'aval voit déjà le FR.

Principe :
  - **Garde-fou mono-FR (no-op)** : si AUCUN avis n'est non-français (langue d'ingestion
    `Idea.lang`, repli `detect_lang`), on ne fait RIEN. Un corpus 100 % FR est inchangé,
    zéro appel LLM. Aucune langue codée en dur — tout est dérivé des données.
  - **Termes uniques** : on collecte les mots-clés UNIQUES de tout l'arbre, on garde ceux
    qui ne sont PAS déjà français (`is_french(detect_lang(term))`) et on les traduit en
    français par LOT (`pipeline.translate.translate_batch`, modèle cheap). Le LLM recopie
    de toute façon un terme déjà français (repli sûr).
  - **Cache idempotent** : `backend/cache/<dataset>/keywords_fr.json`, clé = terme source,
    validé par le modèle. Un rebuild ne re-traduit QUE les termes nouveaux. Le fichier vit
    à la racine du dataset (hors `analysis/`) → survit à un `store.clear()`.
  - **Remappage** : `node.keywords` est réécrit vers les versions FR en **préservant
    l'ordre** (pertinence c-TF-IDF) et en **dédupliquant** (deux termes → même FR). Le
    `node.label` dérivé des mots-clés est re-dérivé sur le FR (top-`LABEL_K`).

NE touche PAS l'ancrage des claims : seuls les mots-clés d'AFFICHAGE changent (les spans
verbatim restent ancrés sur `text_clean` original).
"""

from __future__ import annotations

import os
from typing import Callable

from backend.recluster import dataset_dir
from pipeline.ingest.lang import detect_lang
from pipeline.translate import DEFAULT_TRANSLATE_MODEL, is_french, translate_batch

KEYWORDS_FR_NAME = "keywords_fr.json"
# Modèle de traduction (CHEAP), surchargeable par env — aucune valeur de corpus en dur.
TRANSLATE_MODEL = os.environ.get("AGORA_TRANSLATE_MODEL", DEFAULT_TRANSLATE_MODEL)
# Nombre de mots-clés tête formant le label (cf. `name_clusters(label_k=3)`).
LABEL_K = 3

ProgressFn = Callable[[int, int], None]


def keywords_fr_path(dataset: str):
    return dataset_dir(dataset) / KEYWORDS_FR_NAME


def _resolve_lang(avis_lang: str | None, text: str) -> str:
    """Langue effective d'un avis : `Idea.lang` si exploitable, sinon détection."""
    lang = (avis_lang or "").strip().lower()
    if lang and lang != "und":
        return lang
    return detect_lang(text, default="und")


def _dataset_has_non_french(avis, lang_of: dict[str, str]) -> bool:
    """Vrai si AU MOINS un avis n'est pas français (→ traduction des mots-clés utile)."""
    for a in avis:
        if not is_french(_resolve_lang(lang_of.get(str(a.id)), a.text or "")):
            return True
    return False


def _unique_terms(tree) -> list[str]:
    """Tous les TERMES de mots-clés uniques de l'arbre, dans l'ordre de 1ʳᵉ apparition."""
    seen: set[str] = set()
    terms: list[str] = []
    for node in tree.nodes.values():
        for kw in node.keywords:
            if kw and kw not in seen:
                seen.add(kw)
                terms.append(kw)
    return terms


def _remap_keywords(terms: list[str], mapping: dict[str, str]) -> list[str]:
    """Remplace chaque terme par sa version FR, en préservant l'ordre + dédup."""
    seen: set[str] = set()
    out: list[str] = []
    for kw in terms:
        fr = mapping.get(kw, kw) or kw
        if fr not in seen:
            seen.add(fr)
            out.append(fr)
    return out


def translate_tree_keywords(
    dataset: str,
    tree,
    lang_of: dict[str, str] | None = None,
    *,
    model: str | None = None,
    refresh: bool = False,
    on_progress: ProgressFn | None = None,
) -> dict[str, str]:
    """Remappe `node.keywords` (et `node.label` dérivé) vers le FRANÇAIS, AU BUILD.

    Renvoie la map `{terme_source: terme_fr}` effectivement appliquée (vide si no-op).
    Mute l'arbre EN PLACE (effet de bord sur `node.keywords`/`node.label`). Idempotent :
    le cache `keywords_fr.json` n'est ré-appelé que pour les termes nouveaux. `refresh`
    ignore le cache disque (re-traduit tout le non-FR). Ne lève jamais sur erreur LLM
    (repli gracieux : terme source conservé).
    """
    from backend.analysis_store import _read_json, write_json  # I/O atomique partagée

    lang_of = lang_of or {}
    avis = getattr(getattr(tree, "prepared", None), "avis", []) or []

    # 1) Garde-fou mono-FR : aucun avis non-français → rien à traduire (no-op).
    if not _dataset_has_non_french(avis, lang_of):
        return {}

    use_model = model or TRANSLATE_MODEL
    path = keywords_fr_path(dataset)
    cached = (_read_json(path) or {}) if not refresh else {}
    if not isinstance(cached, dict):
        cached = {}

    # 2) Termes uniques de l'arbre ; on garde ceux qui ne sont PAS déjà français.
    terms = _unique_terms(tree)
    non_fr = [t for t in terms if not is_french(detect_lang(t))]

    # 3) Tri cache HIT (modèle inchangé) / à traduire.
    mapping: dict[str, str] = {}
    to_translate: list[str] = []
    for t in non_fr:
        prev = cached.get(t)
        if isinstance(prev, dict) and prev.get("model") == use_model and prev.get("fr"):
            mapping[t] = prev["fr"]
        else:
            to_translate.append(t)

    # 4) Traduction batchée des manquants (un seul appel LLM par lot).
    new_cache: dict[str, dict] = dict(cached)
    total = len(to_translate)
    if total:
        results = translate_batch(to_translate, model=use_model)
        for term, fr in zip(to_translate, results):
            if fr and fr.strip():
                fr = fr.strip()
                mapping[term] = fr
                new_cache[term] = {"fr": fr, "model": use_model}
            # échec LLM (None) → pas de cache, terme source conservé en aval
        if on_progress:
            on_progress(total, total)
        write_json(path, new_cache)

    if not mapping:
        return {}

    # 5) Remappage en place : keywords (ordre + dédup) puis label dérivé.
    for node in tree.nodes.values():
        if not node.keywords:
            continue
        node.keywords = _remap_keywords(node.keywords, mapping)
        if node.keywords:
            node.label = " · ".join(node.keywords[:LABEL_K])
    return mapping

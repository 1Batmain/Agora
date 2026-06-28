"""Enrichissement SERVE-TIME des indices d'un dataset — couverture + fidélité verbatim.

Ajoute DEUX métriques de CONFIANCE honnêtes au payload `/analysis` CACHÉ, SANS toucher
le moindre cache de build (lecture seule de l'arbre persisté + de `claims.json`) :

  - **couverture**        : part des contributions citoyennes rattachées à un VRAI thème
                            (vs le résidu non classé `__noise__`).
  - **fidélité_verbatim** : part des claims dont TOUS les spans + la cible sont des
                            sous-chaînes EXACTES du `text_clean` source (zéro trahison).

Les deux dérivent d'artefacts DÉJÀ persistés (l'arbre d'`analysis.json` + `claims.json`),
donc elles ne déclenchent JAMAIS de rebuild. La revérification verbatim (qui parcourt
tous les claims) est MÉMOÏSÉE par le mtime de `claims.json` → un /analysis répété reste
instantané tant que le cache d'extraction n'a pas bougé.
"""

from __future__ import annotations

import json

from backend.claims_endpoint import CLAIMS_NAME, DEFAULT_MIN_CHARS, _avis_from_ideas
from backend.recluster import CACHE_DIR
from pipeline.claims.span import as_claim

# Clé MACHINE du cluster de bruit (HDBSCAN/non classé) — alignée sur `frontend NOISE_KEY`.
NOISE_KEY = "__noise__"

# Mémoïsation de la fidélité verbatim par (dataset → (mtime de claims.json, index)).
# Recomputer sur CHAQUE /analysis marcherait, mais relit + revérifie tout le cache
# d'extraction — inutile tant que `claims.json` n'a pas changé.
_FIDELITY_CACHE: dict[str, tuple[float, dict]] = {}


def _is_noise(theme: dict) -> bool:
    """Vrai si ce thème EST le cluster non classé (par id/label/title == `__noise__`)."""
    return NOISE_KEY in (theme.get("id"), theme.get("label"), theme.get("title"))


def coverage_index(payload: dict, total_ideas: int) -> dict | None:
    """Indice `couverture` dérivé de l'arbre CACHÉ : part d'avis rattachés à un thème.

    Numérateur = avis des thèmes RACINE réels (hors `__noise__`). Dénominateur = TOUTES
    les contributions ingérées (`total_ideas`) : un avis sans claim/cluster compte comme
    non classé. Repli sur themed+noise des thèmes si `total_ideas` indisponible.
    """
    themes = payload.get("themes") or []
    roots = [t for t in themes if isinstance(t, dict) and t.get("parent_id") is None]
    if not roots:
        return None
    themed = sum(max(0, int(t.get("n_avis", 0) or 0)) for t in roots if not _is_noise(t))
    noise_themes = sum(max(0, int(t.get("n_avis", 0) or 0)) for t in roots if _is_noise(t))
    denom = total_ideas if total_ideas > 0 else (themed + noise_themes)
    if denom <= 0:
        return None
    noise = max(0, denom - themed)
    value = max(0.0, min(1.0, themed / denom))
    return {
        "key": "couverture",
        "value": round(value, 4),
        "detail": {"classes": themed, "noise": noise, "total": denom},
    }


def _compute_fidelity(dataset: str, ideas) -> dict | None:
    """Recompute la fidélité verbatim sur `claims.json` CACHÉ (lecture seule, zéro LLM)."""
    path = CACHE_DIR / dataset / CLAIMS_NAME
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    claims = rec.get("claims")
    if not isinstance(claims, dict):
        return None
    # `a.text` = text_clean (texte canonique masqué) — base d'ancrage des spans/cible.
    text_by_id = {a.id: a.text for a in _avis_from_ideas(ideas, DEFAULT_MIN_CHARS)}
    n_claims = n_vb = 0
    for aid, lst in claims.items():
        text = text_by_id.get(str(aid))
        if text is None:
            continue
        for cd in (lst or []):
            n_claims += 1
            if as_claim(cd).is_verbatim(text):
                n_vb += 1
    if n_claims == 0:
        return None
    return {
        "key": "fidelite_verbatim",
        "value": round(n_vb / n_claims, 4),
        "detail": {"n_claims": n_claims, "n_verbatim": n_vb},
    }


def fidelity_index(dataset: str, ideas) -> dict | None:
    """Indice `fidelite_verbatim` (mémoïsé par mtime de `claims.json`)."""
    path = CACHE_DIR / dataset / CLAIMS_NAME
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _FIDELITY_CACHE.get(dataset)
    if cached is None or cached[0] != mtime:
        idx = _compute_fidelity(dataset, ideas)
        if idx is None:
            return None
        _FIDELITY_CACHE[dataset] = (mtime, idx)
        cached = _FIDELITY_CACHE[dataset]
    return cached[1]


def dataset_keywords(payload: dict, top_n: int = 14) -> list[str]:
    """Mots-clés REPRÉSENTATIFS du dataset : agrège les c-TF-IDF des MACROS, pondérés par
    la taille du thème (n_avis) et le rang du mot, dédupliqués. Dérivé du payload caché —
    zéro LLM, zéro rebuild. Donne les termes saillants à afficher avec la synthèse globale.
    """
    macros = [t for t in payload.get("themes", [])
              if isinstance(t, dict) and not t.get("parent_id")]
    weight: dict[str, float] = {}
    for m in macros:
        n = float(m.get("n_avis") or 0) or 1.0
        for rank, kw in enumerate((m.get("keywords") or [])[:5]):
            if isinstance(kw, str) and kw:
                weight[kw] = weight.get(kw, 0.0) + n / (rank + 1)
    return [k for k, _ in sorted(weight.items(), key=lambda x: -x[1])[:top_n]]


def enrich_indices(payload: dict, dataset: str, ideas) -> dict:
    """Ajoute couverture + fidelite_verbatim à `dataset_stats.indices` (EN MÉMOIRE).

    Mute le `payload` FRAÎCHEMENT chargé (`read_analysis` re-parse à chaque requête, donc
    cette écriture ne RETOURNE JAMAIS au cache disque). Idempotent : remplace toute entrée
    de même clé. No-op si la forme attendue est absente (robustesse).
    """
    stats = payload.get("dataset_stats")
    if not isinstance(stats, dict):
        return payload
    indices = stats.get("indices")
    if not isinstance(indices, list):
        return payload

    total_ideas = len(ideas) if ideas is not None else 0
    extra = [ix for ix in (coverage_index(payload, total_ideas),
                           fidelity_index(dataset, ideas)) if ix is not None]
    if extra:
        new_keys = {e["key"] for e in extra}
        keep = [ix for ix in indices
                if not (isinstance(ix, dict) and ix.get("key") in new_keys)]
        stats["indices"] = keep + extra
    kws = dataset_keywords(payload)
    if kws:
        stats["keywords"] = kws
    return payload

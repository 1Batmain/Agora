"""Moteur d'ABSTRACTION — couche macro par étiquette canonique + affectation embedding.

Au-dessus de la couche PLATE (γ, pic de modularité), on regroupe les thèmes REDONDANTS en
macros SANS souder les sujets distincts. Validé (`research/abstraction_note.md`) :

  1. le LLM normalise chaque thème en une ÉTIQUETTE canonique (3-6 mots) — surface → sens,
     ce qui rapproche les redondants (les ~5 « addiction » deviennent la même catégorie) ;
  2. le LLM PROPOSE un petit jeu de CATÉGORIES abstraites (sa force : nommer l'abstrait) ;
  3. l'affectation thème → catégorie se fait par EMBEDDING (produit scalaire dans l'espace
     recentré) — ce qui GARANTIT une partition stricte, là où le regroupement LLM libre
     double-assigne les thèmes ambigus.

Fonctions PURES (chat_fn / embed_fn injectés → testable, découplé de Mistral). Le résultat est
DÉTERMINISÉ par le cache disque (`compute` une fois au build, `load` par les autres étapes) —
indispensable à la cohérence de l'arbre entre build_analysis / build_opinion / build_arguments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.cluster.layers import centre, flat_partition
from pipeline.embed.registry import resolve_model_id

MIN_THEMES = 4          # en dessous, la couche plate suffit (pas d'abstraction)
PROFILE_MAX_TOKENS = 450  # cap conservateur, sûr pour toutes les fenêtres d'embedder supportées
                          # (arctic-l 8192, nomic-v2 512) — jamais tronqué à l'embed


def _profile(claims: list[str], *, chat_fn, model: str) -> str:
    """PROFIL du thème destiné à être EMBEDDÉ : fidèle et précis, mais canonique dans son
    ouverture pour que deux thèmes du même sujet se rejoignent dans l'espace. C'est le levier
    central (moteur B) : ce qu'on embedde décide si les redondants fusionnent SANS perdre la
    précision. Une étiquette courte sur-collapse (perd la précision) ; une phrase riche varie
    trop (ne merge pas) ; le profil est l'équilibre — commence par nommer le SUJET DE FOND."""
    ex = "\n".join(f"- {c[:200]}" for c in claims[:20])
    msg = [
        {"role": "system", "content":
         "Rédige un PROFIL de ce thème en 3 à 5 phrases (≤ 300 mots). Commence par UNE phrase qui "
         "nomme le SUJET DE FOND en termes généraux et canoniques, puis précise fidèlement les "
         "angles et positions portés par les témoignages. Ce profil servira à regrouper ce thème "
         "avec ceux qui traitent du même sujet — reste fidèle ET canonique. Juste le profil."},
        {"role": "user", "content": f"Témoignages :\n{ex}\n\nProfil :"},
    ]
    return chat_fn(msg, model=model, temperature=0.0, max_tokens=PROFILE_MAX_TOKENS).strip()


def compute(cluster_texts: list[list[str]], *, chat_fn, embed_fn, model: str) -> dict | None:
    """Couche macro par RÉ-EMBEDDING (moteur B). `cluster_texts[i]` = claims représentatifs du
    thème i.

    Pipeline : profil fidèle par thème (LLM) → ré-embedding local du profil → clustering des
    profils (même moteur `flat_partition`, γ au pic de modularité) = couche macro. Le profil
    normalise la surface vers le sens SANS perdre la précision (cf. `research/profile_embed_note`).

    Renvoie `{"profiles":[...], "assign":[macro par thème]}` ou `None` si trop peu de thèmes ou si
    tout retombe dans un seul macro (pas d'abstraction utile). Les macros sont nommés en aval
    (c-TF-IDF + titre LLM), comme tout nœud à enfants.
    """
    n = len(cluster_texts)
    if n < MIN_THEMES:
        return None
    profiles = [_profile(c, chat_fn=chat_fn, model=model) for c in cluster_texts]
    vecs = centre(np.asarray(embed_fn(profiles), dtype=np.float64))
    part, _meta = flat_partition(vecs, seed=42)
    assign = part.tolist()

    used = sorted(set(assign))
    if len(used) < 2:
        return None                                  # tout dans un macro = pas d'abstraction
    remap = {m: i for i, m in enumerate(used)}
    return {"profiles": profiles, "assign": [remap[a] for a in assign]}


# --- Cache disque : DÉTERMINISE l'abstraction entre les étapes du build ------------------- #
def signature(clusters: list[list[int]], *, embedder: str = "", chat_model: str = "") -> str:
    """Empreinte STABLE de l'abstraction : partition (ordre-insensible) + EMBEDDER + modèle
    de chat. L'embedder EN FAIT PARTIE car les profils sont ré-embeddés : un cache construit
    avec un modèle (ex. jina) ne doit JAMAIS être re-servi pour un build sur un autre modèle
    (ex. nomic-v2, permissif) — question de licence ET de cohérence d'espace."""
    # Embedder NORMALISÉ (alias → id canonique) : « arctic-l » et l'id résolu
    # « Snowflake/… » désignent le MÊME modèle → même signature (sinon cache miss → repli plat
    # → désync des theme_id entre analysis/opinion/arguments).
    emb = resolve_model_id(embedder) if embedder else ""
    key = repr((sorted(tuple(sorted(c)) for c in clusters), emb, chat_model))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def load(path: Path, clusters: list[list[int]], *, embedder: str = "",
         chat_model: str = "") -> dict | None:
    """Relit l'abstraction cachée si elle correspond à CETTE partition ET au MÊME embedder/
    modèle (sinon None → recalcul)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("signature") != signature(clusters, embedder=embedder, chat_model=chat_model):
        return None                                  # partition/embedder changé → cache périmé
    return data.get("result")


def save(path: Path, clusters: list[list[int]], result: dict, *, embedder: str = "",
         chat_model: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = signature(clusters, embedder=embedder, chat_model=chat_model)
    path.write_text(json.dumps({"signature": sig, "result": result},
                               ensure_ascii=False, indent=2), encoding="utf-8")

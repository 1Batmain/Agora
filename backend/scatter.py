"""Nuage de points UMAP 2D — une VRAIE contribution par point.

Contrairement à `densityScatter.ts` (front) qui échantillonne des points
SYNTHÉTIQUES depuis la grille de densité, ce module renvoie les coordonnées
UMAP 2D RÉELLES de chaque contribution, alignées avec son cluster et son texte.

Sources (tout précalculé, zéro calcul à la requête) :
  - `cache/<dataset>/umap2d.npy`   : projection 2D (N, 2) alignée à ideas.jsonl
  - `cache/<dataset>/ideas.jsonl`  : texte + id de chaque contribution
  - `cache/<dataset>/analysis/avis.json` : mapping avis → claims → cluster_id/color

Le payload est léger : coords (x, z) + cluster_id + color + id + extrait texte.
Pour les gros datasets (>5000), on échantillonne déterministement pour rester
fluide côté canvas (le texte complet reste accessible via /avis/{id}).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from backend.recluster import dataset_dir, cache_paths


# Budget de points pour les gros datasets (échantillonnage déterministe).
# 1621 (TikTok) < seuil → tous les points. 28k (Grand Débat) → échantillonné.
MAX_POINTS = 5000
# Longueur de l'extrait texte servi par point (payload léger ; texte complet
# via /avis/{id}).
TEXT_EXCERPT = 140


class ScatterUnavailable(RuntimeError):
    """Nuage indisponible (UMAP absent et pas de cache umap2d.npy)."""


def _load_avis_clusters(dataset: str) -> dict[str, dict]:
    """Lit `analysis/avis.json` → {avis_id: {cluster_id, leaf_id, color}}.

    On ne garde que le PREMIER claim de chaque avis pour la couleur/cluster
    (un avis peut avoir plusieurs claims dans des clusters différents, mais
    pour la visu scatter on veut une seule couleur par point).
    """
    avis_path = dataset_dir(dataset) / "analysis" / "avis.json"
    if not avis_path.exists():
        return {}
    try:
        with open(avis_path, encoding="utf-8") as fh:
            avis_data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}

    out: dict[str, dict] = {}
    for avis_id, entry in avis_data.items():
        claims = entry.get("claims", []) if isinstance(entry, dict) else []
        if not claims:
            continue
        first = claims[0]
        out[avis_id] = {
            "cluster_id": first.get("leaf_id") or first.get("cluster_id"),
            "color": first.get("color", ""),
        }
    return out


def scatter_payload(dataset: str, limit: int | None = None) -> dict:
    """Payload `GET /scatter` : liste de points UMAP 2D réels.

    Chaque point = une contribution avec ses coords (x, z), son cluster,
    sa couleur, son id, et un extrait texte. Tout est lu depuis le cache
    précalculé — zéro UMAP, zéro clustering à la requête.

    `limit` échantillonne déterministement si > MAX_POINTS (défaut).
    Lève `ScatterUnavailable` si `umap2d.npy` est absent.
    """
    from backend.density import compute_umap2d, DensityUnavailable

    try:
        coords = compute_umap2d(dataset)
    except DensityUnavailable as exc:
        raise ScatterUnavailable(str(exc)) from exc

    # ideas.jsonl aligné à umap2d.npy par index.
    _, ideas_path, _ = cache_paths(dataset)
    ideas: list[dict] = []
    with open(ideas_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ideas.append(json.loads(line))

    n = min(len(ideas), coords.shape[0])
    if n == 0:
        return {"points": [], "total": 0, "returned": 0}

    clusters = _load_avis_clusters(dataset)

    # Construit tous les points.
    all_points: list[dict] = []
    for i in range(n):
        idea = ideas[i]
        idea_id = idea.get("id", str(i))
        cl = clusters.get(idea_id, {})
        text = idea.get("text_clean") or idea.get("text") or ""
        all_points.append({
            "x": round(float(coords[i, 0]), 4),
            "z": round(float(coords[i, 1]), 4),
            "id": idea_id,
            "cluster_id": cl.get("cluster_id"),
            "color": cl.get("color", ""),
            "text": text[:TEXT_EXCERPT],
        })

    # Échantillonnage déterministe si trop de points.
    cap = limit or MAX_POINTS
    if len(all_points) > cap:
        rng = random.Random(42)
        all_points = rng.sample(all_points, cap)

    return {
        "points": all_points,
        "total": n,
        "returned": len(all_points),
    }

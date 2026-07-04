"""Pont collect → pipeline Agora complet, SANS toucher à `pipeline/ingest/`.

`prepare` transforme les données brutes déjà collectées (`data/collect/raw/<slug>/`,
cf. `pipeline/collect/`) en une source canonique consommable par la machinerie
générique existante :

  1. lecture des fichiers bruts via les loaders de `pipeline.collect` (imports
     seulement — aucun module existant n'est modifié) ;
  2. fonte des colonnes texte libre (heuristique `pipeline.collect.classify`) ;
  3. écriture d'un JSONL canonique + d'un DESCRIPTEUR généré sous
     `data/ingest_full/` (artefacts dérivés, gitignorés via `data/`).

Ensuite le pipeline STANDARD prend le relais, inchangé :

    uv run python -m pipeline.ingest_full.prepare --slug <slug> [--question …]
    uv run python -m backend.build_cache --dataset <slug> \
        --descriptor data/ingest_full/<slug>.descriptor.json
    uv run python -m backend.build_analysis --dataset <slug>   # (clé Mistral)
    uv run python -m backend.build_opinion --dataset <slug>    # (clé Mistral)

Générique : le slug est un paramètre, rien n'est codé en dur.
"""

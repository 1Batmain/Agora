"""Collecte open data — consultations citoyennes de l'Assemblée nationale → DuckDB.

Module AUTONOME (ne touche pas à `pipeline/ingest/`) : il scrape le portail
https://data.assemblee-nationale.fr/autres/consultations-citoyennes pour découvrir
les consultations publiées (JAMAIS de liste en dur — contrat de généricité),
télécharge leurs fichiers open data dans `data/collect/raw/<slug>/`, puis charge
le tout dans une base DuckDB canonique (`data/collect/consultations.duckdb`).

Schéma : catalogue (`consultations`, `files`), stats par colonne (`questions`),
réponses fondues (`responses`, colonnes texte libre uniquement par défaut) et la
vue `contributions` (interface stable : une ligne = une réponse ouverte).

Usage :
    uv run --extra collect python -m pipeline.collect run [--only SLUG] [--limit N]
    uv run --extra collect python -m pipeline.collect catalog
    uv run --extra collect python -m pipeline.collect status
"""

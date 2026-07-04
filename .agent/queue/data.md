# Lane data — ingestion · prétraitement · anonymisation

Owns: `data/` (gitignored), `pipeline/ingest/`, `pipeline/collect/`. Sortie = Idea JSONL canonique.

## T-D5 · Collecte multi-consultations AN → DuckDB
- Goal : scraper le portail open data (data.assemblee-nationale.fr/autres/
  consultations-citoyennes), télécharger toutes les consultations publiées et les
  exposer dans une base DuckDB canonique (catalogue + vue `contributions`).
- Accept : `python -m pipeline.collect run` catalogue ~30 consultations ; les
  pathologiques (dump SQL 476 Mo, fichiers vides serveur) sont cataloguées
  skipped/empty sans bloquer ; re-run idempotent ; zéro slug en dur.
- Deps : aucune (module autonome ; pattern DuckDB repris de remontrances).

## T-D1 · Scripts de download reproductibles
- Goal : récupérer Consultation TikTok (open data AN) + x-stance (ZurichNLP) dans
  `data/` via script idempotent (pas de données versionnées).
- Accept : `python -m pipeline.ingest.download` régénère `data/raw/*` from scratch.
- Deps : aucune. Contract : produit du `Idea.text/source/ts` brut.

## T-D2 · Nettoyage + normalisation
- Goal : fautes, casse, espaces, ponctuation, suppression vide/quasi-vide.
- Accept : `text_clean` non vide pour ≥ X% ; rapport de réduction.
- Deps : T-D1.

## T-D3 · Déduplication
- Goal : near-dup par similarité embedding (cosine > 0.95) → garder 1, incrémenter
  `weight`. (Dépend de l'embedding service nlp → peut être un 2e passage.)
- Accept : taux de dup mesuré ; idées fusionnées conservent un compteur.
- Deps : T-D2 + nlp T-N1.

## T-D4 · Anonymisation + détection de langue
- Goal : `author_hash` (pas de PII), `lang` (fasttext/langdetect).
- Accept : zéro PII en clair dans le JSONL canonique ; lang renseigné.
- Deps : T-D2. Contract : remplit `author_hash`, `lang`.

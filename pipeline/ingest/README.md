# Lane DATA — `pipeline/ingest/`

Ingestion, nettoyage et **anonymisation** des avis citoyens → JSONL canonique
consommé par la lane **nlp**. Possède `data/` (gitignored) et `pipeline/ingest/`.

## Sortie : `data/processed/ideas.jsonl`

Un objet **Idea** par ligne, aligné sur le `GraphNode` du contrat
(`queue/cross-lane.md`) :

```json
{
  "id": "tiktok:20",
  "type": "idea",
  "label": "<texte court d'affichage (≤ 80 car.)>",
  "props": {
    "text": "<texte brut nettoyé des espaces>",
    "text_clean": "<normalisé : casse/espaces/ponctuation, PII masquées>",
    "ts": "2025-05-12 18:03:00",
    "lang": "fr",
    "author_hash": "<sha256 salé tronqué, AUCUNE PII en clair>",
    "source": "tiktok|xstance|synthetic",
    "weight": 1.0
  }
}
```

## Régénérer from scratch

```bash
# 1. (optionnel) télécharger les sources brutes dans data/raw/ (idempotent)
uv run python -m pipeline.ingest.download

# 2. construire ideas.jsonl (télécharge tout seul si data/raw/ est vide)
uv run --with langdetect python -m pipeline.ingest.build
```

`build` imprime les compteurs : total, lus en entrée, % vides retirés, par
source, par langue. Options utiles :

| Option | Effet |
|--------|-------|
| `--max-per-source N` | plafonne le nb d'avis par source (box partagée sobre) |
| `--synthetic`        | force l'échantillon synthétique (ignore `data/raw/`) |
| `--out PATH`         | chemin de sortie alternatif |

> `langdetect` est recommandé (`--with langdetect`). Sans lui, le pipeline ne
> bloque pas : il retombe sur une heuristique FR/EN/DE (qualité moindre).

## Sources

| Source | Origine | Contenu retenu |
|--------|---------|----------------|
| **x-stance** | [`ZurichNLP/xstance`](https://github.com/ZurichNLP/xstance) (zip public) | commentaires politiques labellisés FAVOR/AGAINST — **FR uniquement** par défaut (~17 k). `AGORA_XSTANCE_ALL_LANGS=1` pour garder de/it. |
| **Consultation TikTok** | [open data Assemblée nationale](https://data.assemblee-nationale.fr/autres/consultations-citoyennes/tiktok) (CSV, 33 609 répondants) | réponses à la **question ouverte** « décrire ce sentiment de mal-être / faits de harcèlement » (col. 141), seul champ de texte libre riche (~1,8 k témoignages). |
| **synthetic** *(fallback)* | généré localement (`synthetic.py`) | ~300 avis FR « mobilité urbaine », avec duplicats + paraphrases pour exercer la dédup (T-D3) et le clustering. Utilisé seulement si le réseau échoue. |

Le CSV TikTok est encodé **cp1252**, séparateur `;` ; le schéma de colonnes est
figé dans `config.py`.

## Anonymisation & langue (RGPD-friendly)

- `author_hash` = `sha256(sel | source | auteur)` tronqué (16 hex). Le sel
  (`AGORA_HASH_SALT`) rend les hash stables mais non ré-identifiables. Un
  garde-fou (`assert_no_pii`) refuse toute valeur non opaque.
- `text_clean` masque les **emails, téléphones, URLs et @mentions**.
- `lang` détecté par `langdetect` (déterministe, seed fixé).
- `data/` reste **gitignored** : aucune donnée brute/perso n'est versionnée.

## Fixture committé

`fixtures/ideas.sample.jsonl` (~50 lignes, 100 % synthétique/anonyme) — démarre
nlp + eval sans données réelles. Régénérer :

```bash
uv run --with langdetect python -m pipeline.ingest.make_fixture
```

## Modules

| Fichier | Tâche | Rôle |
|---------|-------|------|
| `download.py`   | T-D1 | téléchargement idempotent → `data/raw/` |
| `normalize.py`  | T-D2 | nettoyage / normalisation / `text_clean` / filtrage vides |
| `anonymize.py`  | T-D4 | `author_hash` salé |
| `lang.py`       | T-D4 | détection de langue (langdetect + repli) |
| `build.py`      | —    | orchestration → `ideas.jsonl` + compteurs |
| `synthetic.py`  | —    | corpus FR de repli / fixture |
| `make_fixture.py` | —  | génère le fixture committé |

> **T-D3 (déduplication)** dépend des embeddings (lane nlp) → 2ᵉ passage ultérieur.

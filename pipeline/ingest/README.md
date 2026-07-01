# Lane DATA — `pipeline/ingest/`

Ingestion, nettoyage et **anonymisation** des avis citoyens → JSONL canonique
consommé par la lane **nlp**. Possède `data/` (gitignored) et `pipeline/ingest/`.

> **Généricité (principe directeur).** L'outil tourne sur **des centaines de
> consultations**, sujets et **langues** variés. Ici, **un corpus = un
> descripteur** (`descriptors/*.json`), pas du code. Aucune particularité
> corpus-spécifique ne vit dans la logique : encoding, délimiteur, colonnes,
> URL… sont des **valeurs** de descripteur. Multilingue par défaut.

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
    "source": "tiktok|xstance|demo_sante|synthetic|…",
    "weight": 1.0
  }
}
```

## Régénérer from scratch

```bash
# 1. (optionnel) télécharger les sources brutes dans data/raw/ (idempotent,
#    suit l'`url` de chaque descripteur)
uv run python -m pipeline.ingest.download

# 2. construire ideas.jsonl (télécharge tout seul si data/raw/ est vide)
uv run --with langdetect python -m pipeline.ingest.build
```

`build` découvre **tous** les descripteurs de `descriptors/`, ingère ceux dont
le fichier source est présent, et imprime les compteurs (total, lus en entrée,
% vides retirés, par source, par langue). Options utiles :

| Option | Effet |
|--------|-------|
| `--descriptor PATH` | n'ingérer que ce(s) descripteur(s) explicite(s) (répétable) |
| `--max-per-source N` | plafonne le nb d'avis par source (box partagée sobre) |
| `--synthetic`        | force l'échantillon synthétique (ignore les descripteurs) |
| `--out PATH`         | chemin de sortie alternatif |

> `langdetect` est recommandé (`--with langdetect`). Sans lui, le pipeline ne
> bloque pas : repli heuristique élargi (fr/en/de/es/pt/it/nl), défaut `und`.

## Ajouter une consultation = écrire un descripteur (zéro code)

Déposez un fichier JSON dans `descriptors/`. Champs (`id` et `text`
obligatoires, le reste optionnel) :

```jsonc
{
  "name": "ma_consultation",   // préfixe des ids + author_hash
  "format": "csv",             // "csv" | "jsonl"
  "path": "data/raw/x.csv",    // relatif à la racine repo (ou absolu)
  "url": "https://…",          // optionnel : download idempotent
  "encoding": "utf-8",         // défaut "utf-8"
  "delimiter": ",",            // csv, défaut ","
  "has_header": true,          // csv, défaut true
  "archive": "zip",            // optionnel : `path` = zip…
  "members": ["train.jsonl"],  // …membres jsonl à lire
  "columns": {                 // CHAMP CANONIQUE -> référence de colonne
    "id":   0,                 //   int  => index 0-based (csv sans noms)
    "text": "comment",         //   str  => clé (jsonl, ou csv via en-tête)
    "ts":   1,                 //   ts/author/lang/weight OPTIONNELS
    "author": "author",
    "lang": "language",
    "weight": "poids"
  },
  "lang_keep": ["fr"]          // KNOB optionnel : sous-ensemble langue.
}                              //   ABSENT (défaut) = garder TOUTES les langues.
```

- `author` non mappé → défaut = `id` (1 réponse = 1 répondant).
- `lang` fourni par la source → conservé ; sinon (re)détecté (défaut `und`).
- `weight` non mappé → `1.0`.

**Démo (acceptation).** Un descripteur seul ingère une source factice d'un
autre sujet, aux colonnes nommées et au délimiteur `|`, multilingue :

```bash
uv run python -m pipeline.ingest.build \
  --descriptor pipeline/ingest/fixtures/demo_source.descriptor.json \
  --out /tmp/demo_ideas.jsonl
# -> 8 avis, langues {fr,en,es,de,it,pt}, poids 2.0/3.0 lus — AUCUN code touché.
```

## Sources (= descripteurs)

| Descripteur | Origine | Contenu retenu |
|-------------|---------|----------------|
| `descriptors/tiktok.json` | [open data Assemblée nationale](https://data.assemblee-nationale.fr/autres/consultations-citoyennes/tiktok) (CSV `cp1252`, `;`, 33 609 répondants) | réponses à la **question ouverte** « décrire ce sentiment de mal-être » (col. 141), seul champ de texte libre riche (~1 772 témoignages). |
| `descriptors/xstance.json` | [`ZurichNLP/xstance`](https://github.com/ZurichNLP/xstance) (zip public, jsonl) | commentaires politiques **multilingues de/fr/it**. **Toutes les langues gardées** par défaut. Pour une démo FR-only, ajouter `"lang_keep": ["fr"]`. |
| `descriptors/` *(le vôtre)* | n'importe quel CSV/JSONL | cf. « Ajouter une consultation ». |
| **synthetic** *(fallback)* | généré localement (`synthetic.py`) | ~300 avis FR « mobilité urbaine » (duplicats + paraphrases). Utilisé seulement si aucune source réelle. |

## Anonymisation & langue (RGPD-friendly)

- `author_hash` = `sha256(sel | source | auteur)` tronqué (16 hex). Le sel
  (`AGORA_HASH_SALT`) rend les hash stables mais non ré-identifiables. Un
  garde-fou (`assert_no_pii`) refuse toute valeur non opaque.
- `text_clean` masque les **emails, téléphones, URLs et @mentions**.
- `lang` = celui fourni par la source si présent, sinon `langdetect` (déterministe).
  **Défaut `und`** (jamais `fr`) : multilingue de 1er ordre, on ne jette ni ne
  mal-étiquette aucune langue par défaut.
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
| `sources.py`    | —    | **descripteur déclaratif + `read_generic` unique** (cœur généricité) |
| `descriptors/`  | data | un JSON par consultation (tiktok, xstance, le vôtre) |
| `download.py`   | T-D1 | téléchargement idempotent (suit les `url` des descripteurs) → `data/raw/` |
| `normalize.py`  | T-D2 | nettoyage / normalisation / `text_clean` / filtrage vides |
| `anonymize.py`  | T-D4 | `author_hash` salé |
| `lang.py`       | T-D4 | détection de langue (langdetect + repli élargi, défaut `und`) |
| `build.py`      | —    | orchestration → `ideas.jsonl` + compteurs |
| `synthetic.py`  | —    | corpus FR de repli / fixture |
| `make_fixture.py` | —  | génère le fixture committé |

> **T-D3 (déduplication)** dépend des embeddings (lane nlp) → 2ᵉ passage ultérieur.

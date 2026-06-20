# NOTE — Ingestion générique (lane ingest-gen)

Résout les constats d'`eval/genericity_audit.md` **#1, #10, #11, #13**.
Principe : **un corpus = un descripteur (config), pas du code.**

## Ce qui change

| Avant (corpus-spécifique) | Après (générique) |
|---|---|
| `read_xstance()` + `read_tiktok()` codés en dur dans `build.py` | un seul **`read_generic(descriptor)`** (`sources.py`) |
| `TIKTOK_TEXT_COL=141`, `TIKTOK_ENCODING="cp1252"`, `delimiter=";"`, URLs… constantes de module (`config.py`) | **valeurs DANS le descripteur** `descriptors/tiktok.json` |
| 2 sources en dur ; nouvelle consultation = **patcher le code** | déposer un `descriptors/*.json` = **zéro code** |
| `download.py` : `download_xstance/_tiktok` codés en dur | suit l'`url` de **chaque descripteur** |
| `XSTANCE_FR_ONLY` (défaut : jette le non-FR) | défaut = **garde toutes les langues** ; `lang_keep` = knob explicite |
| `lang.detect_lang(default="fr")`, repli FR/EN/DE | défaut **`"und"`**, repli élargi fr/en/de/es/pt/it/nl |
| `lang` absent → supposé FR | `lang` source respecté, sinon (re)détecté (défaut `und`) |

- `#1` 🔴 → descripteur déclaratif + `read_generic` unique.
- `#10` 🟠 → `lang.py` : défaut `und`, repli multilingue élargi.
- `#11` 🟠 → multilingue par défaut ; sous-ensemble langue = knob `lang_keep`.
- `#13` 🟠 → `to_idea` émet **toujours** un `lang` ; source-fournie sinon détectée,
  jamais `fr` par défaut. (Le défaut consommateur `lang="fr"` de
  `pipeline/cluster/io.py` appartient à une autre lane ; côté ingest la sortie
  porte désormais toujours un `lang` correct, donc le cas « champ absent » ne se
  produit plus pour nos JSONL.)

## Acceptation — vérifiée

**1. Non-régression TikTok (~1772) + x-stance multilingue :**
```
$ uv run --with langdetect python -m pipeline.ingest.build
  par source : {'tiktok': 1772, 'xstance': 67271}
  par langue : {'fr': 18834, 'de': 48621, 'it': 1454, 'en': 58, 'es': 7, ...}
```
→ TikTok = **1772** ✓. x-stance garde **DE/FR/IT** (avant : FR-only) ✓.

**2. Nouveau CSV factice via un SEUL descripteur, sans toucher au code :**
source `fixtures/demo_source.csv` (autre sujet « santé/ruralité », colonnes
**nommées**, délimiteur `|`, colonnes `poids`+`langue`), descripteur
`fixtures/demo_source.descriptor.json` :
```
$ uv run python -m pipeline.ingest.build \
    --descriptor pipeline/ingest/fixtures/demo_source.descriptor.json --out /tmp/demo.jsonl
  total avis : 8   (2 vides filtrés)
  par langue : {'fr': 3, 'en': 1, 'es': 1, 'de': 1, 'it': 1, 'pt': 1}
```
→ 6 langues conservées, `weight` 2.0/3.0 lus, ids `demo_sante:r0xx` — **aucun
code modifié** ✓.

**3. Knob `lang_keep` (sous-ensemble = opt-in, jamais un défaut) :**
```
défaut (absent)      langues : [de, en, es, fr, it, pt]   n=10
lang_keep=["fr"]     langues : [fr]                        n=5
```
→ subset uniquement quand demandé explicitement ✓.

## Fichiers
- `sources.py` — `SourceDescriptor` + `read_generic` + `load_descriptors`.
- `descriptors/{tiktok,xstance}.json` — sources réelles en **config**.
- `fixtures/demo_source.{csv,descriptor.json}` — démo d'acceptation.
- `config.py` — purgé des constantes corpus-spécifiques (+`DESCRIPTORS_DIR`).
- `build.py` / `download.py` — pilotés par descripteurs. `lang.py` — multilingue.

## Hors lane (signalé, non modifié)
- `.gitignore` (racine) : ajout d'**une exception** `!pipeline/ingest/fixtures/*.csv`
  pour committer la fixture de démo (sinon `*.csv` global l'exclut).
- `pipeline/cluster/io.py:32,60` (`lang="fr"` par défaut côté **lecture**) : lane
  cluster. Côté ingestion le `lang` est désormais toujours écrit (cf. #13).

# ACQUIS_NOTE — jeux réels acquis & convertis (M-ABSA + WikiSection)

> Worker `work/dataacq`. Acquisition + conversion **reproductibles** des 2 jeux réels
> validés par la recon (`../DATASETS_RECON.md`). Brut volumineux sous `data/datasets/`
> (**gitignoré**) ; **code + échantillons + cette note** sous `eval/segmentation/datasets/`.
> Cible : entraîner la **tête légère** du segmenteur appris (PLAN §4-5), **jamais** sur
> notre synthétique. Stats vérifiées au pull du **2026-06-21**.

## Arborescence
```
eval/segmentation/datasets/
  fetch_mabsa.py            # télécharge HF + convertit → data/datasets/mabsa/<lang>.jsonl
  fetch_wikisection.py      # convertit le clone → data/datasets/wikisection/<lang>.jsonl
  make_samples.py           # (re)génère les échantillons committés (déterministe)
  loaders.py                # load_mabsa(lang) / load_wikisection(lang) + stats CLI
  mabsa.sample.jsonl        # 40 phrases (fr/de/en/es, dont multi-aspect)  [COMMITTÉ]
  wikisection.sample.jsonl  # 20 docs (en/de), frontières + labels         [COMMITTÉ]
data/datasets/              # BRUT, GITIGNORÉ (jamais committé)
  mabsa/<lang>.jsonl        # 21 langues
  wikisection/<lang>.jsonl  # en, de
  WikiSection/              # clone git + .json décompressés (~634 Mo)
```

---

## 1. M-ABSA — signal A (multi-aspect par phrase courte)

- **Source** : HF `Multilingual-NLP/M-ABSA` — ABSA multilingue parallèle, 7 domaines
  (coursera, food, hotel, laptop, phone, restaurant, sight) × 21 langues × 3 splits.
- **Licence** : **Apache-2.0** ✅ (commercial / service déployé OK, sans friction).
- **Re-téléchargement** (le brut est gitignoré) :
  ```bash
  uv run --with datasets --with huggingface_hub \
      python -m eval.segmentation.datasets.fetch_mabsa            # les 21 langues
  uv run --with datasets --with huggingface_hub \
      python -m eval.segmentation.datasets.fetch_mabsa --langs fr de en es
  ```
- **Conversion** : ligne brute `texte####[[terme, catégorie, polarité], ...]` →
  on extrait les **catégories d'aspect DISTINCTES** de la phrase. `multi_aspect = (≥2
  catégories distinctes)` → la phrase porte une **frontière interne** (multi-thème),
  cible directe du détecteur multi-aspect (PLAN §4). Polarité & terme ignorés (non requis).
- **Format de sortie** (`data/datasets/mabsa/<lang>.jsonl`, 1 phrase/ligne) :
  ```json
  {"id":"mabsa-fr-restaurant-train-12","lang":"fr","domain":"restaurant","split":"train",
   "text":"...","aspect_categories":["food quality","service general"],
   "n_aspects":2,"multi_aspect":true}
  ```
- **Stats (309 062 phrases au total, 24.6 % multi-aspect)** — nb par langue / % multi-aspect :

  | lang | n | % multi | lang | n | % multi | lang | n | % multi |
  |---|---|---|---|---|---|---|---|---|
  | ar | 14758 | 24.7 | hr | 14749 | 24.7 | sk | 14741 | 24.7 |
  | da | 14736 | 24.7 | id | 14760 | 24.7 | sv | 14744 | 24.7 |
  | **de** | 14754 | 24.7 | ja | 14759 | 24.7 | sw | 14724 | 24.6 |
  | **en** | 14776 | 24.7 | ko | 14765 | 24.6 | th | 14752 | 24.7 |
  | **es** | 14747 | 24.7 | nl | 14647 | 24.5 | tr | 14662 | 24.5 |
  | **fr** | 14194 | 22.9 | pt | 14747 | 24.6 | vi | 14758 | 24.7 |
  | hi | 14772 | 24.7 | ru | 14756 | 24.7 | zh | 14761 | 24.6 |

- **Caveats** :
  - ⚠️ **Corpus parallèle** (mêmes phrases traduites entre langues) → le transfert
    cross-langue mesuré dessus est **optimiste** (lexique aligné). Vraie preuve de
    généricité = frontières **natives** par langue (cf. WikiSection / WP maison, PLAN §5.4).
  - ⚠️ **Pas d'italien** (`it` absent des 21 langues M-ABSA). FR/DE/EN/ES disponibles ;
    pour `it`, ni M-ABSA ni WikiSection ne couvrent — à combler ailleurs (WP maison FR/IT/DE).
  - ⚠️ **Catégories non normalisées entre domaines** : restaurant/hotel = `entity attribute`
    minuscule (`food quality`), laptop/phone = `ENTITY#ATTRIBUTE` ou `Entity#Attribute`
    (`LAPTOP#QUALITY`, `Screen#General`) → ~279 « catégories » distinctes par langue, mais
    c'est l'union de **7 schémas hétérogènes**. Pour de l'apprentissage **par domaine**,
    filtrer sur `domain`. Le signal `multi_aspect` (≥2) reste robuste indépendamment du schéma.
  - Domaine = produits/services (≠ civique) ; granularité **phrase** (≈ notre avis court) ✅.

---

## 2. WikiSection — signal B (frontières de section + label thème)

- **Source** : `github.com/sebastianarnold/WikiSection` — sections Wikipédia étiquetées,
  domaines `disease` & `city`, langues **EN + DE**.
- **Licence** : **CC-BY-SA-3.0** ✅ — ⚠️ **share-alike** : OK pour une tête entraînée **en
  interne** (on ne redistribue pas le corpus) ; un dérivé **publié** devrait rester CC-BY-SA.
- **Re-téléchargement** (brut gitignoré) :
  ```bash
  git clone https://github.com/sebastianarnold/WikiSection data/datasets/WikiSection
  (cd data/datasets/WikiSection && tar xzf wikisection_dataset_json.tar.gz)
  uv run python -m eval.segmentation.datasets.fetch_wikisection           # en, de
  ```
- **Conversion** (doc encyclopédique long → docs ~taille-avis à frontières) :
  1. **fusion** des annotations consécutives de même `sectionLabel` (frontière = vrai
     changement de thème, pas une sous-section) ;
  2. **troncature** de chaque segment à `--max-seg-chars` (350) sur une **frontière de
     phrase** (`.!?…`/saut de ligne, langue-agnostique) ;
  3. **fenêtrage** en docs de `--segs-per-doc` (3) segments, **≥2 → ≥1 frontière interne**.
- **Format de sortie** (`data/datasets/wikisection/<lang>.jsonl`) = **format gold** du
  harness (`seg_bench.GoldItem` / `gold_large.json`), frontières AVANT l'espace de jointure :
  ```json
  {"id":"wikisection-en-disease-Pneumonic_plague-0","lang":"en","domain":"disease",
   "split":"train","type":"multi","text":"...","boundaries_char":[297,619],
   "segment_labels":["disease.symptom","disease.cause","disease.treatment"],
   "segment_headings":["Signs and symptoms","Cause","Treatment"]}
  ```
  Validé : **108 179 frontières**, 100 % tombent sur un espace, offsets strictement
  croissants, `len(segment_labels) == len(boundaries_char)+1` (cf. `loaders.py`).
- **Stats** (texte moyen ~720 car) :

  | lang | docs | frontières | front./doc |
  |---|---|---|---|
  | en | 40 030 | 72 636 | 1.81 |
  | de | 20 080 | 35 543 | 1.77 |

- **Caveats** :
  - ⚠️ **Share-alike** (cf. licence ci-dessus).
  - Domaine **étroit** (maladies & villes) et **encyclopédique** (≠ opinion) ; transitions
    de section **franches** → risque d'apprendre des frontières trop « dures » vs nos
    transitions douces (mode d'échec PLAN §3). À mélanger avec M-ABSA (court, opinion).
  - Le **fenêtrage** assemble des segments **consécutifs du même article** (pas de couture
    artificielle inter-articles) → frontières réelles, mais docs = fragments d'articles.
    Reproductible/déterministe (pas de tirage aléatoire). Régler via `--max-seg-chars` /
    `--segs-per-doc` si on veut des docs plus longs/courts.

---

## 3. Loader & critère d'acceptation (vérifié)

```python
from eval.segmentation.datasets.loaders import load_mabsa, load_wikisection
mabsa_fr = load_mabsa("fr")              # list[MabsaItem]  (.aspect_categories, .multi_aspect)
mabsa_fr_train = load_mabsa("fr", split="train")
wiki_en = load_wikisection("en")         # list[GoldItem]   (.boundaries_char, .seg_themes)
```
- `load_mabsa` → `MabsaItem(id, lang, text, aspect_categories, multi_aspect, domain, split)`.
- `load_wikisection` → **`GoldItem`** (réutilisable tel quel par `seg_bench` / l'entraînement).
- Stats imprimables : `uv run --with numpy python -m eval.segmentation.datasets.loaders`.

**Vérifié soi-même** :
- `loaders.py` charge M-ABSA (fr/de/en/es ; `it` signalé absent) et WikiSection (en/de) sans
  erreur, imprime les stats. ✅
- Échantillons committés parsent ; `mabsa.sample.jsonl` a bien des phrases ≥2 aspects (24/40
  multi-aspect, jusqu'à 5) ; `wikisection.sample.jsonl` a des `boundaries_char` cohérents
  avec le `text` (chaque frontière = espace). ✅
- Brut (mabsa 92 Mo, wikisection 60 Mo, clone 634 Mo) sous `data/datasets/` **gitignoré** —
  rien de volumineux committé. ✅

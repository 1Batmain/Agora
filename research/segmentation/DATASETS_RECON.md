# RECON — jeux de données RÉELS pour le segmenteur thématique / extracteur multi-aspect

> Recherche read-only (worker `work/datarecon`). Objectif : trouver du **réel externe,
> humain, multilingue** pour **entraîner** la tête légère du segmenteur appris (cf.
> `PLAN.md` §4-5) — **jamais** sur notre synthétique. Tous les faits ci-dessous sont
> **vérifiés sur source** (URLs citées). Ce qui n'a pas pu être vérifié est marqué `⚠ non vérifié`.
>
> Notre cas cible (rappel) : **avis citoyen court d'opinion → N thèmes**, multilingue
> (FR/IT/DE/EN), frontière = changement de sujet. Deux signaux d'apprentissage utiles :
> **(A) multi-aspect / multi-label** (1 texte court → plusieurs thèmes — le plus proche),
> **(B) frontières de segmentation** (où le sujet change dans un texte plus long).

---

## 1. Tableau comparatif (vérifié)

| Jeu | Ce que c'est | Langues | Taille | Licence (utilisable ?) | Accès | Signal (A multi-aspect / B frontière) | Granularité | Proximité avis-citoyen | Caveats |
|---|---|---|---|---|---|---|---|---|---|
| **M-ABSA** | ABSA multilingue, triplets (terme, catégorie, polarité) | **21** (ar da de en es fr hi hr id ja ko nl pt ru sk sv sw th tr vi zh) | ~**310 k** phrases (train 185k / val 45.9k / test 79.7k) | **Apache-2.0** ✅ (commercial OK) | HF `Multilingual-NLP/M-ABSA` | **A** : multi-aspect par phrase (≥2 triplets fréquents) | phrase | **Très haute** (avis courts d'opinion, multi-thème, multilingue) | Corpus **parallèle** (mêmes phrases traduites entre langues → cross-lingue = traduction, pas natif) ; domaines produits/services, pas civique |
| **SemEval ABSA** 2014/15/16 | ABSA, ancêtre de M-ABSA | 2016 : **8** (en ar zh nl fr ru es tr) | ~quelques k phrases/langue/domaine | **CC-BY-NC-ND** ⚠ (acad. + non-commercial, **pas de dérivés**) via ELRA/META-SHARE | [alt.qcri.org/semeval2016/task5](https://alt.qcri.org/semeval2016/task5/) ; subsets HF non officiels | **A** : multi-aspect par phrase | phrase | Haute (restaurants/laptops/hôtels) | Licence **bloquante** si Agora = service public/payant ; EN-centré en pratique ; XML |
| **WikiSection** | Sections Wikipédia = thèmes (+labels) | **EN, DE** | ~**38 k** docs / **242 k** sections | **CC-BY-SA-3.0** ✅ (partage à l'identique) | [github.com/sebastianarnold/WikiSection](https://github.com/sebastianarnold/WikiSection) | **B** : frontières = limites de section (+ label thème) | section / phrase | Moyenne (maladies & villes — encyclopédique, pas opinion) | Domaine étroit (2 classes Wikidata) ; share-alike contamine les dérivés |
| **Wiki-727K / Wiki-50** | Segmentation par table des matières Wikipédia | **EN** | **727 746** docs (Wiki-50 = 50 docs test) | **CC-BY-4.0** ✅ | [Zenodo 4737322](https://zenodo.org/records/4737322) ; [github.com/koomri/text-segmentation](https://github.com/koomri/text-segmentation) | **B** : frontières de section (benchmark de réf.) | section / phrase | Faible (encyclopédique, monolingue) | **EN seulement** ; 1.6 GB ; sujets longs ≠ avis courts |
| **Wikipédia maison** (FR/IT/DE) | Sections extraites par nous = frontières | **n'importe** (FR/IT/DE visés) | illimité (dumps) | **CC-BY-SA-3.0** ✅ (texte WP) | dumps + `mwparserfromhell` / WikiExtractor | **B** : `==Section==` → frontière | section / phrase | Faible-moyen (encyclopédique) | À construire (effort) ; même biais domaine que WikiSection ; share-alike |
| **Amazon MARC** | Reviews multilingues + note étoiles | **6** (en ja de fr zh es) | 200k/5k/5k par langue | **Non-commercial recherche** ⚠ + **retiré** (defunct) | HF `defunct-datasets/amazon_reviews_multi` | (ni A ni B : 1 note globale) | document | Moyenne (opinion réelle) mais **mono-label** | **Plus hébergé par Amazon** ; licence restrictive ; pas de découpe aspect |
| **Allociné** | Critiques ciné FR | **FR** | **200 k** (160k/20k/20k) | **MIT** ✅ | HF `tblard/allocine` | (mono-label binaire) | document | Moyenne (opinion FR réelle) | Sentiment binaire **document-level**, pas multi-aspect ni frontière |
| **QMSum** (AMI/ICSI/parlement) | Réunions + frontières de sujet humaines | **EN** | 232 réunions | **MIT** ✅ | [github.com/Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum) | **B** : `topic_list` + `relevant_text_span` (bornes humaines) | tour de parole / span | Faible (oral, EN) | Transcripts oraux ; EN seul ; peu de docs |
| **GUM** | Corpus multi-genre, segmentation discours (RST/EDU) | **EN** | ~26–32 k EDUs / 213 docs / 12 genres | **CC-BY-4.0** (annotations) ✅ | [gucorpling.org/gum](https://gucorpling.org/gum/) ; [github.com/amir-zeldes/gum](https://github.com/amir-zeldes/gum) | **B** : EDU = frontière de discours (clause) | clause (EDU) | Faible-moyen (genres variés dont forum Reddit) | EN ; EDU ≠ thème (trop fin) ; petit |
| **Brand24/mms** | Méga-corpus sentiment massivement multilingue | **27** | 79 jeux, ~**5.5 M** ex. | **CC-BY-NC-4.0** ⚠ (non-commercial) | HF `Brand24/mms` | (3-classes mono-label) | phrase/doc | Moyenne (opinion, 27 langues) | **Mono-label** ; non-commercial ; agrégat hétérogène |
| **DimABSA** | ABSA dimensionnel (valence-arousal) | **6** | 76 958 aspects / 42 590 phrases | ⚠ non vérifié | [arXiv 2601.23022](https://arxiv.org/pdf/2601.23022) | **A** : aspects multiples + opinion terms | phrase | Haute (ABSA multi-aspect) | Récent ; licence/accès **non vérifiés** ; à surveiller |

---

## 2. Fiches détaillées (les pièces maîtresses)

### ⭐ M-ABSA — `Multilingual-NLP/M-ABSA` (le plus proche de notre cas)
- **Quoi** : "most extensive multilingual parallel dataset for ABSA", 7 domaines (coursera, hotel,
  laptop, restaurant, phone, sight, food), labels = **triplets** `[terme d'aspect, catégorie d'aspect,
  polarité]`. **Plusieurs triplets par phrase** sont fréquents → c'est exactement notre « 1 avis → N thèmes ».
- **Format** : `phrase ####  [[term, category, sentiment], ...]`. La **catégorie d'aspect** est le label
  thème exploitable directement comme cible multi-label (zéro lexique, langue-agnostique).
- **Licence** : **Apache-2.0** → utilisable sans friction (y compris service déployé). Source :
  [HF dataset card](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA), [arXiv 2502.11824](https://arxiv.org/pdf/2502.11824).
- **Accès** : `from datasets import load_dataset; ds = load_dataset("Multilingual-NLP/M-ABSA")`.
- **Comment en tirer du signal** :
  - **Tête multi-label (A)** : `nb de catégories distinctes dans la phrase ≥ 2` → la phrase **porte une
    frontière interne** (multi-thème). Cible directe pour le « détecteur multi-aspect » du PLAN. Réel, court, opinion, multilingue.
  - **Transfert cross-langue** (preuve de généricité, PLAN §5.4) : train sur split `fr`, eval sur `de`/`it`/`vi`… Natif côté langues, mais ⚠ **parallèle** (phrases traduites) → le transfert mesuré est optimiste (vocabulaire aligné).

### WikiSection — `github.com/sebastianarnold/WikiSection` (frontières + labels, EN/DE)
- **Quoi** : 242 k sections étiquetées (maladies/villes), EN+DE. JSON : `text` + `annotations[]` avec
  `begin`, `length`, `sectionHeading`, `sectionLabel` (normalisé, ex. `disease.treatment`).
- **Signal (B)** : transition entre deux sections adjacentes = **frontière positive** ; deux phrases
  d'une même section = **négatif**. Le `sectionLabel` donne en bonus un thème.
- **Licence** : **CC-BY-SA-3.0** (texte Wikipédia). ✅ utilisable, ⚠ **share-alike** (un dérivé publié
  doit rester CC-BY-SA). Pour une **tête entraînée en interne**, sans redistribuer le corpus, c'est OK.
- **Accès** : `wikisection_dataset_json.tar.gz` dans le repo. Source : [README](https://github.com/sebastianarnold/WikiSection), [arXiv 1902.04793](https://arxiv.org/abs/1902.04793).

### Wiki-727K / Wiki-50 — `Zenodo 4737322` (frontières à l'échelle, EN)
- **Quoi** : 727 746 docs WP segmentés par table des matières ; Wiki-50 = jeu de test de référence.
- **Signal (B)** : frontières de section ; benchmark de segmentation **standard** (comparabilité littérature).
- **Licence** : **CC-BY-4.0** ✅ (la plus propre du lot segmentation). 1.6 GB (`wiki_727K.tar.bz2`).
- **Accès** : [Zenodo](https://zenodo.org/records/4737322) / [github.com/koomri/text-segmentation](https://github.com/koomri/text-segmentation). EN seulement → utile pour *volume* et *comparabilité*, pas pour le multilingue.

---

## 3. Recommandation classée — SOCLE train / test

### TRAIN (socle réel, diversifié + multilingue) — 2 obligatoires + 1 d'extension

1. **M-ABSA** *(primaire, signal A — multi-aspect)* — **Apache-2.0**, 21 langues, avis **courts d'opinion
   multi-thème** : c'est le jumeau de notre tâche. Entraîne le **détecteur multi-aspect** (1 avis → N thèmes)
   et fournit le **banc de transfert cross-langue** natif. Aucune friction de licence.
2. **WikiSection (EN+DE)** *(secondaire, signal B — frontières)* — **CC-BY-SA-3.0**, apporte la
   supervision « **où le sujet change** » dans du texte plus long + un label de section. Complète M-ABSA
   (qui est court) et donne déjà **2 langues** pour la frontière. Tête interne → share-alike non bloquant.
3. **Wikipédia maison FR/IT/DE** *(extension, signal B multilingue)* — à construire (`mwparserfromhell` :
   `==Section==` → frontière). **CC-BY-SA**. Sert la **preuve de généricité** PLAN §5.4 (train FR → eval DE/IT)
   sur de la **frontière** et non plus seulement de l'aspect. Fallback **Wiki-727K** (EN, CC-BY-4.0) si on
   veut du volume/benchmark avant d'investir l'extraction multilingue.

**Pourquoi ce trio** : couvre les **deux signaux** (A multi-aspect court + B frontière long), **deux familles
de domaine** (opinion + encyclopédique → anti-surapprentissage de style), **licences exploitables**, et un
**axe multilingue réel** (M-ABSA natif 21 langues, WikiSection EN/DE, WP maison FR/IT/DE).

### Pourquoi **M-ABSA plutôt que SemEval ABSA**
Même famille (M-ABSA descend des schémas SemEval), mais M-ABSA est **Apache-2.0** (vs **CC-BY-NC-ND**
bloquant), **21 langues** (vs 8, EN-centré en pratique), format **unifié/nettoyé**, multi-aspect préservé.
SemEval reste l'ancêtre de référence à **citer**, mais sa licence non-commerciale + no-derivatives le rend
inadapté à un service déployé.

### TEST
- **In-domain (vérité terrain produit)** : notre **gold de témoignages** (`gold.json` / `gold_large.json`, 305).
  C'est le juge final (eval-as-truth, PLAN §3).
- **Held-out multi-aspect** : split **test M-ABSA** → mesure A + **transfert cross-langue** (train `fr` → test `de`/`it`/…).
- **Held-out frontière** : **WikiSection test** et/ou **Wiki-50** (benchmark comparable littérature).
- **Règle** : avis/docs **jamais vus** en train (CV stricte, PLAN §5.3) ; aucun chevauchement de source entre train et test.

### Accès concret (copier-coller)
```python
from datasets import load_dataset
mabsa = load_dataset("Multilingual-NLP/M-ABSA")          # Apache-2.0, 21 langues, multi-aspect
# WikiSection : git clone https://github.com/sebastianarnold/WikiSection
#   puis tar xzf wikisection_dataset_json.tar.gz   (EN/DE, frontières + labels)
# Wiki-727K  : https://zenodo.org/records/4737322  (wiki_727K.tar.bz2, CC-BY-4.0)  + Wiki-50 test
# Wikipédia maison : dumps dumps.wikimedia.org + pip install mwparserfromhell
```

---

## 4. Honnêteté — licences, domain gaps, non-vérifiable

**Licences à surveiller**
- **SemEval ABSA** : **CC-BY-NC-ND** (non-commercial, **pas de dérivés**) via ELLA/META-SHARE. Si Agora est
  un service public/financé, l'usage est juridiquement risqué → **écarté du socle** (gardé pour citation/compa).
- **Amazon MARC** : licence **non-commerciale recherche** **et** corpus **retiré par Amazon** (`defunct-datasets/…`).
  À éviter pour de la prod.
- **Brand24/mms** : **CC-BY-NC-4.0** → non-commercial. Utile pour explorer, **pas** pour entraîner un modèle déployé.
- **WikiSection / Wikipédia maison** : **CC-BY-SA** → **share-alike**. OK pour une tête entraînée **en interne**
  (on ne redistribue pas le corpus) ; **attention** si un jour on publie le modèle/corpus dérivé.
- **M-ABSA (Apache-2.0), Wiki-727K (CC-BY-4.0), Allociné (MIT), QMSum/AMI/ICSI (MIT/CC-BY-4.0), GUM (CC-BY-4.0)** :
  permissifs, sans friction.

**Domain gaps (le réel ≠ notre réel)**
- **Aucun** corpus public n'est du **témoignage citoyen civique**. M-ABSA = avis produits/services ;
  WikiSection/Wiki-727K = encyclopédique ; Allociné = ciné ; QMSum = réunions orales. On entraîne sur du
  **proxy d'opinion multi-thème**, on **valide sur notre gold** → c'est la bonne discipline, mais le
  **transfert domaine** doit être mesuré, pas supposé.
- **Granularité** : M-ABSA = phrase (≈ notre avis court) ✅ ; WikiSection/Wiki-727K = section (sujets **longs**,
  transitions **franches**) → risque d'apprendre des frontières trop « dures » vs nos transitions douces
  (« et… / du coup… », mode d'échec connu PLAN §3). GUM/EDU = **trop fin** (clause ≠ thème).
- **Multilingue « natif »** : M-ABSA est **parallèle** (phrases traduites) → le transfert cross-langue qu'il
  mesure est **optimiste** (lexique aligné). La **vraie** preuve de généricité passe par des frontières
  **natives** par langue → d'où l'intérêt de **Wikipédia maison FR/IT/DE** (textes natifs, non traduits).

**Piège méthodo à éviter**
- Tentation de **concaténer** des phrases M-ABSA pour fabriquer des « documents à frontières connues » :
  texte réel mais **assemblage artificiel** → frontière trop facile (rupture lexicale nette). À n'utiliser
  qu'en **augmentation contrôlée**, jamais comme socle ; le PLAN proscrit le synthétique en train.

**Non vérifié / à confirmer avant usage**
- **DimABSA** : tailles citées depuis l'abstract ([arXiv 2601.23022](https://arxiv.org/pdf/2601.23022)),
  **licence et accès HF non vérifiés** — prometteur (ABSA multi-aspect, 6 langues) mais à valider.
- **SemEval** : tailles exactes par langue/domaine non recomptées ici (variables selon sous-tâche/année).
- **QMSum/AMI/ICSI** : licence QMSum = MIT (repo) ; statut précis des audios/transcripts sous-jacents
  (AMI/ICSI ≈ CC-BY-4.0) non re-vérifié pièce par pièce.
- Les **chiffres HF** (splits M-ABSA, etc.) proviennent des dataset cards à la date de consultation
  (**2026-06-21**) ; revérifier au pull (les cards évoluent).

---

### Sources
- M-ABSA : [HF](https://huggingface.co/datasets/Multilingual-NLP/M-ABSA) · [arXiv 2502.11824](https://arxiv.org/pdf/2502.11824)
- SemEval-2016 ABSA : [tâche QCRI](https://alt.qcri.org/semeval2016/task5/) · [ACL S16-1002](https://aclanthology.org/S16-1002/) · licence FR via [META-SHARE/ELRA](http://metashare.elda.org/repository/browse/semeval-2016-absa-restaurant-reviews-french-train-data-subtask-1/b99b226269cb11e59117842b2b6a04d7838bf239baf5485e8d810283dc97eaf3/)
- WikiSection : [GitHub](https://github.com/sebastianarnold/WikiSection) · [arXiv 1902.04793](https://arxiv.org/abs/1902.04793)
- Wiki-727K / Wiki-50 : [Zenodo 4737322](https://zenodo.org/records/4737322) · [GitHub koomri](https://github.com/koomri/text-segmentation) · [arXiv 1803.09337](https://arxiv.org/pdf/1803.09337)
- Amazon MARC : [HF defunct](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi) · [arXiv 2010.02573](https://arxiv.org/abs/2010.02573)
- Allociné : [HF tblard/allocine](https://huggingface.co/datasets/tblard/allocine)
- QMSum : [GitHub Yale-LILY](https://github.com/Yale-LILY/QMSum) · [arXiv 2104.05938](https://arxiv.org/pdf/2104.05938)
- GUM : [gucorpling.org/gum](https://gucorpling.org/gum/) · [GitHub amir-zeldes/gum](https://github.com/amir-zeldes/gum)
- Brand24/mms : [HF](https://huggingface.co/datasets/Brand24/mms)
- DimABSA : [arXiv 2601.23022](https://arxiv.org/pdf/2601.23022) *(non vérifié)*

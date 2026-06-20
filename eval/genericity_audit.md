# Audit de généricité — Agora (read-only)

> **Contrat** (`queue/cross-lane.md` → PRINCIPE DIRECTEUR) : l'outil tournera sur
> **des centaines de consultations originales**, sujets et **langues** variés. Tout
> doit être **générique / dérivé des données** ; le corpus TikTok FR est un *cas de
> test*, pas une cible. Tout littéral corpus-spécifique dans le code = bug.

**Méthode** : grep large (`eval/genericity_scan.py`, reproductible) + lecture ciblée
de tout le code de prod (`pipeline/`, `backend/`, `frontend/src/`) et d'éval. Le scan
*signale* ; ce rapport *juge le contexte* (défaut raisonnable exposé en knob = OK ;
constante enfouie non dérivée = à corriger).

**Légende sévérité** : 🔴 casse sur un autre corpus · 🟠 dégrade silencieusement ·
🟢 cosmétique / acceptable.

---

## Synthèse — top 5 actions pour rendre le pipeline corpus-agnostique

1. **Adaptateur d'ingestion générique** (🔴 #1). Aujourd'hui on ne peut ingérer QUE
   `xstance` et `tiktok` : deux `read_*()` codés en dur + indices de colonnes figés
   (`TIKTOK_TEXT_COL = 141`). Une nouvelle consultation **exige d'écrire du code**.
   → Descripteur de source déclaratif (CSV/JSON : chemin, encoding, délimiteur,
   mapping colonnes id/text/ts) ; `read_generic(descriptor)` unique.
2. **Naming langue-agnostique + stopwords dérivés** (🔴 #2, #3). Le titrage des
   thèmes repose sur une **liste FR figée** et un **regex Latin-only** : sur un
   corpus non-FR les labels se remplissent de mots-outils étrangers, sur un corpus
   non-latin (cyrillique, grec, arabe, CJK) le tokenizer ne capte **rien** → labels
   vides. → Stopwords dérivés du corpus (document-frequency, `max_df`) ;
   tokenisation Unicode (`\p{L}` / `re.UNICODE`).
3. **Défaut d'embedding = le gagnant multilingue, pas e5-small** (🔴 #5). Le
   pipeline batch (`build.py`) embedde par défaut avec `e5-small`, dont le VERDICT
   du contrat dit qu'il **clusterise PAR LANGUE** (NMI langue=0.81) → inutilisable
   en multilingue. → `DEFAULT_MODEL_ID = nomic-v2` (le winner validé), ou modèle
   explicite obligatoire.
4. **Dériver les seuils de la distribution, pas du corpus TikTok** (🟠 #6, #7). Les
   défauts (`threshold` cosine, `k`, `resolution_*`, `min_sub_size`) sont « calés
   sur TikTok FR ~1,5 k » et **incohérents entre les 3 modules**. → Dériver le seuil
   d'arête d'un percentile de la distribution des cosinus k-NN ; rendre
   `min_sub_size` **relatif** à N ; centraliser les défauts par modèle.
5. **Détection de langue robuste + ne pas jeter le non-FR par défaut** (🟠 #10, #11).
   Repli heuristique limité à FR/EN/DE, défaut `'fr'`, et `XSTANCE_FR_ONLY` qui
   écarte le non-FR — à rebours du « multilingue = 1er ordre ». → langdetect en
   dépendance dure (ou repli élargi), défaut `'und'`, garder toutes les langues.

---

## Tableau des constats (trié par sévérité)

### 🔴 Casse sur un autre corpus

| # | fichier:ligne | extrait | pourquoi ça casse | fix générique |
|---|---|---|---|---|
| 1 | `pipeline/ingest/build.py:62-84`, `:129-134` · `pipeline/ingest/config.py:36-43` | `read_tiktok()` ; `readers = [("xstance",…),("tiktok",…)]` ; `TIKTOK_TEXT_COL = 141`, `TIKTOK_ENCODING = "cp1252"`, `delimiter=";"` | L'ingestion ne connaît que **2 sources en dur**, avec un schéma de colonnes figé au CSV TikTok 2025. Une nouvelle consultation ne peut pas être chargée sans **patcher le code**. | Descripteur de source déclaratif (encoding, délimiteur, colonnes id/text/ts) + un `read_generic(descriptor)`. Le corpus = une config, pas du code. |
| 2 | `pipeline/cluster/naming.py:16-28`, `:51-55` | `FRENCH_STOPWORDS = {"le","la",…}` passé à `TfidfVectorizer(stop_words=…)` | Le naming des thèmes (visible à l'utilisateur) filtre avec une **liste FR**. Sur un corpus DE/EN/ES, les mots-outils étrangers ne sont pas retirés → labels pollués (« the », « der »…). Le contrat impose explicitement de **dériver** les stopwords de la **document-frequency**. | Dériver les stopwords du corpus (`max_df`≈0.5–0.8, ou top-X% DF) ; optionnellement unir des listes par langue détectée. Langue-agnostique. |
| 3 | `pipeline/cluster/naming.py:30` | `_WORD_RE = re.compile(r"[a-zàâäéèêëîïôöùûüçœ]+", re.I)` | Tokenizer **Latin-only**. Sur un corpus en cyrillique / grec / arabe / hébreu / CJK, `findall` ne renvoie **rien** → tous les `keywords` vides → labels dégénèrent en « thème N ». Le clustering marche (embeddings), mais les thèmes deviennent **anonymes**. | Tokenisation Unicode : `regex` `\p{L}+` ou `re.compile(r"\w+", re.UNICODE)` (en excluant les chiffres). |
| 4 | `eval/coherence.py:29-61`, `:63` | `STOPWORDS` DE/FR/IT figés + `_WORD_RE = [a-zàâä…ßüäö]+` | Le module qui **mesure la cohérence et sert à CHOISIR le modèle de prod** a la même cécité Latin/3-langues. Sur un corpus hors DE/FR/IT (ou non-latin), la métrique de qualité est faussée/vide → le **choix du modèle n'est pas validé** pour ce corpus. | Idem #2/#3 (stopwords dérivés + tokenisation Unicode) dans la chaîne d'éval. |
| 5 | `pipeline/embed/embedder.py:30` + `pipeline/cluster/build.py:274` | `DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"` ; `Embedder()` sans modèle | Le pipeline **batch** embedde par défaut avec e5-small, dont le VERDICT du contrat dit qu'il **clusterise par langue** (NMI cluster↔langue=0.81, topic=0.05) → multilingue cassé. Le backend live, lui, utilise nomic-v2 (cache) : **incohérence** entre les deux chemins. | `DEFAULT_MODEL_ID` = le winner validé (`nomic-v2`), ou rendre le modèle explicite obligatoire pour `build_payload`. |

### 🟠 Dégrade silencieusement

| # | fichier:ligne | extrait | pourquoi ça dégrade | fix générique |
|---|---|---|---|---|
| 6 | `pipeline/cluster/hierarchy.py:36-42` · `pipeline/cluster/build.py:238-239` · `pipeline/cluster/knn.py:55-56` · `backend/recluster.py:61-67` | `RESOLUTION_MACRO=1.0/SUB=3.0`, `MIN_SUB_SIZE=15` « calés sur TikTok FR » ; `threshold=0.84` (build) vs `0.80` (knn) vs `0.60` (backend) ; `k=8`/`10`/`12` | Défauts **corpus-calibrés** et **incohérents entre 3 modules**. Le seuil cosine dépend du **modèle** (0.84 e5 ≠ 0.60 nomic) et de la **densité du corpus** ; figé, il sur/sous-connecte le graphe sur un autre corpus → trop de clusters ou un seul gros. | Dériver `threshold` d'un percentile de la distribution des cosinus k-NN ; `k` ∝ log(N) ; centraliser les défauts **par modèle** (p.ex. dans le registry). Exposés en knob = OK, mais le **défaut** doit être dérivé. |
| 7 | `pipeline/cluster/hierarchy.py:42` · `backend/recluster.py:67` | `DEFAULT_MIN_SUB_SIZE = 15` ; `min_sub_size=18` | Taille mini **absolue**. Sur un petit corpus (quelques centaines d'avis) elle fusionne tout en un sous-thème ; sur un très gros corpus elle laisse de la poussière. | Rendre **relatif** : `max(5, round(frac * N))` (frac dérivé), ou borne adaptative. |
| 8 | `backend/server.py:26-39`, `:46-52` | bornes knobs `dedup 0.90–0.99`, `threshold 0.40–0.85`… en dur dans `KNOBS` **et** `Field(ge=…, le=…)` | Les **bornes** des sliders sont tunées pour nomic-v2/TikTok. Un autre modèle/corpus peut nécessiter une valeur **hors clamp** → pydantic **rejette** (422) une valeur pourtant légitime. | Dériver les bornes du modèle/corpus (ou les élargir/relâcher la validation). Les bornes aussi doivent être data-driven, pas figées. |
| 9 | `pipeline/cluster/scoring.py:22`, `:43` | `DUP_THRESHOLD = 0.93` | Seuil cosine « quasi-doublon » **enfoui**, utilisé pour `diversity`. Dépend de l'échelle de similarité du modèle ; non exposé, non dérivé → la diversité affichée varie selon le modèle sans contrôle. | Exposer en knob et/ou lier au `dedup` ; dériver de la distribution intra-cluster. |
| 10 | `pipeline/ingest/lang.py:19-26`, `:38` | repli `_STOP` = FR/EN/DE seulement ; `detect_lang(…, default="fr")` | Si `langdetect` absent, la détection ne couvre que 3 langues et **retombe sur 'fr'** ; un corpus ES/PT/PL est mal étiqueté → le filtre `lang` et la cohérence par-langue se trompent. | `langdetect` (ou fasttext-lid) en dépendance dure ; défaut `'und'` plutôt que `'fr'` ; élargir le repli. |
| 11 | `pipeline/ingest/config.py:56-58` · `pipeline/ingest/build.py:51` | `XSTANCE_FR_ONLY = … != "1"` (défaut : FR only) | Par **défaut** on jette tout le non-FR — à rebours de « multilingue = 1er ordre ». Acceptable comme défaut démo, mais piège pour un corpus multilingue. | Défaut = garder toutes les langues ; le sous-ensemble par langue reste un knob explicite. |
| 13 | `pipeline/cluster/io.py:32`, `:60` | `lang: str = "fr"` ; `lang=get("lang", "fr")` | Quand le champ `lang` manque dans le JSONL, on suppose **FR**. Sur un corpus importé sans langue, tout devient « fr » → cohérence par-langue et filtres biaisés. | Défaut `"und"` ; (re)détecter si absent. |

### 🟢 Cosmétique / acceptable (à connaître)

| # | fichier:ligne | extrait | note |
|---|---|---|---|
| 12 | `pipeline/ingest/synthetic.py:1-41` | échantillon synthétique « mobilité urbaine » FR | Uniquement **fallback de dernier recours** quand aucune source réelle. Corpus-flavoré mais hors chemin de prod. OK. |
| 14 | `frontend/index.html:2` | `<html class="dark" lang="fr">` | Attribut HTML `lang` figé ; cosmétique (a11y). Pourrait suivre la langue dominante du corpus. |
| 15 | `pipeline/cluster/palette.py:4-17` | `PALETTE` de 20 couleurs, `color_for = id % len` | Au-delà de 20 communautés, les couleurs **se répètent** (collisions visuelles). Générique mais limité ; générer la palette (HSV équiréparti) selon le nombre de thèmes. |
| 16 | `pipeline/ingest/config.py:50` | `HASH_SALT = os.environ.get("AGORA_HASH_SALT", "agora-an-2026")` | Sel par défaut littéral, mais **surchargeable par env**. Acceptable. |

> **Note sur `eval/tiktok_diag.py`** : truffé de littéraux « tiktok » (regex de la
> famille lexicale, tokens…). C'est un **artefact d'analyse** dédié au diagnostic de
> CE corpus (cf. brief : ignorer les rapports/diagnostics). **Non compté** comme bug
> de prod — mais à ne pas généraliser tel quel. De même `backend/build_cache.py`
> (`--source tiktok`, `--lang fr` par défaut) est le script qui **fabrique le cache
> de démo** : ses défauts TikTok/FR sont attendus pour la démo, mais ce sont bien des
> **défauts corpus-spécifiques** (à paramétrer pour une autre consultation).

---

## Ce que je n'ai pas pu vérifier (honnêteté)

- **Pas d'exécution** : pipeline non lancé sur un corpus non-FR/non-latin. L'effet
  « labels vides » (#3) est **déduit du regex**, pas mesuré. Idem l'impact réel des
  magic numbers (#6) : jugé via le contrat et la lecture, non quantifié par ablation.
- **Dépendances de déploiement** : je ne sais pas si `langdetect` / `faiss` sont
  garantis en prod ; la gravité de #10 dépend de leur présence (repli FR/EN/DE sinon).
- **Frontend** : revue limitée à `src/` (greps + `App.tsx`/`api.ts`/`KnobsPanel`).
  Les knobs/bornes viennent du backend `/params` → le front est **data-driven** (bon
  point) ; je n'ai pas audité chaque composant en détail.
- **Le scan `genericity_scan.py`** est heuristique (regex) : il peut rater un
  hardcoding exprimé autrement, et sur-signaler des littéraux dans les commentaires.
  Les constats du tableau, eux, sont issus de la **lecture** du code, pas du seul scan.
- **Choix « bug vs intentionnel »** pour l'ingestion (#1) : techniquement c'est un
  adaptateur de source ; je le classe 🔴 car le contrat vise « des centaines de
  consultations » et il n'existe **aucun** chemin générique d'ingestion.

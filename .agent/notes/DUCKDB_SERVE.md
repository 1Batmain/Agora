# DuckDB comme moteur de LECTURE de `/avis_list` — verdict

**Contexte.** Audit code #1 : `/avis_list` scanne `avis.json` en Python et **replie l'Unicode**
(NFD + casefold) de CHAQUE avis à CHAQUE requête de recherche. Coût O(N)·fold par requête `q`.

**Ce qui a été fait (incrémental, réversible).**
- `backend/bake_duckdb.py` : bake un `analysis.duckdb` *dérivé* par dataset (tables `avis`,
  `claims`, `themes`, `meta`) — index de LECTURE, jamais source de vérité. `text_fold`
  précalculé, `stance` jointe depuis `claim_stance.json`, `payload` = entrée `avis.json`
  verbatim (⇒ item reconstruit à l'identique). Index sur `claims(filter_theme|stance|avis_rank)`.
  Index FTS BM25 best-effort (voir limites).
- `avis.avis_list_duckdb` : miroir SQL de `avis.avis_list` (mêmes prédicats thème/stance/`q`,
  `contains(text_fold, needle)` = même sémantique sous-chaîne que le `in` Python).
- `server.get_avis_list` : **route vers DuckDB UNIQUEMENT quand `q` est présent** ; sinon
  fallback RAM. Index absent/périmé/`duckdb` non installé → fallback (parité prouvée).
- `analysis_store.avis_duckdb_con` : connexion read-only cachée, validée par **signature de
  taille** des sources (table `meta`), pas par mtime — robuste au `git checkout`/`reset --hard`
  d'un `.duckdb` promu.

**Bench (`research/bench_duckdb_avis.py`, republique-numerique, 2724 avis).**

| requête | fallback | duckdb | gain |
|---|---:|---:|---:|
| `q=internet` (258 hits) | ~180 ms | ~7 ms | **×25** |
| `q=reglementation` (11) | ~200 ms | ~8 ms | **×25** |
| `theme_id=n0` (sans q) | ~7 ms | ~24 ms | ×0.3 |
| `stance=favorable` (sans q) | ~3 ms | ~4 ms | ×0.7 |

**Verdict : OUI pour la recherche `q` (objectif ×10+ atteint, ×25 mesuré), NON hors `q`.**
DuckDB ne gagne que là où il y a du travail lourd par ligne (le fold Unicode). Pour un simple
test d'appartenance thème/stance, l'itération dict en RAM (court-circuit au 1ᵉʳ claim, zéro
fold) est déjà optimale et bat le coût fixe d'une requête SQL (~15-25 ms : 2 `execute` +
reconstruction JSON). D'où le **routage sur présence de `q`** : le gain là où il compte, aucune
régression ailleurs. À 22k avis le fallback `q` scalerait à ~1,5 s (fold linéaire) tandis que
DuckDB reste ~7 ms → gain qui croît avec N.

**Limites.**
- **FTS BM25 bâti mais NON utilisé sur le hot path** : BM25 est tokenisé (match par mot),
  incompatible avec la parité sous-chaîne exigée (`reglement` ⊂ `reglementer`). L'index est
  créé quand l'extension `fts` charge (best-effort, ignoré offline/CI) comme capacité pour une
  future recherche *classée* opt-in — pas pour remplacer `contains`.
- Signature de fraîcheur = taille d'octets des sources : un edit de même taille exact (quasi
  impossible en JSON) ne serait pas détecté. Acceptable pour un cache à rebake explicite.
- Le `.duckdb` double ~le poids de `avis.json` sur disque (payload verbatim) — cache jetable.

**Intégration au build.** `analysis.duckdb` vit sous `backend/cache/*/analysis/` (**gitignoré**),
donc jamais commité. Il est (re)baké **juste avant la promotion de cache DEV→PROD** (seule route
de la prod), best-effort. En dev/ad hoc :
`uv run --extra collect python -m backend.bake_duckdb <dataset|--all>`.

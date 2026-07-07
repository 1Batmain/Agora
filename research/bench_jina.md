# Verdict R&D — JINA comme embedder vs nomic-v2 (le servi)

**Date** : 2026-07-07 · **Branche** : `research/bench-jina` · **Demandé par** : Bob
**Réponse : NON — on garde nomic-v2.** Décidé par la **LICENCE**, pas par la qualité.

---

## 0. LICENCE d'abord (le point rédhibitoire, avant toute métrique)

Le protocole l'exige : vérifier la licence des poids AVANT de benché. Résultat :

| Modèle Jina | Multilingue FR/DE/IT ? | Licence des poids | Adoptable Agora ? |
|---|:--:|---|:--:|
| **jina-embeddings-v3** (570M, le flagship) | ✅ oui (89 langues) | **CC-BY-NC-4.0** (usage **NON-COMMERCIAL** ; prod = API payante ou licence commerciale Jina) | ❌ **NON — rédhibitoire** |
| **jina-embeddings-v2-base-de** (161M) | ❌ **bilingue DE-EN** (pas FR/IT) | Apache-2.0 ✅ | ❌ mauvais outil (voir §3) |
| nomic-embed-text-v2-moe (le servi) | ✅ oui | **Apache-2.0** ✅ | ✅ (déjà en prod) |

> Sources vérifiées le 2026-07-07 : carte HF `jinaai/jina-embeddings-v3` (license `cc-by-nc-4.0`),
> API HF `jinaai/jina-embeddings-v2-base-de` (`license: apache-2.0`, sha `3f9eede…`).

**Le vrai JINA multilingue (v3) est non-commercial.** Agora est un produit d'intérêt public
mais réel : une licence NON-COMMERCIALE sur l'embedder du cœur est une interdiction juridique.
**Aucun chiffre de qualité ne peut lever cette interdiction.** C'est déjà un NON pour v3.

Reste l'Apache : v2-base-de. Mais il n'est **pas** multilingue (DE-EN), donc inapte à
regrouper FR/DE/IT par thème — la contrainte de 1er ordre d'Agora. Détail mesuré en §3.

---

## 1. Ce qui a été benché, et comment (protocole IDENTIQUE au témoin)

Même corpus, même partition, mêmes métriques que le verdict nomic (`research/quality_report.md`) :
gold **x-stance** équilibré (thème × langue), **rang-kNN k=15 → Leiden res=1.0, seed=42**,
CPU, bootstrap 4×. Scorecard brute complète : `research/quality_report_jina.md`.
Runner : `research/run_bench_jina.py` (voir §4 pour les contorsions techniques).

- **nomic-v2** (témoin servi) et **e5-small** (référence du piège langue) : via l'`Embedder`
  partagé — donc **re-validation du témoin sur le même harness**.
- **jina-v3** : le code custom Jina (ST `trust_remote_code`) **casse** sur le transformers
  du repo ; seul le **port transformers-natif** `tomaarsen/jina-embeddings-v3-hf` charge (§4).
  Embed = `AutoModel` + mean-pooling + L2-norm, CPU. On le mesure **bien que barré par la
  licence**, pour chiffrer ce qu'on s'interdit (« mesurer avant de trancher »).

Corpus : **n=2214**, langues `{de:738, fr:738, it:738}` (parfaitement équilibrées), 6 thèmes.

---

## 2. Résultats chiffrés (gold x-stance)

| Métrique | sens | **jina-v3** | **nomic-v2** (témoin) | e5-small (piège) |
|---|:--:|:--:|:--:|:--:|
| **NMI(cluster, langue)** | ↓ | **0.003** | **0.008** | 0.812 |
| Pureté linguistique | ↓ | 0.376 | 0.384 | 0.997 |
| **NMI(cluster, thème)** | ↑ | **0.482** | **0.407** | 0.048 |
| Pureté thématique | ↑ | 0.742 | 0.649 | 0.215 |
| Cohérence NPMI | ↑ | -0.056 | -0.106 | -0.159 |
| Silhouette (cosine) | ↑ | 0.113 | 0.069 | 0.072 |
| Modularité (Leiden) | ↑ | 0.726 | 0.613 | 0.679 |
| Stabilité (ARI bootstrap) | ↑ | 0.776 | 0.723 | 0.897 |
| Nb clusters | · | 15 | 13 | 6 |
| Dimension | · | **1024** | **768** | 384 |
| Chargement (s) | ↓ | 4.8 | 13.2 | 6.8 |
| **Latence (ms/texte, CPU)** | ↓ | **210.5** | **44.6** | 8.9 |

**Re-validation du témoin ✅** : nomic-v2 reproduit **exactement** le verdict existant
(NMI langue 0.008, NMI thème 0.407) ; e5-small aussi (0.812 / 0.048). Le harness est fidèle.

**Lecture honnête — jina-v3 est MEILLEUR en qualité, sans tomber dans le piège e5 :**
- Récupère mieux le thème : **NMI thème 0.482 vs 0.407** (+0.075, ~+18 % relatif), pureté
  thématique 0.742 vs 0.649.
- Mixe aussi bien/mieux les langues : NMI langue 0.003 vs 0.008 (les deux ≈ 0 → **PAS** de
  ségrégation par langue, à l'opposé d'e5=0.812). Ce n'est pas une victoire dégénérée type e5.
- Cohérence meilleure (-0.056 vs -0.106).

**Mais deux coûts durs, avant même la licence :**
- **Vitesse : ~4.7× plus lent** (210 vs 45 ms/texte, CPU). Re-embedder un cache servi
  (~30 min CPU en nomic) passerait à **~2,3 h/dataset**.
- **Dimension 1024 vs 768** → caches `claims_emb.npz`/`embeddings.npy` +33 % de volume,
  plus de RAM au service. Empreinte modèle ~572M params (~2,3 Go fp32) vs nomic MoE ~475M.

---

## 3. Le seul JINA Apache (v2-base-de) : mauvais outil ET code rotté

`jina-embeddings-v2-base-de` est **bilingue DE-EN** : aucun entraînement FR/IT. Sur un corpus
FR/DE/IT il se comporterait comme e5 (**cluster par langue**, faute de comprendre FR/IT) —
exactement le piège documenté. Ce n'est pas un embedder multilingue, donc hors-sujet pour Agora
(produit à primauté FR, trilingue).

En prime, **son code custom est incompatible avec le transformers du repo** — cascade
d'imports morts rencontrée en tentant de le charger : `transformers.onnx` (retiré),
`find_pruneable_heads_and_indices` (retiré de `pytorch_utils`), `config.is_decoder` (absent).
Le faire tourner exigerait de **vendoriser un vieux transformers** → dette de maintenance,
argument supplémentaire **contre** l'adoption. Non benché (mauvais outil de toute façon).

---

## 4. Notes techniques (reproductibilité)

- **v3 via sentence-transformers `trust_remote_code`** : casse aussi (le custom
  `xlm-roberta-flash-implementation` référence `all_tied_weights_keys`, API transformers
  modifiée). **Seul** le port natif `tomaarsen/jina-embeddings-v3-hf` (classe transformers
  intégrée `jina_embeddings_v3`, **sans** code distant) charge proprement sur ce repo.
- **Adaptateur de tâche v3** : mesure faite avec la config **générique** (pas de LoRA
  'separation'/'retrieval' activé) + mean-pooling. Un adaptateur dédié pourrait encore
  améliorer v3 — ce qui **renforce** le constat qualité, sans changer le verdict licence.
- **Registre** : `jina-v2-base-de` ajouté à `pipeline/embed/registry.py` (Apache, épinglé)
  pour tracer la piste Apache ; il n'est pas runnable ici (§3) et n'est PAS le défaut.
- Aucun cache servi ré-embeddé. Aucun changement prod/`main`.

---

## 5. Verdict

**NON — on conserve nomic-v2 comme embedder servi.**

1. **Licence (décisif)** : le JINA qui vaut le coup (v3, multilingue) est **CC-BY-NC-4.0,
   non-commercial → juridiquement inadoptable** dans Agora. Le JINA libre (v2-base-de) est
   **bilingue DE-EN**, pas un embedder multilingue.
2. **Même sans la licence, le coût d'adoption n'est pas gagné** : ~4.7× plus lent (CPU),
   dim 1024 (+33 % de caches), et il faudrait **re-embedder TOUS les caches servis + re-valider
   le témoin** — pour un gain qualité réel (+0.075 NMI thème) mais **modéré** au regard de ce prix.
3. **La leçon e5 se prolonge** : e5 nous a appris à ne pas laisser une bonne métrique **interne**
   (silhouette) dicter le choix ; jina-v3 nous apprend à ne pas laisser un bon **score de qualité**
   le dicter quand la **licence** l'interdit. La mesure reste honnête dans les deux cas.

**Si un jour** Jina publiait un modèle multilingue sous licence permissive (Apache/MIT) et que
le débit CPU devenait acceptable, la piste mériterait un re-test (le gain qualité est réel).
En l'état : **rien à changer**.

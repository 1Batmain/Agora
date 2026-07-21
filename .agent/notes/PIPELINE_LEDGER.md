# Registre R&D — pipeline de synthèse (méthodes testées)

Synthèse vivante des verdicts. Règle : **mesurer avant d'adopter**. Invariants : **verbatim**
(zéro reformulation) + **souveraineté** (embeddings locaux). Sources détaillées : les notes
`.agent/notes/*` et `research/*_note.md` citées. Statuts : ✅ adopté · ❌ écarté · 🔶 ouvert/partiel.

## 1. Extraction des positions (claims)
- ✅ **Verbatim, pas paraphrase** — invariant ; l'ancrage rejette tout span non exact. (`argmine_verbatim_note`)
- ✅ **Prompt relâché (B) vs strict (A)** — +16 % de positions captées, à verbatim égal. (`extract_ab_note`)
- 🔶 **Anti-sur-segmentation (v2)** — corrige la sur-découpe (12 avis), mais coût rappel jamais re-mesuré. (`extract_v2_note`, `extract_b2_note`)
- 🔶 **Cible de stance à l'extraction (v3)** — obligatoire elle droppait des claims (88→61 %) ; gardée optionnelle. (`extract_v3_note`)
- ✅ **Batching (N avis/appel)** — prod ; delta qualité mono↔lot jamais mesuré (angle mort).
- 🔶 **Repli « avis entier » sur extraction vide** — bug de précision : injecte du charabia (proposition Fable P1).

## 2. Espace d'embedding
- ✅ **nomic-v2 (défaut)** — qualité/souveraineté, Apache, multilingue.
- ❌ **jina-v3** — licence CC-BY-NC (non commerciale). (`bench_jina`)
- ❌ **mistral-embed (API)** — regroupe par langue, pas par thème ; casse la souveraineté. (`bench_mistral`)
- 🔶 **arctic-l** — marginalement meilleur, pas assez pour re-embedder le cache. (`ab_embedder_note`)
- ✅ **Recentrage (anti-anisotropie)** — +19 % ARI, corrige la hubness, zéro paramètre. (`EMBEDDING_SPACE`)
- ❌ **Blanchiment (whitening)** — détruit le signal ; recentrer oui, blanchir non. (`EMBEDDING_SPACE`)
- ✅ **Vecteurs L2-normalisés** — pas de perte (nomic normalise déjà). (`embed_norm_note`)

## 3. Clustering
- ✅ **Leiden sur graphe k-NN (défaut)** — communautés, hiérarchique, robuste.
- 🔶 **UMAP-5D → HDBSCAN (contender)** — dispo en comparaison console. (`HDBSCAN_NOTE`)
- ❌ **Transformer le poids d'arête k-NN** — gain dans le bruit ; garder le cosinus brut. (`knn_weight_note`)

## 4. Hiérarchie des thèmes
- ❌ **Seuil de dispersion tau + RES_LADDER** — bascule sur 2 claims d'écart (pile ou face). (`HIERARCHY_TAU`)
- ✅ **k comme robinet de zoom** — balaie le voisinage au lieu de dériver k du nombre de claims. (`HIERARCHY_KMOD`, `HIERARCHY_LAYERS`)
- ❌ **sauce_magique (re-coupe macro)** — adoptée puis retirée : réparait un artefact d'anisotropie que le recentrage supprime. (`sauce_magique_note`, `HIERARCHY_LAYERS`)
- ✅ **Chaîne d'emboîtement (multi-niveaux)** — l'arbre suit toute la chaîne (tiktok 4→9→16). (`HIERARCHY_LAYERS`)
- ❌ **SBM emboîté MDL (Peixoto)** — égalité avec la chaîne sur le gold, bien plus lourd. (`sbm_vs_chain_note`)
- ❌ **Fusion post-clustering (sous-consolidation)** — pas de redondance géométrique au clustering. (`cluster_merge_note`)

## 5. Nommage des thèmes
- ✅ **c-TF-IDF (défaut) + centroïde + LLM local, switchables** — distinctif, tiré des réponses. (`NAMING_SWITCH_NOTE`)
- ✅ **Titres LLM « journaliste » + ancres mélangées** — neutre ET spécifique, ≤10 mots. (`backend/titles.py`)

## 6. Opinion, position & clivage
- ✅ **Position sur le SUJET du cluster** (vs cibles per-claim) — agrège net. (`stance_proto_note`)
- ✅ **Cible de stance (b)** — dédupe sans coût (panel aveugle). (`stance_target_ab_verdict`)
- ✅ **Pré-filtre de pertinence avant stance** — corrige le sur-classement « favorable » tangentiel. (`relevance_prefilter_note`)
- 🔶 **Émergence d'arguments par densité** — proto ; « émerger de la donnée » vs sélection LLM. (`emerge_note`)
- 🔶 **Argument mining servi = paraphrase** — invariant cassé ; V-SELECT verbatim bat la paraphrase. (`argmine_verbatim_note`)

## 7. Frontières ouvertes (mesurées, non résolues)
- 🔶 **Redondance entre thèmes frères** — sémantique, PAS géométrique (ni centroïde ni kNN) → juge LLM. (`sibling_redundancy_note`)
- 🔶 **Granularité inégale / rattachements 50/50** — un k global trop fin ici, trop grossier là (harcèlement à cheval 193/192).
- 🔶 **Rappel d'extraction (~6 %)** — concessions minoritaires sacrifiées ; propositions Fable en attente de banc. (`var/audit/proposition-extraction.md`)
- 🔶 **k-sweep vs γ-sweep** — k est un PROXY de la résolution Leiden (le seuil tombe à 0 dès k≈150 → pure densification). Tester le balayage direct de γ sur graphe fixe (1 knn au lieu de 20).

_Registre à tenir à jour : chaque nouveau verdict ajoute/déplace une ligne._

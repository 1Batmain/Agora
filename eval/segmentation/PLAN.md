# PLAN — extraction multi-thématique (synthèse de synchro)

> Document de synchro architecte↔Bob. Ce qu'on s'apprête à implémenter, et POURQUOI.
> Mis à jour au fil des décisions. Principe directeur : **généricité / zéro hardcoding**
> (cf. `queue/cross-lane.md`) — tout dérivé des données, langue-agnostique.

## 1. Le problème (recadrage produit)
Le clustering actuel **partitionne** : 1 avis → 1 thème. Or un témoignage citoyen peut
soulever **plusieurs problèmes** (« harcèlement + addiction + image du corps »). On veut
**1 avis → N thèmes**, puis **pondérer par fréquence** (% de citoyens) + **graphe de
co-occurrence** (quels thèmes reviennent ensemble). Cause technique du blocage : l'embedding
d'un avis entier est une **moyenne** de ses aspects → il atterrit *entre* les thèmes.

## 2. Le pipeline cible
```
avis → SEGMENTATION sémantique (1 avis → N segments ~mono-thème)
     → embedding de chaque segment (nomic-v2, le winner multilingue mesuré)
     → clustering des segments → thèmes
     → agrégation : fréquence par thème (% citoyens, via weight) + co-occurrence
```
Le clustering/embedding (nomic) est **déjà fait et validé**. Le maillon en R&D = la
**segmentation**.

## 3. Ce qui est DÉJÀ mesuré (eval-as-truth)
- Segmenter par **trajectoire d'embedding** (PELT/change-point sur vecteurs-tokens) =
  **médiocre** sur transitions naturelles : F1_multi 0.44, Pk 0.28, **sur-coupe** (70% des
  avis mono). Faisabilité token-embeddings nomic : OK.
- Segmenter par **ATTENTION** (chute du flux inter-blocs) = **bien meilleur** : e5-base
  F1_multi **0.77**, Pk **0.15**, **précision 90%**, faux-positifs mono **14%**. Contrôle
  décisif : bat la trajectoire d'embedding du *même* modèle → c'est le **signal attention**.
  Signal **diffus** sur les têtes des couches **basses-moyennes**. Attention extractible
  trivialement sur e5/bge (eager) ; sur nomic = custom (Wqkv+rotary, non requis si e5 segmente mieux).
- Mode d'échec : rate les **transitions douces** (« et… / du coup… »), rappel 67%.

## 4. L'optimisation décidée : segmenteur APPRIS (pas réglé à la main)
Au lieu de choisir « lowmid / mean / seuil c » à la main, on **apprend** la combinaison :
- **Features par position** : flux d'attention inter-blocs `cross_{L,H}(p)` pour chaque
  **couche × tête** (gelées) **+** dérive d'embedding. Langue-agnostique (zéro lexique — on a
  écarté l'idée du lexique de connecteurs).
- **Modèle** : **régression logistique** (poids interprétables = « où vit le signal ») +
  **gradient boosting** (perf). Modèle d'embedding/attention **GELÉ** — on n'entraîne qu'une
  **tête légère** (CPU, cheap, reproductible).

## 5. Discipline ML — les NON-NÉGOCIABLES
1. **Train sur du RÉEL externe** (humain, diversifié, multilingue) — **JAMAIS** sur notre
   synthétique (sinon on apprend le style du générateur, pas la cohésion thématique).
2. **Test** = notre gold de témoignages (in-domain) + held-out réel.
3. **Validation croisée stricte** (avis jamais vus) — sinon auto-illusion.
4. **Transfert cross-langue** (train FR → eval DE/IT) = preuve de généricité.

## 6. Données (recon en cours)
Candidats réels : **SemEval ABSA** (avis multi-aspect, multilingue — le plus proche de notre
domaine), **WikiSection** (sections = thèmes, EN/DE), **Wiki-727K** (benchmark seg), Wikipédia
maison (FR/IT/DE), reviews/forum/transcripts. → fiche comparative `DATASETS_RECON.md` puis choix.

## 7. Intégration prod (après validation)
Segmenteur appris → branché dans le backend (`/recluster` sur segments) → viz : thèmes
**pondérés par fréquence** + **graphe de co-occurrence**, **switchable** avec le mode clustering
actuel. Souveraineté maintenue (modèles locaux/EU, features internes).

## 8. État & prochaine étape
Mergé sur `agora` : bench segmentation + gold (305) + attention validée. **Prochain pas** :
acquisition de données réelles (recon) → segmenteur appris en CV → si transfert OK → câblage prod.

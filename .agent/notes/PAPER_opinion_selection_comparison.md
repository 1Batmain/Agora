# Étude « Opinion Selection for Online Deliberation » vs Agora — comparaison (read-only)

> Agent de recherche, 2026-07-26. Brief : `var/paper-comparison-brief.md`.
> **Règle du témoin de contrôle** : chaque affirmation est marquée **[C]** CONFIRMÉ (page du PDF
> ou `fichier:ligne`) ou **[I]** INTERPRÉTÉ (raisonnement de l'agent). Je n'ai touché à aucun code.
>
> Papier : Hafid, Berriche, Cointet (Sciences Po médialab) — *Algorithmic Approaches to Opinion
> Selection for Online Deliberation: A Comparative Study*, Pluralistic Alignment Workshop @ ICML
> 2026. PDF 29 p. lu en entier (openreview kMeny2IJF0). Code auteurs :
> github.com/SalimHFX/Algorithms-for-Opinion-Selection-in-Deliberation (non exécuté).

---

## 1. Résumé fidèle du papier

### 1.1 Problème et cadre
- **Tâche** : en délibération en ligne (Polis, Remesh, Community Notes), **sélectionner un petit
  sous-ensemble de k opinions représentatives** parmi de nombreuses contributions en texte libre,
  pour produire un rapport digeste. [C p1 Abstract, p1-2 Intro]
- **Danger central** identifié dès l'abstract : forcer des stratégies de **consensus** revient à
  *« ignorer ou aplatir les préférences conflictuelles, ce qui peut effacer les voix minoritaires
  et réduire la diversité »*. [C p1 Abstract]
- **Cadre formel** : vote multi-gagnant par **approbation**. Matrice d'approbation binaire
  `A ∈ {0,1}^{n×m}` (n users, m opinions) ; `A_{u,s}=1` ssi user u approuve l'opinion s. Chaque
  user approuve un ensemble `A_u ⊆ [m]`. On sélectionne `S ⊆ [m]`, `|S|=k`. [C p3 §3.1, p4 §4.1]
- Les votes manquants sont **inférés** (Remesh est une matrice creuse) ; probabilités seuillées à
  0.5 pour binariser. `0` = désapprobation explicite, **pas** un vote manquant → distance de
  **Hamming normalisée** (pas Jaccard, qui ignore les zéros partagés). [C p5, p2 note 2]

### 1.2 DiverseBJR (la contribution algorithmique)
Empile trois propriétés issues du choix social [C p3-4 §3.2] :
- **JR — Justified Representation** (Aziz 2017) : tout groupe « assez grand » (≥ n/k users) qui
  partage au moins une opinion approuvée en commun a droit à ≥1 opinion dans S. [C Def 3.1 p4]
- **BJR — Balanced JR** (Fish 2024) : renforce JR en imposant que chaque opinion sélectionnée
  représente **à peu près le même nombre de participants** (budgets de représentation ⌊n/k⌋ ou
  ⌈n/k⌉). Corrige le défaut de JR — *« sélectionner très peu d'opinions très populaires suffit à
  satisfaire JR »* (JR = deux tiers de la population représentés par un tiers du slate). [C Def 3.2
  p4, Appendice B p12-13]
- **ε-diversité** (nouveauté) : parmi les slates BJR-valides, S est **localement ε-diverse** si on
  ne peut pas remplacer une opinion par une non-sélectionnée qui ne soit PAS un ε-voisin d'une déjà
  choisie (ε-voisin = distance de Hamming sur vecteurs d'approbation ≤ ε). Notion **clone-robuste**
  (Procaccia 2025) : la sortie doit être ~invariante à la duplication d'une opinion. [C Def 3.3/3.4
  p4]
- **Algorithme 1** (Appendice C, p15) : procédure gloutonne 2-étages type GreedyCC. Étage 1 =
  couvrir les budgets BJR (départage par diversité : moins d'ε-voisins, plus d'approbateurs
  « uniques ») ; étage 2 = remplir les slots restants par une règle de score `f`, en marquant les
  ε-voisins inéligibles **seulement si** un simulateur glouton (Algorithme 2, p14) certifie que les
  budgets BJR restent satisfiables. `ε=0.8` en défaut. [C p6, Algo 1/2, Appendice H p23]
- **Honnêteté des auteurs** : ils *« ne cherchent pas à prouver l'existence de DiverseBJR pour toute
  instance où BJR tient »* ; le simulateur est un test de **suffisance** (True = certificat
  constructif ; False = non concluant). [C p4, Algo 2 p14]

### 1.3 Métriques démocratiques (la lentille d'évaluation — la partie la plus réutilisable)
Cinq métriques sur la matrice d'approbation, avec un partitionnement des users en **groupes
politiques** `𝒢 = {G_1..G_γ}` (auto-déclaré : *very/slightly conservative/liberal*). [C p4-5 §4.1]
1. **Sous-représentation individuelle** `U_all(S) = 100·(1/n)·Σ_u r̄_u(S)`, où `r̄_u=1` si u
   n'approuve AUCUNE opinion de S. [C eq 1 p4]
2. **Sous-représentation de groupe** `median_U(S) = median{U_g(S)}` sur les groupes (médiane, pas
   Rawlsian max-min, pour la robustesse aux outliers). [C eq 2 p5]
3. **Consensus** `= max_{s∈S} min_{g} (part d'approbation de s dans le groupe g)` = **maximin
   inter-groupes** : « existe-t-il ≥1 opinion sélectionnée qui satisfasse tous les groupes ? ». [C
   eq 3 p5]
4. **Coverage gap** `CG(S) = max_{o∉S} min_{s∈S} d(o,s)` : la plus grande distance entre une opinion
   NON-sélectionnée et sa plus proche sélectionnée. Grand CG = un pan de l'espace d'opinions ignoré.
   JR/BJR **ne garantissent PAS** un petit CG. [C eq 4 p5]
5. **Opinion redundancy** `= (Σ_g (|C_g|−1)) / k` : fraction d'opinions redondantes DANS S (les
   `C_g` sont des groupes-clones reliés par `d(i,j)≤ε`). Redondance non désirée dans un petit S. [C
   eq 5 p5]

### 1.4 Données, baselines, résultats
- **Données** : *« Polarized Issues »* de Remesh — dialogues collectifs réels, échantillons N300
  représentatifs du public US, sujets clivants (droit de manifester), avec **orientation politique
  auto-déclarée**. 6 questions Q1-Q6 (2 controversées Q1-Q2, 4 consensuelles Q3-Q6), 105-307
  participants chacune. Réplication sur un 2e jeu à votes **complets** (Virtual Citizen Assembly /
  Habermas Machine, Tessler 2024). [C p6 §4.3, Table 1 p9, Appendice E p20]
- **Baselines** : Random, Engagement (score = nb d'approbations = utilitariste), Bridging (CGA =
  accord maximin inter-groupes / Diverse Approval), Diversity (glouton min-CG), JR, BJR, DiverseBJR.
  Un baseline zéro-shot ChatGPT (gpt-4.1-mini) est **exclu des résultats principaux** car il renvoie
  souvent moins de k opinions. [C p6, Appendice G p22]
- **Résultat clé** : **aucune stratégie ne domine sur tous les critères**. À **petit k (≈1-5)**,
  DiverseBJR offre le meilleur **compromis représentation proportionnelle / diversité**. [C Abstract,
  p7 §4.4 (1), Takeaways p7]
- Table 2 (k=3, questions controversées) : DiverseBJR = sous-représentation individus **1.30**
  (meilleur, vs 22.9 Engagement, 19.7 Random) et groupes **1.39** (meilleur) ; CG 0.46 ; redondance
  0.17 ; consensus 0.58. Diversity a redondance 0.00 (par construction) mais consensus faible 0.46 ;
  Bridging a le meilleur consensus 0.62. [C Table 2 p9]
- Illustration (Appendice A p12, Fig 3) : sur k=3, **Engagement choisit 3 opinions du MÊME point de
  vue** ; DiverseBJR couvre ce point de vue **plus une perspective orthogonale** (« it did not change
  it at all »). Engagement sélectionne des opinions en clusters serrés (redondantes) ; DiverseBJR
  étale la sélection dans l'espace. [C p12, Fig 3 p12]
- **Scope explicitement exclu** : les résumés abstractifs par LLM. Les auteurs citent Zhu et al.
  2025 : *« faiblesses persistantes de la synthèse de délibération par LLM, dont la
  SOUS-REPRÉSENTATION des voix minoritaires »*. Ils se concentrent sur la **sélection**
  algorithmique, PAS sur la reformulation en un résumé cohérent. [C p3, §Scope]

---

## 2. Tableau de correspondance papier ↔ Agora (concept par concept)

| Concept papier | Agora | Nature du lien |
|---|---|---|
| **Paradigme** : sélectionner k opinions verbatim et s'arrêter [C p1] | **Clusteriser + synthétiser TOUT** ; pas de k d'opinions [C `analysis.py:1-21`] | Irréductible (§4.1) |
| **Entrée** : matrice d'approbation `A∈{0,1}^{n×m}`, votes (inférés) [C p3 §3.1] | **Texte libre → claims verbatim → embeddings**, **zéro vote** [C `.agent/README.md:14`] | Irréductible (§4.2) |
| **Groupes** politiques auto-déclarés `𝒢` [C p4] | **Aucun label de groupe** ; groupes = clusters sémantiques émergents [I] | Irréductible (§4.3) |
| **Consensus** = maximin inter-groupes d'approbation [C eq 3 p5] | **consensus** = cos moyen entre paires de claims (serrage sémantique) [C `analysis.py:54,118`] | **Même mot, sens opposé** (§5) |
| **Opinion redundancy** eq 5 (clones ε dans S) [C p5] | **Redondance entre thèmes frères** — axe reconnu NON mesuré ; moteur B la traite par ré-embedding [C `HIERARCHY_LAYERS.md:51-56`, `abstraction.py:1-15`] | Convergent, mesures différentes (§3, §5) |
| **Coverage gap** eq 4 (espace d'opinions couvert) [C p5] | Couverture implicite : **tout claim tombe dans un cluster** (partition exhaustive) [I sur `layers.py:flat_partition`] | Convergent, garanti différemment (§5) |
| **Représentation proportionnelle** (BJR : chaque opinion ~même nb de participants) [C Def 3.2] | **poids social** par thème = somme des poids d'avis (défaut 1.0, hérité de l'avis) [C `analysis.py:55,121`, `claims_endpoint.py:223`] | Divergent — poids ≠ garantie (§5) |
| **Diversité / voix minoritaires / long tail** [C p2] | **effusion** (équitabilité de Pielou) + **concentration** (top_share, Gini) au niveau global [C `analysis.py:762-773`] | Partiel — mesure la répartition, pas une garantie (§5) |
| **Bridging** (surfacer l'accord inter-groupes) [C p2, p6] | **Absent** ; travail stance/clivage identifie l'AXE de clivage (l'inverse) [C `PIPELINE_LEDGER.md:41-46`] | Absent chez nous (§6) |
| **Sélection de représentants** verbatim [C p1] | `_representatives` / `_hero_avis` : surface les claims/avis **les plus centraux** par thème [C `analysis.py:333-357`, `344`] | Analogie la plus proche — mais logique inverse (§5) |
| **Verbatim** (opinions = texte exact des users) [C Fig 3] | **Invariant verbatim** aux claims [C `.agent/README.md:40`] — MAIS synthèse servie paraphrase [C mémoire `argmining-verbatim-verdict`] | Convergent sur l'entrée, divergent sur la sortie (§5) |

---

## 3. Ce qu'on peut EMPRUNTER (concret, actionnable)

### 3.1 Leurs 5 métriques comme LENTILLE D'AUDIT démocratique d'Agora (le plus fort)
Notre culture est « mesurer avant d'adopter » (`README.md:35`) ; il nous manque justement une
**métrique de sous-représentation**. Leurs métriques sont calculables chez nous **à condition de
synthétiser une matrice d'approbation** — et c'est faisable, car **leur propre baseline LLM le fait
déjà** : le prompt zéro-shot dit *« Use semantic similarity to infer approvals »* [C p22 Appendice G].

Recette [I] :
1. Construire `A` : user u « approuve » l'opinion/claim i si `cos(embedding des claims de u, i) ≥ τ`
   (espace recentré, cf. `layers.centre`). Sans groupes politiques → sauter les métriques de groupe
   (median_U, consensus maximin, bridging), garder les métriques **agnostiques aux groupes** :
   `U_all` (sous-représentation individuelle, eq 1), **coverage gap** (eq 4), **opinion redundancy**
   (eq 5).
2. Appliquer ces trois métriques à l'ensemble effectivement SERVI par Agora — d'abord le **jeu de
   thèmes** (chaque thème = 1 « opinion » représentée par son représentant), puis les **représentants
   intra-thème** (`_representatives`, `analysis.py:333`).
3. Verdict recherché : *Agora laisse-t-il des citoyens sans aucun thème/représentant proche d'eux
   (U_all) ? sert-il des thèmes redondants (redundancy) ? couvre-t-il l'espace (CG) ?* → répond
   quantitativement à la question du §5, aujourd'hui sans réponse chiffrée.

### 3.2 `opinion redundancy` (eq 5) = la métrique qui manque à la frontière « redondance frères »
`HIERARCHY_LAYERS.md:51-56` et `PIPELINE_LEDGER.md:49` déclarent la redondance entre thèmes frères
comme un **axe non mesuré** (« piste : panel LLM ‘même sujet ?’ ou recouvrement kNN »). L'eq 5 du
papier EST une opérationnalisation prête à l'emploi : compter les groupes-clones (thèmes reliés par
`d(i,j)≤ε` dans un espace d'approbation ou d'embedding) et diviser par k. [C p5] → à adopter comme
**mesure** de la redondance frère-à-frère, indépendamment de savoir si on l'utilise pour agir. [I]

### 3.3 DiverseBJR pour choisir les représentants DANS un thème
Aujourd'hui `_representatives` classe par **centralité au centroïde × développement** (`analysis.py:344`)
→ il surface le claim le **plus typique**. C'est exactement le comportement « Engagement / cluster
serré » que le papier montre comme aplatissant (Fig 3, Appendice A) [C p12]. Piste [I] : construire
une **mini-matrice d'approbation locale** au thème (claim i « approuve » claim j si ε-voisins
sémantiques) et faire tourner l'Algorithme 1 pour sélectionner k représentants **proportionnels +
ε-diverses** — de sorte qu'une sous-position minoritaire mais réelle du thème obtienne un
représentant, pas seulement le medoïde. À BENCHER (panel aveugle) avant d'adopter, comme tout le
reste.

### 3.4 Vocabulaire honnête : renommer notre « consensus »
Point gratuit et immédiat : notre `consensus` = serrage sémantique (cos moyen entre paires), le leur
= accord de vote inter-groupes. Le code le note déjà à moitié (`analysis.py:847-851` : « cohesion =
cohésion SÉMANTIQUE, pas un accord d'opinion »). Le champ **`consensus` reste servi sous un nom qui
prête à confusion** avec l'acception démocratique du papier. [C `analysis.py:850-851`] Clarifier
côté contrat/front réduit le risque de sur-vendre un « consensus » qui n'en est pas un. [I]

---

## 4. Différences FONDAMENTALES (irréductibles)

1. **Sélection vs clustering+synthèse.** Eux : choisir k opinions verbatim et **s'arrêter** (le
   rapport digeste est fait ensuite, par des humains/LLM, explicitement HORS scope [C p3]). Nous :
   **regrouper tout** puis **synthétiser** (titres/opinion/arguments LLM par thème) [C `analysis.py:1`].
   Conséquence : on ne « rate » jamais un thème (partition exhaustive), mais on **reformule** — pile
   le régime où Zhu 2025, cité par le papier, signale la sous-représentation des minorités [C p3].

2. **Votes vs texte + embeddings.** Leurs algos (JR/BJR/DiverseBJR) et 4 de leurs 5 métriques sont
   définis SUR une matrice d'approbation `A` [C p3-5]. Agora n'a **aucun vote** [C `README.md:14`].
   Les appliquer suppose de **fabriquer** `A` par similarité sémantique (§3.1) — un choix de
   modélisation qui change l'ontologie (approbation inférée ≠ approbation exprimée). C'est faisable
   pour l'audit, mais ce n'est pas « les mêmes données ».

3. **Groupes étiquetés vs groupes émergents.** Leur représentation de groupe, leur consensus maximin
   et tout le bridging **exigent des attributs de groupe** (orientation politique auto-déclarée) [C
   p4-6]. Agora n'a pas de labels de groupe ; nos « groupes » sont des clusters sémantiques. → les
   métriques de GROUPE et le bridging **ne transposent pas** sans une source de labels externe.

4. **Petit k digeste vs profondeur navigable.** Leur cadre vit à **k≈1-5** (contrainte « rapport
   digeste ») [C p7]. Agora sert une **hiérarchie multi-niveaux drill-down** (tiktok 4→9→16) sans
   curseur de k [C `HIERARCHY_LAYERS.md:47-49`]. Objectifs produits différents : eux répondent
   « quelles 3 phrases montrer », nous « comment naviguer 30k contributions ».

---

## 5. Critique HONNÊTE d'Agora à travers leur lentille (la partie la plus utile)

**Question du brief : Agora écrase-t-il les minorités ?** Réponse nuancée — *pas au niveau des
thèmes, mais probablement au niveau de la synthèse et des représentants, et ce n'est pas mesuré.*

- **✅ Garde-fou réel au niveau du partitionnement.** Contrairement à la sélection de k opinions, le
  clustering d'Agora est **exhaustif** : chaque claim atterrit dans un cluster (`layers.flat_partition`).
  Une position minoritaire ne peut pas être « non sélectionnée » comme dans un slate de taille k —
  elle forme (ou rejoint) un thème. C'est structurellement PLUS protecteur que Engagement/JR sur le
  risque « U_all » (users sans aucune opinion proche). [I, sur `layers.py:flat_partition`]

- **⚠️ Mais trois fuites de minorité, non gardées :**
  1. **Représentants intra-thème centrés sur le medoïde.** `_representatives` trie par
     `cos(claim, centroïde) × développement` (`analysis.py:344`) → surface le claim le plus **typique**.
     Une nuance dissidente DANS un thème (la concession minoritaire, le « ça n'a rien changé » de la
     Fig 3) n'est jamais remontée. C'est exactement le motif Engagement que le papier montre
     aplatissant [C Fig 3 p12]. **Non mesuré, non gardé.**
  2. **Poids social = biais de popularité.** Le `weight` par thème (somme des poids d'avis,
     `analysis.py:55,121`) et le tri d'affichage par poids décroissant (`analysis.py:250`) amplifient
     les voix nombreuses. Sur un dataset où `weight` = engagement (likes TikTok — champ optionnel lu
     tel quel, `sources.py:171-176`), c'est **littéralement la baseline Engagement** que le papier
     classe pire que Random sur la représentation [C Table 2 p9 : Engagement 22.9 > Random 19.7].
     Un thème très « liké » mais minoritaire en nombre de personnes pèse lourd. **Non gardé.**
  3. **Synthèse LLM dans le régime signalé par Zhu 2025.** Titres/opinion/arguments sont générés par
     LLM par thème. Le papier se met explicitement HORS de ce régime *parce que* la synthèse LLM
     sous-représente les minorités [C p3]. Agora est en plein dedans, et l'invariant verbatim est
     **cassé** côté argument mining servi (paraphrase — mémoire `argmining-verbatim-verdict`).

- **⚠️ Consensus surchargé.** L'indice global `consensus` (`analysis.py:775-781`) est une moyenne
  pondérée-population du serrage sémantique intra-thème, shrinké bayésien. Il ne mesure **rien** de
  l'accord inter-groupes (il n'y a pas de groupes). Servir « consensus = 0.58 » à côté d'un papier
  dont le « consensus » est le maximin démocratique invite au malentendu. Le code est honnête en
  interne (`analysis.py:847`) ; le **contrat servi** ne l'est pas encore. [C]

- **⚠️ Redondance : géométrique traitée, sémantique non.** Le moteur B (`abstraction.py`) absorbe la
  redondance en ré-embeddant des **profils** canoniques → il rapproche les redondants. Mais
  `HIERARCHY_LAYERS.md:51-56` reconnaît que la redondance frère reste **sémantique, pas géométrique**,
  et **non mesurée**. Le papier fournit la métrique manquante (eq 5) et la montre discriminante
  (Diversity 0.00 vs Engagement 0.67, Table 2). [C p9] → notre angle mort a un instrument tout prêt.

- **✅ Couverture, plutôt bonne par construction.** Le coverage gap (eq 4) pénalise l'espace
  d'opinions non couvert. Agora, en partitionnant tout et en servant TOUS les thèmes (pas un top-k),
  a un CG structurellement faible au niveau thème. Le risque de CG se déplace **dans** le thème
  (représentants centrés — cf. fuite 1). [I]

**Bilan honnête** : Agora évite le pire écueil du papier (jeter des positions entières via un slate
trop court), mais **reproduit le biais de centralité/popularité un cran plus bas** (dans le thème,
dans la synthèse, via le poids social) — et **ne le mesure pas**. La lentille du papier transforme
un « on pense que c'est bon » en un test.

---

## 6. Bridging — pertinent pour nous ?

- Bridging (Community Notes, Polis) = sélectionner ce qui reçoit l'**approbation à travers des
  groupes qui d'ordinaire divergent** [C p2, p6]. Il **exige** votes + labels de groupe — Agora n'a
  ni l'un ni l'autre → **non transposable en l'état**. [C/I]
- Lien avec notre stance/clivage : notre travail identifie la **cible de clivage** et la position
  (`PIPELINE_LEDGER.md:41-46`) — c'est l'**opposé du bridging** : on surface l'axe de désaccord, pas
  le point de pont. Complémentaire, pas concurrent. Si un jour Agora récupère un signal de groupe
  (métadonnée démographique, ou clusters de stance comme proxy de groupes), le bridging deviendrait
  calculable : « quel claim est approuvé/proche à travers les deux pôles de stance ». **Piste
  spéculative**, pas une action. [I]

---

## 7. Où chacun est plus fort · limites de chacun

**Papier — forces** : garanties théoriques (choix social), protection explicite des minorités et de
la représentation proportionnelle, adapté au petit k digeste, conscient des groupes, métriques
d'évaluation propres et réutilisables. [C p2, p7]
**Papier — limites** : exige une **matrice d'approbation + labels de groupe** (collecte type
Remesh/Polis) ; ne traite pas le texte libre brut ; k doit rester petit ; vérification exacte
NP-difficile (approximations gloutonnes) ; ε data-spécifique [C Appendice H p23] ; **laisse la
production du rapport digeste hors scope** — donc réintroduit le problème minorité au moment de la
reformulation [C p3].

**Agora — forces** : fonctionne sur **texte libre sans aucun vote** (avantage pratique majeur : la
plupart des consultations n'ont pas de matrice d'approbation), souverain/local, hiérarchie navigable
multi-niveaux, traçabilité verbatim, passe à l'échelle (dizaines de milliers), pas besoin d'attributs
politiques. [C `.agent/README.md`]
**Agora — limites** : aucune garantie de représentation proportionnelle ; aucune métrique de
sous-représentation/minorité ; `consensus` surchargé ; représentants biaisés vers la centralité ;
poids social = biais de popularité ; synthèse LLM dans le régime à risque minorité ; redondance
sémantique frère non mesurée. (Toutes détaillées §5.)

---

## 8. Verdict — ce que cette étude change (ou pas) pour la feuille de route

**Ne change PAS le paradigme.** On n'a pas de votes, et clusteriser-tout-puis-synthétiser reste le
bon choix pour du texte libre sans matrice d'approbation. On n'adopte **pas** BJR/DiverseBJR comme
moteur de sélection au niveau racine. [I]

**Change ce qu'on devrait MESURER (3 emprunts concrets, par ordre de valeur)** :
1. **Audit démocratique par leurs métriques agnostiques-aux-groupes** (U_all, coverage gap, opinion
   redundancy) sur une matrice d'approbation **synthétisée par similarité** (recette §3.1, légitimée
   par leur propre prompt LLM p22). → premier chiffre sur « Agora écrase-t-il les minorités ». **À
   groomer comme tâche R&D** (harnais `research/`, verdict OUI/NON).
2. **Adopter `opinion redundancy` (eq 5)** comme la mesure manquante de la redondance frère-à-frère
   (frontière ouverte `HIERARCHY_LAYERS.md:51`). Instrument, pas action produit — cohérent avec la
   culture « mesurer avant d'adopter ».
3. **Tester DiverseBJR intra-thème** pour les représentants (§3.3) contre le medoïde actuel, panel
   aveugle. Seulement si (1) montre une fuite de minorité réelle.

**Corrections gratuites, hors R&D** :
- Clarifier le nom/contrat de `consensus` (§3.4) — c'est du serrage sémantique, pas du consensus
  démocratique.
- Documenter que le **poids social** peut = engagement (biais de popularité connu du papier) et
  décider explicitement s'il doit peser sur l'ordre d'affichage des thèmes.

**Ce que l'étude VALIDE chez nous** : la partition exhaustive nous protège du pire écueil (jeter des
positions) ; notre invariant verbatim rejoint le leur (opinions = texte exact) ; notre intuition que
la redondance est un axe à part entière est confirmée et outillée.

---

## 9. Ce que je N'AI PAS pu couvrir

- **Code des auteurs non exécuté** (github SalimHFX) : je n'ai pas vérifié l'implémentation réelle
  d'Algorithme 1/2 ni reproduit un chiffre ; tout vient du texte du PDF.
- **Faisabilité empirique de la matrice d'approbation synthétique** (§3.1) : l'idée est légitimée par
  le papier lui-même, mais le **choix de τ**, la validité de « cos ≥ τ ≈ approbation » et l'impact du
  recentrage ne sont **pas testés** — c'est précisément ce que la tâche R&D proposée doit trancher.
- **Chiffres Agora** : je n'ai lancé aucun build ni mesure (read-only). Les affirmations sur le
  comportement d'Agora sont tirées de la LECTURE du code (`file:ligne`), pas d'une exécution — un
  biais de représentant réel dépend du corpus et n'est pas quantifié ici.
- **`build.py` (26 k lignes cumulées cluster) et `hierarchy.py`** parcourus seulement via `analysis.py`
  et les notes ; je n'ai pas audité chaque chemin d'enrichissement LLM (opinion/arguments) ligne à
  ligne — la fuite « synthèse LLM » (§5) s'appuie sur le papier (Zhu 2025) + mémoire projet, pas sur
  une revue exhaustive du prompt d'opinion.
- **Métriques de groupe et bridging** : écartées faute de labels de groupe chez Agora ; non explorées
  au-delà du constat d'inapplicabilité.

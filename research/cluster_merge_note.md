# Sous-consolidation des clusters (fusion post-clustering) — VERDICT

**VERDICT : NON.** La « sous-consolidation » signalée sur tiktok (des macros au sens
proche qui restent séparés — ex. « Réseaux sociaux et addiction » vs « Boucle
d'addiction aux applications ») **n'existe pas au niveau du clustering** : mesurée par
la structure du graphe, **chaque paire suspecte est un vrai cluster distinct** que
Leiden a correctement coupé. Le remède demandé — une passe de FUSION sur la similarité
centroïde-centroïde — est soit **inutile** (garde sauce_magique → 0 fusion), soit
**destructeur** (sans garde → il fusionne des thèmes distincts). La cause réelle du
symptôme perçu par Bob est un **artefact de nommage** (les TITRES LLM tombent sur des
quasi-synonymes) amplifié par **l'anisotropie** de l'embedding sur ce corpus
mono-sujet. Aucune modification du pipeline. Harnais : `research/cluster_merge.py`
(mesure) + `research/cluster_merge_remedy.py` (prototype fusion, gardé sauce_magique).

Reproductible (zéro LLM, modèle du cache épinglé, embeddings nomic cachés) :

    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra faiss \
        python research/cluster_merge.py        --dataset tiktok --recut
    MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra faiss \
        python research/cluster_merge_remedy.py --dataset tiktok   # puis granddebat

---

## 1. Les paires suspectes existent bien… à la lecture des titres

Façade servie tiktok (17 macros post-recut). Les paires que Bob pointe sont réelles
**dans les titres LLM** :

| paire (titres servis) | sim centroïde | ratio graphe inter/intra |
|---|---:|---:|
| n0 « Réseaux sociaux et addiction » ⟷ n265 « Boucle d'addiction aux applications » | 0.876 | **0.014** |
| n161 « Perte de temps numérique » ⟷ n280 « Temps perdu à scroller » | 0.899 | **0.025** |
| n265 ⟷ n275 « Boucle algorithmique de contenus tristes » | 0.859 | **0.013** |
| n243 « Impact des écrans… » ⟷ n275 | 0.846 | **0.007** |
| n161 ⟷ n243 (paire centroïde la + haute du corpus) | 0.956 | **0.110** |

Distribution des 136 sims centroïde-centroïde inter-macros : médiane **0.85**, moyenne
0.81, max **0.956**, σ 0.12. **Tout est proche de tout** : le seuil de coarsening actuel
μ+σ = **0.93** est au-dessus de presque toute la masse (d'où `root_coarsen` qui ne
fusionne RIEN : n_fine 11 → n_macros 11). Ce n'est pas un réglage trop prudent : c'est
que **la similarité centroïde n'est pas un signal exploitable ici** (partie 2).

## 2. ANISOTROPIE — l'hypothèse de Bob est MESURÉE et CONFIRMÉE (mais mal nommée)

Bob : « l'embedding a atteint sa précision max sur un corpus mono-TikTok ». Mesurable, et
vrai — c'est de **l'anisotropie**, pas une perte de précision :

| mesure (tiktok, 2511 claims) | valeur | lecture |
|---|---:|---|
| cos moyen de paires **ALÉATOIRES** du corpus | **0.593** | un corpus bien étalé serait ~0.1–0.3 → espace **très comprimé** |
| cos intra-cluster (A, B) | 0.65 – 0.67 | |
| cos **INTER** clusters suspects (claim↔claim) | **0.635** | à peine au-dessus du hasard : **inter − baseline = +0.04** |
| **centroïde-centroïde** A·B | **0.956** | mais les centroïdes sont quasi colinéaires ! |

Le point décisif : deux clusters **thématiquement distincts** ont une sim
**centroïde 0.956** alors que leurs claims ne se ressemblent (0.635) presque pas plus que
deux claims tirés au hasard (0.593). **La sim centroïde ne mesure pas la proximité de
sujet** — elle mesure surtout la composante commune « tiktok-ité » partagée par tous les
vecteurs (cf. `tiktok_diagnostic.md` : PC1 corrélé −0.81 à « mentionne tiktok »).

**all-but-the-top** (retrait du vecteur moyen + re-norm) confirme et répare le contraste :

| espace | baseline aléa. | intra A/B | inter A-B | centroïde A·B | contraste intra−inter |
|---|---:|---:|---:|---:|---:|
| brut | 0.593 | 0.66 | 0.635 | **0.956** | 0.029 |
| ABTT d=1 | ~0 | 0.15 | 0.079 | **0.510** | **0.073** (×2.5) |
| ABTT d=2 | ~0 | 0.09 | 0.016 | **0.159** | 0.078 |
| ABTT d=3 | ~0 | 0.05 | −0.027 | **−0.418** | 0.083 |

Retirer la composante commune **dé-comprime** l'espace (centroïde 0.956 → 0.16) et
**double le contraste** inter-clusters. Donc l'anisotropie est réelle et substantielle —
mais elle **ne crée pas de doublons** : elle rend seulement la sim centroïde inutilisable
comme détecteur de fusion. (Même conclusion que `tiktok_diagnostic.md` exp. 3 : ABTT au
seuil de prod **effondre** le graphe → à ne faire qu'avec re-calibrage de seuil, hors
périmètre ici.)

## 3. GRAPHE — Leiden sépare par VRAIE structure, pas par accident de résolution

Densité d'arêtes kNN (graphe global de build, k=13, seuil=0.61) intra vs inter pour la
paire suspecte. Ratio inter/intra → 1 = pas de vraie coupure (accident) ; « 1 = vraie
structure.

- **Toutes** les top-12 paires macro tiktok : ratio **0.05 – 0.19** (moy ~0.09).
- Les paires **littérales de Bob** : ratio **0.007 – 0.025** — quasiment aucune arête ne
  traverse. Ex. n0⟷n265 : 87 arêtes inter pour des densités intra ~0.07 → ratio 0.014.
- granddebat (contrôle) : top-12 paires macro (« référendum/RIC » ⟷
  « proportionnelle », « transparence/dépenses » ⟷ « décisions/loi »…) ratio
  **0.06 – 0.14**.

**Verdict graphe : sur les deux corpus, toutes les paires « qui se ressemblent » par leur
centroïde sont en réalité des communautés bien séparées.** Il n'y a **pas** de
sur-découpage à réparer par fusion.

## 4. REMÈDE — fusion centroïde + gap, gardée sauce_magique : mesurée, NON

Prototype (`cluster_merge_remedy.py`) : seuil de fusion = **gap analysis sur la queue
haute** de la distribution des sims centroïde (dérivé, zéro constante corpus) ; fusion
gloutonne **gardée par sauce_magique** (accepte une fusion seulement si le score de la
façade **ne se dégrade pas**).

| dataset | façade | seuil gap | paires candidates | fusions **acceptées** | rejets |
|---|---|---:|---:|---:|---:|
| **tiktok** | 17 macros (N_eff 10.7 ≈ N_cible 10.8) | 0.943 | 4 | **0** | 4 |
| **granddebat** | 17 macros (N_eff 11.7 ≈ N_cible 11.8) | 0.946 | 7 | **0** | 7 |

- **Sans garde**, le seuil-gap sélectionnerait des fusions **aberrantes** : sur tiktok
  n34 « Haine et désinformation » ⟷ n184 « Comparaison des corps/influenceurs » (sim
  0.953) ; sur granddebat « référendum/RIC » ⟷ « proportionnelle/élections » (deux
  sous-thèmes officiels DISTINCTS des 14 OpinionWay). Le remède naïf **casse** le témoin.
- **Avec la garde sauce_magique**, chaque candidat est rejeté car il **dégrade** le score
  (tiktok 0.55 → 0.61–0.69 ; granddebat 0.51 → 0.56–0.69). Raison structurelle : sur une
  façade déjà équilibrée (N_eff ≈ N_cible), fusionner **baisse** N_eff sous la cible
  (terme β pire), **monte** top1 (δ pire) et **baisse** la cohésion (α pire) — les trois
  termes s'opposent. **Non-régression témoin : 0 fusion ⇒ aucun thème officiel fusionné**,
  trivialement respectée sur granddebat.
- La garde **n'est pas un no-op de construction** : sur une façade délibérément
  sur-fragmentée (les 201 feuilles, N_eff 166 ≫ cible 11), le mécanisme **accepte 3
  fusions**. Mais même là le couple centroïde+gap ne propose que 5 candidats — preuve que
  **la sim centroïde est un signal de candidat trop faible** dans un espace anisotrope.

**sauce_magique (fusion) et recut (best_cut) sont duaux** : la fusion n'aide que si la
façade est sur-fragmentée, or recut empêche déjà la sur-fragmentation. Sur tiktok, recut
a **SPLITé 11 → 17** macros (l'inverse d'une consolidation) et l'objectif l'a jugé
meilleur (score 0.67 → 0.55). Autrement dit **la fonction objectif du projet estime
elle-même que la façade tiktok n'est PAS sous-consolidée.**

## 5. Cause réelle du symptôme + reco

Ce que Bob voit vient à ~100 % du **nommage**, pas du clustering :

1. **Titres LLM tombant sur des quasi-synonymes.** Les macros distincts n0/n265/n243/n275
   reçoivent des titres qui répètent « addiction / temps / scroller / boucle ». Les
   *mots-clés* c-TF-IDF, eux, sont déjà distincts et corrects (instagram/application vs
   appli/désinstallé vs algorithme/triste — cf. `tiktok_diagnostic.md`, fix nommage livré)
   → **le problème résiduel est au niveau des TITRES LLM**, à traiter en nommage
   contrastif (titres conscients de leurs frères), lane naming — **hors périmètre fusion**.
2. **Anisotropie** (partie 2) : effet géométrique réel qui *plafonne* la qualité mais ne
   crée pas de doublons. Remède éventuel = all-but-the-top sur vecteurs cachés **avec
   re-calibrage du seuil**, à valider au banc qualité seulement si un gain est mesuré
   (risqué, cf. `tiktok_diagnostic.md` reco secondaire). **Pas** une passe de fusion.

**Reco :** ne PAS ajouter de passe de fusion centroïde (inutile si gardée, dangereuse
sinon). Router le symptôme vers le **nommage contrastif des titres**. Conserver la garde
sauce_magique comme invariant : elle prouve qu'aucune fusion n'améliore la façade servie.

## Limites / honnêteté

- **n = 1 corpus** pour le signal fort (tiktok, FR, mono-sujet) ; granddebat sert de
  contrôle multi-sujets. Rien ne se généralise au-delà sans mesure.
- Le **ratio graphe** dépend du seuil d'arête dérivé (k=13, seuil 0.61) ; il est
  cohérent avec l'ABTT (deux signaux indépendants convergent : distinct partout).
- Le cache granddebat dev est le **3k `ministral-3b`** (pas le 22k) ; le modèle est
  épinglé depuis le cache (fail-closed `extracted == 0`), zéro extraction déclenchée.
- La garde sauce_magique hérite des **poids v1 non calibrés** (cf.
  `sauce_magique_calibration.md`, verdict NON) — mais la conclusion « 0 fusion » tient
  pour tout jeu de poids raisonnable tant que la façade est proche de N_cible.

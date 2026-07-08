# Argument mining VERBATIM — VERDICT R&D (lane stance-argmining, 2026-07-07)

**Statut : RESEARCH, zéro modif pipeline/prod.** Corpus : `lutte-contre-les-fausses-informations`
(seul dataset avec `arguments.json` servi). Méthode VERDICT (mesurer avant d'adopter).

## Problème

L'argument mining servi (`backend/build_arguments.py`) fait « synthèse puis re-sourçage » : le
LLM **rédige** des arguments *reformulés* → le texte affiché **n'est PAS un span verbatim** de
témoignage. L'invariant Agora (« extraits exacts, zéro paraphrase ») est cassé sur le titre
d'argument ; le front l'admet (`ArgumentsPanel.tsx:11`). Baseline mesurée : **20 arguments,
0 % verbatim, 118/140 candidats droppés (84 %)**.

## Deux variantes re-conçues (texte servi = span verbatim, décision Bob « verbatim seul »)

Groupes = partition POUR/CONTRE de `claim_stance.json`, ≥2 claims. Harnais : `argmine_verbatim.py`.

| méthode | texte servi | LLM ? | args | **verbatim** | couverture |
|---|---|---|---:|---:|---:|
| CURRENT (servi) | phrase reformulée | 1/groupe (rédige) | 20 | **0 %** | — |
| **V-CLUSTER** | claim MÉDOÏDE | **0** (offline) | 24 | **100 %** | 100 % |
| **V-SELECT** | claim SÉLECTIONNÉ | 1/groupe (choisit, ne rédige pas) | 36 | **100 %** | 67 % |

`verbatim` = contrôle DUR (`Claim.is_verbatim` contre le texte d'avis) : 100 % par construction
pour les deux (le texte servi EST un claim ancré).

## Panel aveugle — CURRENT vs V-CLUSTER vs V-SELECT

`argmine_panel.py` : juge `mistral-large` T=0, X/Y/Z anonymisés (seedé), 3 passes (accents
neutre/fidélité/distinctivité), classement 1er→3e, critères = fidélité + représentativité.
**3 exécutions** (durcissement parser/tokens) → même direction :

| run | V-SELECT | CURRENT | V-CLUSTER |
|---|---:|---:|---:|
| rang moyen (bas=mieux) | **1.65–1.69** | 2.03–2.18 | 2.18–2.31 |

Tête-à-tête (run 34 bulletins) : **V-SELECT > CURRENT 24/34 (71 %)**, V-SELECT > V-CLUSTER 22/34
(65 %), CURRENT ≈ V-CLUSTER (18/34). *Caveat* : sur 15 thèmes en intersection, 3 gros thèmes font
sur-élaborer le JSON du juge → 10-12 thèmes effectifs / bulletin (à durcir ; ne renverse pas le
signe, robuste sur 3 runs).

## Verdict

- **ADOPTER V-SELECT** (le LLM SÉLECTIONNE des claims, ne rédige jamais). On restaure l'invariant
  verbatim **ET** le panel aveugle le juge **meilleur que la paraphrase actuelle** (71 % tête-à-
  tête) : le verbatim n'est pas un sacrifice de qualité, c'est un gain. La couverture (67 %) est
  le point à surveiller (des claims non-rattachés à un exemplaire choisi) — relever `MAX_K` ou
  compléter par V-CLUSTER pour les claims orphelins.
- **V-CLUSTER = repli offline** (zéro clé, zéro coût) : invariant garanti mais médoïdes trop
  **génériques** (le plus central = le plus fade, ex. « Revoir la charte de déontologie »). À
  n'utiliser que sans budget LLM.
- **NE PAS gérer le texte servi par génération** (statu quo cassé) : garder le champ `argument`
  = verbatim. Une passe de résumé IA pourra venir plus tard en champ SÉPARÉ étiqueté (UX), pas en
  remplacement — décision Bob « verbatim d'abord pour un vrai feedback ».

## Intégration (design, hors périmètre proto — nécessite GO impl)

Réécrire `backend/build_arguments.py` étape 1 : remplacer `synthesize_group` (LLM rédige) par
`vselect` (LLM choisit des indices) ; le reste (`back_match` exclusif, rollup parents,
`arguments.json`) est réutilisable. Ajouter `verbatim:true` + garder `sources`. Contrôle CI :
100 % des `argument` servis = sous-chaînes d'avis (le gate `Claim.is_verbatim` existe déjà).

## Artefacts

`argmine_verbatim.py` (mineur) · `argmine_verbatim_results.json` · `argmine_panel.py` (panel) ·
`argmine_panel_votes.json`.

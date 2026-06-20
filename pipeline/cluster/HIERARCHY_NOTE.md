# NOTE — Thèmes hiérarchiques (macro → sous-thèmes) sur le réel

> Pour l'architecte. Arbre `macro (level=0) → sous-thèmes (level=1)` obtenu sur la
> **consultation TikTok AN** (FR), via Leiden 2 niveaux. Naming **TF-IDF inchangé**
> (décision Bob), appliqué aux deux niveaux.

## Comment reproduire

```bash
uv run python -m pipeline.cluster.build \
  --input data/processed/ideas.jsonl \
  --source tiktok --lang fr --min-chars 12 --dedup 0.95 \
  --k 12 --threshold 0.84 \
  --hierarchical \
  --out frontend/public/graph.json
```

Défauts du mode hiérarchique (calés sur ce corpus) :
`--resolution-macro 1.0` (basse → grandes communautés), `--resolution-sub 3.0`
(plus fine, par sous-graphe induit), `--min-sub-size 15` (fusion des miettes vers
le sous-thème le plus **proche** en cosine). `seed=42`, reproductible.

## Procédé

1. **Niveau 0 (macro)** — Leiden basse résolution sur le graphe k-NN complet
   → **7 grandes communautés** (modularité macro **0.569**).
2. **Niveau 1 (sous-thèmes)** — pour chaque macro, on extrait le **sous-graphe
   induit** (arêtes internes au macro) et on relance Leiden plus fin. Les
   sous-clusters < `min_sub_size` sont fusionnés dans le sous-thème viable le plus
   proche sémantiquement → **24 sous-thèmes** au total (pas de poussière).
3. **Naming** — TF-IDF : macro = TF-IDF inter-macros ; sous-thème = TF-IDF
   **contrasté dans son macro** (sous-thèmes entre eux).

Corpus : 18 985 lignes → subset `tiktok/fr/min-chars≥12` = 1 604 → dédup 0.95 =
**1 514 avis**, 14 227 arêtes (avg degré 18.8).

## Arbre obtenu (1 514 avis)

- **[0] tiktok · tik · tok** — n=303, w=350, 9 sous-thèmes
  - [8] vais · aime · passe temps (n=49) — usage/temps passé
  - [7] école · fille · collège (n=45) — exposition des collégiens
  - [11] tiktok · tiktok étant · petite (n=38, w=77)
  - [13] défis · sujet · influenceurs (n=38) — défis / influenceurs
  - [9] rapport · difficulté · vis (n=35)
  - [10] dépression · adolescents · informations (n=34) — santé mentale ados
  - [12] dérrière · insulter rabaisser · arrivais (n=23) — rabaissement
  - [15] exemple quand · pris · regarde (n=21)
  - [14] cause algorithme · algorithme · mentale (n=20) — algorithme & mental
- **[1] fille · gênants · vidéos** — n=250, w=254, 4 sous-thèmes
  - [16] menaces · collège · fille (n=105) — menaces au collège
  - [17] mal · triste · algorithme (n=75)
  - [19] peuvent · ressentir · vidéos (n=36)
  - [18] toujours · venant · réel (n=34)
- **[2] tiktok · tok · difficile** — n=240, w=247, 4 sous-thèmes
  - [21] tiktok · propagande · dire (n=101) — désinformation/propagande
  - [22] reels · tiktok · instagram (n=51) — comparaison plateformes
  - [20] vient · mal · tiktok (n=47)
  - [23] suis · tiktok · temps (n=41)
- **[3] vidéos · mal · dégradants** — n=205, w=209, 3 sous-thèmes
  - [24] violence · propos · personnes (n=96) — violence/propos
  - [25] culpabilité · vide · passe (n=68)
  - [26] parfait · corps · corps parfait (n=41) — **image corporelle**
- **[4] culpabilité après · temps · négativité** — n=204, w=211, 2 sous-thèmes
  - [27] autres · temps · même (n=129)
  - [28] temps · passé temps · perte (n=75) — **perte de temps**
- **[5] parental · contrôle parental · tiktok** — n=172, w=173, 1 sous-thème
  - [29] tiktok · réseaux · fille (n=172) — **contrôle parental** (macro cohérent, indivis)
- **[6] insulte · chiens · mal** — n=140, w=160, 1 sous-thème
  - [30] mal · contenu · sentiment (n=140) — **insultes / harcèlement** (indivis)

## Intégrité (vérifiée)

`pipeline.cluster.hierarchy.check_integrity(payload)` → **0 erreur** :
- `children` d'un macro = exactement les feuilles dont `parent_id` = ce macro ;
- chaque feuille pointe un macro valide ; ids macro/feuille **disjoints** ;
- chaque nœud → feuille existante (`cluster_id`) + `macro_id` concordant ;
- couleur du nœud = couleur du **macro** (l'essaim se lit par macro, finesse au
  drill-down) ; toutes les feuilles peuplées.

## Lecture / arbitrage pour l'architecte

- **Macros 1, 3, 4, 5, 6** ont des labels nets (contenu gênant pour enfants ·
  contenu dégradant/violence · temps & culpabilité · contrôle parental ·
  harcèlement). Les sous-thèmes ressortent bien : *image corporelle* (26),
  *perte de temps* (28), *propagande/désinfo* (21), *santé mentale ados* (10).
- **Limite connue — labels macro génériques.** Macros 0 et 2 s'intitulent
  « tiktok · tok » : sur un corpus **mono-sujet** (tout parle de TikTok), le terme
  ubiquitaire « tiktok » n'est pas filtré par l'IDF quand il n'y a que 7
  macro-documents. Le naming **reste TF-IDF inchangé** (décision Bob) ; le niveau
  sous-thème, lui, contraste DANS le macro (plus de documents) et donne de bien
  meilleurs labels. Pistes éventuelles (à l'architecte de trancher) : stopwords de
  corpus (« tiktok/réseau/application »), ou titrage LLM ultérieur.
- **Granularité réglable** : `--resolution-macro` (nb de macros),
  `--resolution-sub` + `--min-sub-size` (finesse/miettes des sous-thèmes). Banc
  `eval` pourra arbitrer (cohérence NPMI/c_v, modularité, mixité-langue).

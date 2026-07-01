# NOTE — Modèle gagnant câblé : `nomic-v2` + arbre macro→sous sur le réel

> Pour l'architecte. Le `frontend/public/graph.json` de la démo est désormais
> régénéré avec le **modèle de prod retenu au banc qualité** (`research/quality_report.md`) :
> **`nomic-ai/nomic-embed-text-v2-moe`** (`nomic-v2`). À comparer avec l'ancien
> arbre e5-small documenté dans [`HIERARCHY_NOTE.md`](./HIERARCHY_NOTE.md).

## Modèle & paramètres

- **Embedding** : `nomic-ai/nomic-embed-text-v2-moe` (dim 768, préfixes
  `search_document:` / `search_query:`, `trust_remote_code`, extra
  `uv sync --extra embed-contender`). Le défaut global du registre **reste
  e5-small** (les autres consommateurs restent légers) ; nomic est sélectionné
  par `--model nomic-v2`.
- **Re-tuning vs e5 — indispensable.** nomic-v2 sépare mieux : ses cosinus k-NN
  sont **plus bas** (médiane du 1ᵉʳ voisin ≈ **0.806**, contre des valeurs qui
  saturaient sous e5). Le seuil e5 `--threshold 0.84` **vidait** le graphe sous
  nomic (avg-degré 0.97, 1250 macros singleton — inutilisable). On a re-calé le
  seuil sur la distribution réelle de nomic.

Commande reproductible :

```bash
uv sync --extra embed-contender
uv run python -m pipeline.cluster.build \
  --input data/processed/ideas.jsonl \
  --source tiktok --lang fr --min-chars 12 --dedup 0.95 \
  --hierarchical --model nomic-v2 \
  --k 12 --threshold 0.60 \
  --resolution-macro 1.0 --resolution-sub 1.5 --min-sub-size 18 \
  --out frontend/public/graph.json
```

Corpus : 1772 avis TikTok/FR → subset `min-chars≥12` = 1604 → dédup 0.95 =
**1597 avis**, 15 581 arêtes k-NN (avg degré **19.5**). `seed=42`, reproductible.

## Arbre obtenu — **8 macros → 47 sous-thèmes** (modularité macro **0.601**)

`check_integrity(payload)` → **0 erreur** (children ↔ parent_id cohérents, ids
macro/feuille disjoints, chaque nœud → feuille + macro concordants, couleur =
couleur du macro).

- **[0] baisse · perdu temps · culpabilité après** — n=292 (10 sous-thèmes)
  - estime soi · **dépression** (56) · perte temps · **procrastination** (36)
  - sentiment **dépendance** · téléphone (35) · scroller · scroll (28)
  - **infériorité · solitude** (23) · perte notion du temps (21)
  - sensation d'inutilité (21) · **culpabilité** après usage (19)
- **[2] tiktok · tok · tik** — n=269 (8 sous-thèmes)
  - **dopamine** · réseaux sociaux (47) · jeunes · **algorithme** (46)
  - arrêter de regarder (39) · addictif · perte de temps (18…)
- **[1] faux compte · collège · menaces** — n=271 (7 sous-thèmes)
  - faux compte · **live · photos** (58) · **moqueries** (43) · **harcèlement** scolaire (43)
  - **commentaires méchants** (40) · **homophobie · haine** (32) · insultes (27) · **suicide** (28)
- **[3] rapeur · voile · video** — n=240 (9 sous-thèmes)
  - **vidéos choquantes** (41) · **contenus haineux · influenceurs** (24)
  - **images choquantes** (21) · **maltraitance / contenu violent** (18)
- **[4] interdit · tik · tok** — n=225 (5 sous-thèmes)
  - exposition des **collégiennes** (63) · comparaison YouTube (46)
  - contrôle parental / installation (45) · **anorexie · automutilation** (39)
  - contenu **malsain** vu (32)
- **[5] tiktok · rend triste · algorithme** — n=121 (4 sous-thèmes)
  - **propagande / désinformation** (39) · **addiction** (32) · **algorithme** & mal-être (31)
- **[6] application · appli · addiction** — n=101 (3 sous-thèmes)
  - heures passées (42) · **application malsaine** (31) · **addiction forte** (28)
- **[7] parfait · comparer · grosse** — n=78 (1 sous-thème, indivis)
  - **image corporelle / comparaison du corps** (78)

## Lecture / arbitrage — amélioration vs e5-small

- **Pourquoi nomic gagne (rappel du banc).** e5-small clusterise **par langue**
  (NMI cluster↔langue 0.81) — disqualifié pour l'usage multilingue. nomic-v2
  **mixe les langues par thème** (NMI langue 0.008, topic 0.41), composite 0.850.
  Sur ce corpus FR mono-langue le bénéfice multilingue n'est pas visible à l'œil,
  mais nomic **sépare plus finement les thèmes** : 47 sous-thèmes nets vs 24 sous
  e5, à modularité comparable (0.601 vs 0.569).
- **Sous-thèmes plus tranchés.** nomic isole proprement des intentions que e5
  noyait : *dépression*, *procrastination*, *infériorité/solitude*,
  *homophobie/haine*, *anorexie/automutilation*, *propagande/désinfo*,
  *image corporelle*, *harcèlement scolaire*. Le drill-down par macro donne un
  arbre lisible (~6 sous-thèmes/macro).
- **Limite connue — inchangée : labels macro génériques.** Sur un corpus
  **mono-sujet** (tout parle de TikTok), le terme ubiquitaire « tiktok/tok/tik »
  n'est pas filtré par l'IDF avec seulement 8 macro-documents → macros [2] et [4]
  s'intitulent « tiktok · tok ». Le **naming reste TF-IDF** (décision Bob) ; le
  niveau sous-thème contraste DANS le macro et donne de bien meilleurs labels.
  Pistes (à trancher) : stopwords de corpus, ou titrage LLM ultérieur.
- **Granularité réglable** : `--threshold` (densité du graphe — **ré-étalonner
  par modèle**), `--resolution-macro` (nb de macros), `--resolution-sub` +
  `--min-sub-size` (finesse/miettes des sous-thèmes).

## Côté viz (`frontend/`)

Le panneau est désormais un **arbre macro → sous-thème → avis** : liste des
macros (triés par poids) → déplier un macro montre ses sous-thèmes → cliquer un
sous-thème liste ses **avis sources** (`node.props.text`). L'essaim 3D reste
**coloré par macro** (`node.color`) ; sélectionner un macro éclaire sa branche,
un sous-thème resserre l'emphase sur la feuille. Repli `graph.sample.json`
conservé. Le worker `addNodes` (Phase 2 live) n'est pas touché.

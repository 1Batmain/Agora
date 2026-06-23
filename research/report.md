# Banc d'arbitrage — Leiden vs UMAP+HDBSCAN (x-stance, eval-as-truth)

Vérité terrain : commentaires x-stance labellisés **FAVOR / AGAINST** par question politique. Pour chaque question on embed les commentaires (`intfloat/multilingual-e5-small`, CPU), on clusterise avec chaque approche, et on compare le clustering aux labels.

## Conditions

- **Échantillon** : 8 questions (sur 172 exploitables), langue `fr`.
- **Filtre question** : ≥ 40 commentaires, ≥ 5 par classe (FAVOR et AGAINST).
- **Embeddings** : 859 commentaires encodés, dim e5-small, seed `42`.
- **Leiden** : k-NN k=8, seuil cosine=0.84, résolution=1.0.
- **HDBSCAN** : UMAP(n_neighbors=15, n_components=5) + HDBSCAN(min_cluster_size=5).
- **Reproductible** : seed `42` (Leiden, HDBSCAN, échantillonnage, bootstrap). Python 3.11.15.

## Scorecard (moyenne ± écart-type sur les questions)

| Métrique | Leiden | HDBSCAN |
|---|---|---|
| NMI ↑ (vs labels) | 0.044 ± 0.028 | 0.057 ± 0.028 |
| ARI ↑ (vs labels) | 0.010 ± 0.015 | -0.001 ± 0.053 |
| Pureté ↑ | 0.711 ± 0.089 | 0.730 ± 0.082 |
| Silhouette ↑ (interne) | 0.101 ± 0.025 | 0.124 ± 0.024 |
| Stabilité ↑ (ARI bootstrap) | 0.545 ± 0.061 | 0.525 ± 0.234 |
| Nb clusters | 5.4 ± 0.5 | 3.2 ± 1.6 |
| Nb bruit (-1) | 0.0 ± 0.0 | 12.8 ± 9.5 |
| Latence clustering (s) | 0.033 ± 0.018 | 1.500 ± 3.392 |

> ↑ = plus haut est meilleur. NMI/ARI/pureté mesurent l'accord avec la vérité FAVOR/AGAINST (2 classes) ; ARI=0 ≈ hasard. La **silhouette** est interne (séparation dans l'espace d'embedding), indépendante des labels. La **pureté** monte mécaniquement avec le nb de clusters — à lire avec la ligne « Nb clusters ».

## Coût

- Embeddings : **859** vecteurs en **34.124 s** (partagés par les deux approches).
- Leiden : clustering total **0.267 s** sur 8 questions.
- HDBSCAN : clustering total **12.004 s** sur 8 questions.
- Stabilité : 5 ré-échantillons (fraction 0.8) par question et par approche.
- Wall-clock total : **54.402 s**.

## Détail par question

| qid | N | FAV/AGN | Leiden NMI | Leiden ARI | Leiden #cl | HDBSCAN NMI | HDBSCAN ARI | HDBSCAN #cl |
|---|---|---|---|---|---|---|---|---|
| 39 | 89 | 61/28 | 0.05 | 0.02 | 5 | 0.03 | -0.01 | 5 |
| 1443 | 85 | 52/33 | 0.01 | -0.01 | 5 | 0.05 | 0.05 | 2 |
| 3450 | 87 | 76/11 | 0.02 | 0.00 | 5 | 0.03 | -0.04 | 2 |
| 3391 | 149 | 95/54 | 0.05 | 0.03 | 6 | 0.05 | -0.04 | 2 |
| 3427 | 168 | 118/50 | 0.07 | 0.01 | 6 | 0.11 | 0.04 | 5 |
| 1434 | 98 | 62/36 | 0.01 | 0.00 | 5 | 0.04 | 0.02 | 6 |
| 3445 | 112 | 94/18 | 0.07 | -0.00 | 6 | 0.05 | -0.10 | 2 |
| 6 | 71 | 32/39 | 0.09 | 0.04 | 5 | 0.10 | 0.08 | 2 |

Questions (libellés) :
- `39` — Aujourd'hui, 1% des paiements directs de l'agriculture vont à la production biologique.  Cette part doit-el…
- `1443` — L’État devrait-il davantage s’engager pour une égalité des chances en matière de formation (p. ex. avec des…
- `3450` — La Confédération devrait-elle soutenir davantage les énergies renouvelables?
- `3391` — La Confédération devrait-elle soutenir davantage les étrangères et étrangers dans leur intégration?
- `3427` — Seriez-vous favorable à ce que le droit de vote et d'élection soit introduit au niveau communal pour les pe…
- `1434` — Pensez-vous qu’il soit justifié que la Confédération soutienne financièrement la garde extra-familiale des …
- `3445` — La Confédération devrait-elle davantage soutenir l'offre des services publics (p. ex. transports publics, p…
- `6` — L'assurance invalidité n'attribue plus de rentes de l'AI en cas de troubles douloureux non décelables de ma…

## Honnêteté (Playbook §5) — ce qui n'est PAS couvert

- **Échelle** : 8 questions, 859 commentaires. Échantillon modeste, à élargir (`--sample-questions`) pour resserrer les écarts-types.
- **Vérité terrain à 2 classes** : x-stance n'a que FAVOR/AGAINST. NMI/ARI pénalisent un clustering qui trouve > 2 groupes même sémantiquement valides (sous-thèmes d'argumentation). La silhouette nuance ce biais.
- **Domaine** : x-stance = votations suisses (FR). Transfert vers la consultation TikTok (témoignages libres, pas de labels) NON validé ici — c'est précisément pourquoi on n'a pas de vérité terrain sur TikTok.
- **Params figés** : un seul jeu de paramètres par approche (défauts pipeline). Pas de sweep d'hyperparamètres ici.
- **Bruit HDBSCAN** compté comme un cluster pour NMI/ARI/pureté (honnête) et exclu de la silhouette.

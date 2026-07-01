# REALDATA — Consultation TikTok réelle dans l'essaim

Branchement de la **vraie consultation citoyenne TikTok** (open data Assemblée
nationale) dans l'essaim 3D, à la place du fixture de démo (36 avis synthétiques).

## Corpus

| Étape | Avis | Détail |
|-------|------|--------|
| `ideas.jsonl` régénéré | 18 985 | x-stance (17 213) + TikTok (1 772) |
| subset `source=tiktok` | 1 772 | toutes langues |
| `lang=fr` | 1 621 | réponses libres FR à la question ouverte (mal-être / harcèlement) |
| `min_chars ≥ 12` | 1 604 | retire les non-réponses (« Néant », « Déprime »…) |
| **après dédup** `cosine>0.95` | **1 514** | −90 near-dups fusionnés (poids cumulé sur le représentant) |

- **Nœuds finaux : 1 514** (dans la cible 1 500–1 772). Aucune voix perdue : un
  near-dup absorbé devient du `weight` sur son représentant (poids max observé = 37,
  somme des poids = 1 604 = total avant dédup).
- **Liens k-NN : 14 227** (degré moyen 18,8). Rendu fluide à cette densité (mesh
  instancié + lignes en un seul `LineSegments`) → pas de plafonnement nécessaire ;
  le knob `--max-links` existe en garde-fou (garde tous les nœuds).

## Clustering — params retenus

```
embeddings   intfloat/multilingual-e5-small (dim 384, CPU, déterministe)
dédup        cosine > 0.95  (garde 1 représentant, cumule weight)
k-NN         k=12, seuil cosine 0.84  (backend sklearn)
Leiden       resolution=2.0, seed=42  →  15 communautés, modularité 0.53
naming       TF-IDF inter-clusters (uni+bigrammes, stopwords FR), pas de LLM
```

Reproductible : `seed=42` partout, embeddings CPU déterministes.

### Commande de régénération

```bash
uv run python -m pipeline.cluster.build \
  --source tiktok --lang fr --min-chars 12 --dedup 0.95 \
  --k 12 --threshold 0.84 --resolution 2.0 \
  --out frontend/public/graph.json
```

### Pourquoi ces réglages (tuning)

Le fixture était calé sur 36 avis (k=8, res=1.5 → 6 thèmes). Sur ~1,5 k avis réels
bruités, j'ai balayé la résolution Leiden (embeddings calculés une fois) :

| résolution | thèmes | modularité | crumbs (<5) |
|-----------:|-------:|-----------:|------------:|
| 1.0 | 7 | 0.57 | 0 |
| 1.5 | 11 | 0.55 | 0 |
| **2.0** | **15** | **0.53** | **0** |
| 2.5 | 20 | 0.51 | 0 |
| 3.0 | 26 | 0.49 | 0 |

`res=2.0` retenu : 15 thèmes **distincts et nommables**, aucune miette, sans
sur-fragmenter (res≥2.5 commence à dédoubler « culpabilité », « contrôle parental »).

## Les 15 thèmes (triés par intérêt = poids × qualité)

| # | label TF-IDF | n | consensus | diversity | lecture |
|--:|--------------|--:|----------:|----------:|---------|
| 2 | insulte · moquerie · rumeurs | 139 | 0.87 | 0.997 | **harcèlement scolaire / cyberharcèlement** |
| 0 | culpabilité après · temps application · perdu temps | 151 | 0.88 | 0.995 | **culpabilité & temps perdu (addiction)** |
| 1 | troubles alimentaires · perd temps | 142 | 0.88 | 0.993 | **troubles alimentaires (TCA)** |
| 3 | tiktok · algorythme · vivent | 127 | 0.89 | 0.976 | rôle de l'**algorithme** sur le moral |
| 4 | parental · contrôle parental · protéger enfants | 126 | 0.88 | 0.994 | **contrôle parental / protéger les enfants** |
| 5 | gênants · morts · mal vidéos | 116 | 0.88 | 0.997 | **contenus choquants** (images dégradantes, morts) |
| 6 | tiktok · tik · tok | 113 | 0.89 | 0.986 | usage TikTok général (familles) |
| 11 | tiktok · tca tiktok · skinny | 71 | 0.89 | 0.984 | **« skinny tok » / maigreur** |
| 7 | lire · instagram · dopamine | 108 | 0.90 | 0.959 | **dopamine / perte de concentration** |
| 8 | fille · accusation · comparaison | 101 | 0.87 | 0.997 | **comparaison physique / filles** |
| 9 | alerte · diffusant · contenus | 94 | 0.88 | 0.998 | signalement de **contenus à modérer** |
| 10 | site · live · thème | 73 | 0.87 | 0.991 | lives / découverte de contenus |
| 12 | homophobie · transphobie · racisme | 62 | 0.87 | 0.998 | **haine (LGBT-phobie, racisme)** |
| 13 | estime soi · déprime · fatigue | 48 | 0.89 | 0.982 | **estime de soi / déprime** |
| 14 | rapeur · américain · collège | 43 | 0.88 | 0.991 | témoignages ciblés (collège) |

> `consensus` haut (~0.88) + `diversity` haut (~0.99) sur l'ensemble ⇒ « même
> intention, formulations variées » — les thèmes sont cohérents sans être de
> simples copies. Labels lisibles ; quelques-uns restent génériques (#6, #10)
> car « tiktok » sature le corpus (toute la consultation parle de TikTok).

## Câblage viz

- `frontend/public/graph.json` = artefact réel (committé).
- `App.tsx` charge **`/graph.json` d'abord**, repli sur `/graph.sample.json` si
  absent (clone neuf avant pipeline). Le worker `addNodes`/le rendu Phase 2 ne
  sont pas touchés.
- Servi sur **:5180** : `curl -H 'Host: forge' http://localhost:5180/graph.json`
  → HTTP 200, `source: tiktok`, 1 514 nœuds, 15 thèmes.

## Contender (éval)

`--with-hdbscan` tracé dans `meta.clustering.hdbscan_contender` ; l'extra
`contender` (umap+hdbscan) n'étant pas installé ici, il est marqué
`{"available": false}`. L'arbitrage Leiden vs HDBSCAN reste du ressort de la lane
**eval** (sur x-stance), pas de ce branchement.

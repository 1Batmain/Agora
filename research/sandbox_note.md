# Bac à sable « console de mixage » — note de mesure (LANE 2 back)

Recluster **live, sans LLM** sur les embeddings cachés (claims + cibles). Le serveur
expose `POST /sandbox` (recluster paramétré) et `GET /explain` (decision-trace). Toute
la chaîne est rejouée à chaque mouvement de fader :

```
blend(α) → k-NN(k) → Leiden(resolution) → subdivision variance-adaptative (τ × tau_mult)
         → coarsening des racines (seuil μ+σ × coarsen_mult) → labels c-TF-IDF
```

Mesures sur **tiktok** (1604 avis, **3027 claims**, couverture cible **87.7 %** =
2655 claims ciblés ; nomic-v2, cache backend).

## 0. Embedding des cibles (knob α)
Au 1ᵉʳ accès, le backend embedde AUSSI les **cibles verbatim** (`target_emb.npz`, aligné
aux claims ; cible absente → vecteur nul + masque False). Le knob α mélange ensuite, de
façon **vectorisée**, `normalize(α·emb(cible) + (1−α)·emb(claim))` si cible, sinon le
claim seul (repli gracieux). Aucun ré-embed à la requête.

- α = 0 → clustering **par claim** (= structure servie aujourd'hui).
- α ↑ → clustering **orienté aspect** (rapproche les claims qui parlent du même *sujet*).

## 1. Démo « addiction » — l'effet du fader α (la pièce maîtresse)

À α = 0, le matériau « addiction » est **fragmenté en deux macros top-level distincts** :

| id  | parent | n_claims | mots-clés |
|-----|--------|---------:|-----------|
| n17 | —      | 173      | application, appli, **addiction**, heures, désinstaller |
| n18 | —      | 72       | **addiction**, addictif, drogue, forte, addicte |

`GET /explain?pair=n17,n18` le **chiffre** : ils ne fusionnent pas.
```
cos(centroïdes)=0.8652 ≤ seuil μ+σ×mult=0.9278  ET  min(cohésions)=0.8392
→ PAS de fusion (thèmes distincts).  same_macro=false
```

À **α = 0.5**, le blend cible tire les centroïdes l'un vers l'autre : un **seul macro
n17** (570 claims) **absorbe** l'addiction-substance et la dépendance psychologique :

| id  | parent | n_claims | mots-clés |
|-----|--------|---------:|-----------|
| n17 | —      | 570      | **addiction**, sentiment, sensation, dépression, **dépendance** |
| n18 | n17    | 293      | **addiction**, addictif, arrêter, drogue |
| n19 | n17    | 277      | sentiment, dépression, **dépendance**, concentration |
| n20 | n19    | 40       | impact, cerveau, angoisse, agressivité |

`GET /explain?cluster=n17` (voisinage, seuil μ+σ = 0.9551 à cet α) :
```
n18  sim=0.9918  would_merge=true  same_macro=true
n19  sim=0.9905  would_merge=true  same_macro=true
n20  sim=0.9769  would_merge=true  same_macro=true
```
`GET /explain?pair=n18,n19` :
```
cos(centroïdes)=0.9648 > seuil μ+σ×mult=0.9551  ET  min(cohésions)=0.83
→ FUSION (les centroïdes se recoupent plus que les membres ne tiennent à leur centroïde).
```

**Conclusion** : on *voit* et on *chiffre* l'intuition du brief — « α rapproche les trois
addiction ». Les sims passent de **0.865** (claim, distinct) à **>0.99** (cible, fusionné).
*(Les ids `nX` sont locaux au recluster sandbox, re-numérotés à chaque appel ; ils ne
correspondent pas un-à-un aux ids de l'analyse servie — c'est le **phénomène** qui compte,
pas le numéro.)*

## 2. Balayage α complet (défauts k/resolution dérivés, mults = 1)

| α   | clusters | macros | τ effectif | ms |
|-----|---------:|-------:|-----------:|---:|
| 0.0 | 21       | 6      | 0.211      | ~2000 |
| 0.3 | 333      | 9      | 0.122      | ~3900 |
| 0.5 | 50       | 14     | 0.161      | ~1500 |
| 0.7 | 445      | 14     | 0.093      | ~4900 |

**Lecture honnête** : le blend cible **comprime la distribution des dispersions** → le
seuil τ (dérivé du plus grand gap) **baisse** → la subdivision variance-adaptative
s'emballe (333/445 nœuds à α=0.3/0.7). C'est exactement la sensibilité que la console
rend visible — et **réglable à la main** (hors-scope nuit = la stabiliser
automatiquement). Le fader **τ (×tau_mult)** la dompte :

| α=0.7 | tau_mult | clusters | ms |
|-------|---------:|---------:|---:|
|       | 1.0      | 445      | ~4900 |
|       | 1.5      | 144      | ~2400 |
|       | 2.0      | 27       | ~1300 |
|       | 3.0      | 18       | ~1300 |

## 3. Autres faders (observé)
- **coarsen_mult** (× seuil μ+σ de fusion des racines) : à α=0, 1.0→0.95 fait passer
  6→3 macros (seuil 0.928→0.881). **Attention** : trop bas (0.9) collapse tout en un
  super-groupe → garde-fou « sur-fusion → on s'abstient » → retour aux fins (13 macros).
  Fader nerveux ; α reste le levier propre pour le regroupement par aspect.
- **k** (voisins k-NN) et **resolution** (Leiden) : défauts dérivés (k=13, μ=0.786,
  seuil d'arête 0.616) ; exposés pour exploration.

## 4. Latence
Objectif ~1 s / acceptance < ~1.5 s. Plancher = **Leiden** (~0.7 s global) + derive
defaults (~0.12 s) + k-NN (~0.19 s, faiss). À défauts (α=0) : ~1.1–2.0 s selon charge
de la box (partagée). Les réglages qui font exploser le nombre de nœuds (τ bas) coûtent
plus cher (chaque subdivision rejoue derive+knn+Leiden local) → **régler τ** pour rester
réactif. Le `ms` est renvoyé dans chaque réponse `/sandbox` (le front débounce).

## 5. Décision-trace (contrat)
`/sandbox` renvoie `trace.pairs` (toutes paires de fins : `sim`/`threshold`/`cohesion_min`/
`merged`) et `trace.nodes` (`dispersion`/`tau`/`subdivided`). `/explain` rejoue ces
critères pour un nœud (k voisins + critères) ou une paire (sim vs seuil vs cohésions) à
partir du **dernier** recluster mémoïsé. « Verre, pas boîte noire. »

## Repro
```
uv run --extra embed-contender --extra faiss --with fastapi python -m backend.test_sandbox
```
(target_emb tiktok généré au 1ᵉʳ accès si absent ; ~60 s d'embed des 2655 cibles, puis caché.)

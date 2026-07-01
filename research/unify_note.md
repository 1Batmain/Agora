# UNIFICATION — tous les datasets sur le pipeline claims+cible v3

> Branche `work/unify`. But : **tiktok**, **granddebat** et **xstance** servent TOUS une
> analyse **claims + cible** cohérente (extraction v3 : claims multi-spans + cible verbatim
> orientée stance, batchée N avis/appel). Extraction `mistral-large-latest`, enrichissement
> (titres/accroches/descriptions/insights) `mistral-small-latest`. **Gate verbatim DUR**
> conservé (claims ET cibles = sous-chaînes exactes, validées PAR AVIS).

## Résultat par dataset — 100 % VERBATIM partout

Vérifié sur le `claims.json` **réellement servi** (lecture seule, zéro ré-extraction) via
`backend.scripts.verify_claims_cache` :

| Dataset | avis | avis avec claims | claims | claims/avis | **verbatim** | **couv. cible** | cibles verbatim | multi-span | avis-entier | thèmes / macros | build (extract+enrich) |
|---------|-----:|-----------------:|-------:|------------:|-------------:|----------------:|----------------:|-----------:|------------:|----------------:|-----------------------:|
| tiktok      | 1604 | 1604 (100 %) | 3041 | 1.90 | **100 %** | **88.2 %** (2681) | 100 % | 24 | 548 | 388 / 6  | ~65 min (3910 s) |
| xstance     | 3000 | 3000 (100 %) | 5045 | 1.68 | **100 %** | **86.9 %** (4385) | 100 % | 14 | 376 | 63 / 20  | ~74 min (4455 s) |
| granddebat  | 3000 | 3000 (100 %) | 6594 | 2.20 | **100 %** | **75.1 %** (4952) | 100 % | 40 | 889 | 25 / 16  | ~87 min (5215 s) |

- **Verbatim : 100 %** sur les 3 corpus (14 680 claims au total), claims ET cibles — gate
  dur PAR AVIS tenu de bout en bout. **0 erreur** (429/5xx) sur les trois extractions.
- **Couverture cible** honnête, dépend du registre du corpus (cible = objet de POSITION,
  jamais inventée — un avis narratif/propositionnel sans objet pointable n'en reçoit pas) :
  - **xstance 86.9 %** et **tiktok 88.2 %** : avis d'OPINION (pour/contre) → objet de stance
    presque toujours pointable.
  - **granddebat 75.1 %** : question OUVERTE et PROPOSITIONNELLE (« que faudrait-il faire
    pour renouer le lien citoyens/élus ? ») → beaucoup d'avis sont des propositions
    narratives sans objet de prise de position net (889 avis pris en entier par le repli).
    Couverture plus basse = **fidèle au corpus**, pas une régression du pipeline.
- Profil cohérent avec la mesure extract-v3 d'origine sur tiktok (3027 claims / 87.7 % avant
  → 3041 / 88.2 % ici ; écart = non-déterminisme LLM, non significatif).

## Ingestion granddebat (préalable — n'existait pas dans ce worktree)

- Source : Grand Débat National 2019, espace « Démocratie et citoyenneté » (open data
  data.gouv.fr, Licence Ouverte 2.0), colonne 13 (question ouverte). Téléchargée (92 Mo CSV)
  puis `backend.build_cache --dataset granddebat --cap 3000 --min-chars 12`.
- Subset : **28 384 → 3000** avis (min_chars ≥ 12, dédup exacte, cap 3000, seed 42).
- Langues détectées en aval : fr 2916 · und 66 · es 15 · pt 2 · it 1 (mono-FR ~97 %, pas
  de `balance` — corpus mono-langue). Embeddings `nomic-embed-text-v2-moe` (3000×768).

## Audit propreté — claims+cible est le SEUL chemin servi

- Serving = **lecture seule** d'un cache PRÉCALCULÉ (`build_analysis` → `analysis.json`,
  `avis.json`, `citations`, `insights`). `/analysis` `/avis` `/insights` `/citations` ne
  font AUCUN calcul lourd à la requête.
- `/avis/{id}` renvoie le **format claim-v2** : `{id, cluster_id, color, spans:[{start,end}],
  target:{start,end}|null, theme_title}` — vérifié end-to-end via le code de serving
  (`analysis_store.read_avis`) sur les 3 datasets (claims réels, spans + target présents).
- Exactement **3 caches** sous `backend/cache/` (tiktok, xstance, granddebat) → `/datasets`
  = 3. Aucun cache orphelin, aucun chemin de traitement « ancien pipeline » dans le serving.
- **Résidu signalé (NON touché — hors lane)** : `/datasets` expose encore `namings` +
  `default_naming` (méthodes de LABELLISATION de cluster, c-TF-IDF & co). Ce n'est pas un
  second pipeline de traitement, juste des libellés passifs côté front. Le PLAN DE NUIT dit
  « pas de KNOB naming » (LANE 2, console) ; retirer le champ relève du front/cluster, pas
  de cette lane.

## ⚠️ À l'attention de l'architecte (merge / app au réveil)

- Le serveur **:8010 actuellement up tourne depuis le repo principal**
  (`/home/bat/projects/Analyse-des-consultations-citoyennes`), PAS depuis ce worktree. Il
  sert donc les caches du repo principal (état ANCIEN : `/avis` granddebat y renvoie vide,
  xstance sans claims). Les caches `backend/cache/<ds>/` sont **gitignored** → un merge
  `work/unify → main` ne les transporte pas.
- Pour que l'app au réveil serve les 3 datasets en claims+cible v3, il faut **rejouer les
  builds côté main** (ou copier les caches `tiktok/ xstance/ granddebat/` du worktree vers
  le repo principal) puis **redémarrer le :8010**. Non fait ici : redémarrer/repointer le
  serveur partagé est hors lane (risque de perturber d'autres lanes de nuit).

## Observation (hors lane — pour la lane clustering / console)

- Granularité très hétérogène : **tiktok 388 thèmes / 6 macros** vs xstance 63/20 vs
  granddebat 25/16. La subdivision variance-adaptative sur-segmente nettement tiktok. C'est
  exactement ce que la **console de mixage (LANE 2)** est censée régler à la main (τ, k,
  résolution, coarsening). Signalé, non modifié (NE touche pas au clustering).

## Reproduire

```bash
# (granddebat) ingestion
uv run --extra contender --extra embed-contender --extra faiss \
  python -m backend.build_cache --dataset granddebat --cap 3000 --min-chars 12

# ré-extraction v3 + build (par dataset) — extraction large, enrichissement small, batch 8
env -u AGORA_OLLAMA_URL AGORA_CLAIMS_BACKEND=api AGORA_CLAIMS_BATCH=8 \
  uv run --extra contender --extra embed-contender --extra faiss \
  python -m backend.build_analysis --dataset <ds> --reextract \
  --model mistral-large-latest --enrich-model mistral-small-latest

# vérif acceptance (verbatim 100 % + couverture cible), lecture seule, exit≠0 si non-verbatim
uv run --extra contender --extra embed-contender --extra faiss \
  python -m backend.scripts.verify_claims_cache --dataset <ds>
```

> Note : `build_analysis` n'a pas de flag `--batch` ; le batching v3 (N avis/appel) est piloté
> par `AGORA_CLAIMS_BATCH` (défaut 8). `--reextract` efface `claims.json` ET l'analyse
> persistée → extraction LLM fraîche.

# Cible de stance a/b/c — IMPRESSION AVANT PANEL (verdict EN ATTENTE du juge)

**Statut : RESEARCH, zéro modif pipeline.** Le juge aveugle est lancé SÉPARÉMENT ; ce
qui suit est la lecture MÉCANIQUE (métriques + inspection), pas le verdict. Harnais :
`research/stance_target_ab.py`. Repro (racine du worktree) :

    MISTRAL_API_KEY=… AGORA_OPINION_MODEL=mistral-large-latest PYTHONPATH=. \
    uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
        python research/stance_target_ab.py --dataset tiktok --leaves 10

## Question

L'objet de clivage servi est quasi-identique d'un thème à l'autre (« réguler X »
partout, symptôme Bob). Peut-on le rendre DISTINCTIF sans piétiner le garde-fou
« central > saillant » du cleavage-v2 (`research/cleavage_v2_note.md`) ? Trois dérivations
par feuille, sur 10 feuilles tiktok variées (titres distincts, top par taille) :

- **(a) ACTUEL** — échantillon CENTRAL (cos↔centroïde) + prompt v2 de prod
  (`build_opinion.cleavage_system`) + mots-clés c-TF-IDF bruts. Fidèle à `build_opinion`.
- **(b) CENTRAL + mots-clés distinctifs en CONTEXTE** — même échantillon central, mais
  mots-clés cadrés « ce qui SÉPARE ce thème des voisins » + consigne de contraste. Change
  UNIQUEMENT le prompt.
- **(c) CLAIMS DISTINCTIFS** — même prompt que (a), échantillon-source sélectionné par
  distinctivité pure (densité de vocabulaire propre au cluster). Change UNIQUEMENT l'input.

## Résultats (10 feuilles, **mistral-large-latest**, 30 appels cleavage)

Modèle aligné sur la PROD (décision « mistral-large-latest par défaut pour TOUTES les
tâches LLM », `8235994` sur dev) — remplace le run exploratoire mistral-small. Sur le
modèle fort, les conclusions **bougent** : le levier (b) cesse d'être du bruit.

| métrique (μ) | (a) actuel | (b) mots-clés distinctifs | (c) claims distinctifs |
|---|---:|---:|---:|
| fit_titre `cos(cible,titre)` — métrique ADOPTÉE par v2 | **0.801** | 0.800 | 0.783 |
| fit_titre médian | 0.781 | **0.805** | 0.778 |
| fit_centroïde `cos(cible,centroïde)` — réf. REJETÉE (info) | 0.768 | 0.781 | 0.762 |
| **INTER-CIBLES `cos` moyen** (bas = distinctes, haut = quasi-identiques) | 0.759 | **0.720** | 0.732 |
| cibles changées vs (a) | — | 9/10 | 8/10 |

Rappel run mistral-small (pour mémoire) : INTER-CIBLES a/b/c = 0.718 / 0.706 / 0.724,
fit_titre = 0.839 / 0.818 / 0.829. Le modèle fort produit une baseline (a) **plus
générique** (INTER-CIBLES 0.718 → 0.759) mais répond **mieux** au cadrage de contraste.

## Impression (à confirmer/infirmer par le panel)

1. **(b) dé-duplique enfin — sur mistral-large, pas sur small.** L'INTER-CIBLES tombe de
   0.759 (a) à **0.720** (b) : Δ = −0.039, ~3× l'effet-bruit du run small (−0.012). Et il le
   fait **sans coût de représentativité** : fit_titre reste plat (0.801 → 0.800), médiane en
   HAUSSE (0.781 → 0.805). Cas d'école n283 « Perte de temps ET culpabilité » : (a) réduit à
   « limiter le temps passé sur les réseaux sociaux » (générique, fitT 0.76, perd les DEUX
   facettes du titre) ; (b) « limiter le temps passé sur TikTok pour réduire la culpabilité »
   récupère les deux → fitT **0.90**. Le cadrage « ce qui SÉPARE ce thème des voisins »
   pousse le modèle fort à re-mobiliser le vocabulaire propre du thème.

2. **(c) est le levier à DOUBLE TRANCHANT — le risque « saillant » de v2 se matérialise.**
   (c) baisse un peu l'INTER-CIBLES (0.732) mais a le fit_titre le PLUS BAS (0.783) et
   **décroche du titre sur 2 feuilles** : n29 « mal-être » (c) « réguler les algorithmes de
   contenu émotionnel » cible le mécanisme réel MAIS fit_titre s'effondre 0.850 → **0.668**
   (hors-sujet déclaré) ; idem n13 « Dépendance » (c) « limiter l'usage algorithmique »
   (0.843 → 0.797). C'est très exactement la dérive vers le SAILLANT que cleavage-v2 a
   écartée. Quand (c) gagne (« réguler le design addictif », n273), c'est plus informatif ;
   quand il perd, il sort du sujet. Verdict qualité = au juge aveugle.

3. **La généricité résiduelle est en AMONT, pas ici.** Les feuilles quasi-synonymes
   (« Addiction » / « Dépendance » / « Temps perdu à scroller » / « Temps excessif sur les
   applis » / « Culpabilité ») produisent des cibles quasi-synonymes en (a) — normal : des
   cibles honnêtes sur des thèmes presque identiques SONT presque identiques (cf.
   `cluster_merge_note.md`). (b) les écarte à la marge sans les rendre distinctes ; le vrai
   levier de dé-duplication reste le clustering / nommage contrastif en amont. Sanity :
   n160 « Harcèlement », thème réellement distinct → a=b=c identiques (« interdire les
   réseaux sociaux aux collégiens »), stable quelle que soit la dérivation.

## Dépendance / honnêteté

- **`backend.develop.select_distinctive_claims` n'existe dans AUCUNE branche** au moment du
  run (dev, origin/dev, feat/titles-ancres : 0). La lane titres-ancres n'a produit que le
  CONTRAT — un test rouge non-committé (`backend/tests/test_develop_distinctive.py`) qui fige
  l'API (`cluster_term_weights` + `select_distinctive_claims(texts, idf, k, anchor_terms=None)`,
  ordre par densité c-TF-IDF, départage par index). Le harnais l'importe en `try/except` :
  dès qu'il atterrit sur dev, il est préféré. En attendant, la variante (c) utilise un
  sélecteur distinctif **LOCAL** (c-TF-IDF de contraste réutilisant `develop.corpus_idf` +
  `naming._tokenizer`) — PAS une copie du helper backend, un sélecteur de recherche à
  réconcilier. **À RÉCONCILIER** : ré-exécuter (c) avec le sélecteur canonique une fois mergé
  (~10 appels cleavage), le reste du harnais est inchangé (l'API locale est déjà alignée sur
  le contrat : `k` + `anchor_terms`).
- **MISE À JOUR post-run** : le helper vient d'atterrir sur `feat/titles-ancres` (`fc07c31`,
  « titrage ANCRÉ sur le vocabulaire distinctif ») — PAS encore sur `dev`. La réconciliation de
  (c) est donc DÉBLOQUÉE : dès que `develop.py` arrive sur dev, ré-exécuter (c) avec le
  canonique (~10 appels, dans le budget d'un follow-up). Même logique c-TF-IDF de densité que
  le sélecteur local → (c) n'est pas attendu qualitativement différent, mais à VÉRIFIER.
- Le sélecteur distinctif local est **bruité** sur ce corpus mono-sujet : le vocabulaire
  « distinctif » remonte encore « tiktok/tik/tok/temps/sur » (pas de stopwords corpus). C'est
  en soi une donnée : la distinctivité par vocabulaire est faible dans un espace anisotrope.
- n=1 corpus (tiktok, FR, mono-sujet). Rien ne se généralise sans mesure sur granddebat.

## Artefacts

- `stance_target_ab_results.json` — rows a/b/c + fits + summary.
- `stance_target_ab_panel.jsonl` — 26 paires ANONYMISÉES (X/Y) pour panel aveugle (titre +
  claims de contexte NEUTRE, aucune étiquette de variante).
- `stance_target_ab_panel_key.json` — clé pair_id → {X,Y}=variante, HORS du fichier aveugle.

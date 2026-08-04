# PipelineProfile — une source de vérité pour « quels modèles, quelle forme »

**Idée (Bob, 2026-08-04)** : une classe qui déclare les modèles et les caractéristiques du
pipeline, instanciée en **profils** — un pour la synthèse FINALE, un pour la synthèse LIVE —
chacun avec les modules qui lui sont propres. Motivation : « plusieurs références au modèle,
on se perd ». **Statut : GROOMÉ, non lancé.**

---

## 0. Le précédent existe déjà dans le dépôt

`pipeline/embed/registry.py` est **littéralement le patron demandé**, appliqué aux embedders :
`@dataclass(frozen=True) ModelSpec` (id, préfixes, révision épinglée, loader, **licence**) +
`REGISTRY` + `ALIASES` + `resolve_model_id()`. Son docstring dit : *« Ajouter un contender =
ajouter une ModelSpec ici (aucun autre changement). »*

→ On ne conçoit rien de neuf : on **étend un patron validé** aux modèles LLM et aux paramètres
de forme. C'est le bon argument pour le faire.

## 1. Inventaire de la dispersion — 14 constantes, 7 fichiers

| Rôle | Constante | Fichier | Valeur effective | Env |
|---|---|---|---|---|
| extraction | `EXTRACT_MODEL` | `build_analysis.py:60` | `mistral-large-latest` | `AGORA_EXTRACT_MODEL` |
| enrichissement (titres/accroches/desc/insights) | `ENRICH_MODEL` | `build_analysis.py:67` | `mistral-large-latest` | `AGORA_ENRICH_MODEL` |
| abstraction (profils macro) | `ABSTRACTION_CHAT_MODEL` | `analysis.py:533` | `mistral-small-latest` | **aucune** |
| opinion / stance | `MODEL` | `build_opinion.py:53` | `mistral-large-latest` | `AGORA_OPINION_MODEL` → `AGORA_ENRICH_MODEL` |
| arguments (V-SELECT) | `MODEL` | `build_arguments.py:61` | `mistral-small-latest` | `AGORA_ARGMINE_MODEL` → `AGORA_ENRICH_MODEL` |
| traduction | `TRANSLATE_MODEL` | `translate.py:35` **et** `keywords_fr.py:39` | `mistral-small-latest` | `AGORA_TRANSLATE_MODEL` |
| nommage (client bas niveau) | `NAMING_MODEL` | `mistral_client.py:24` | `mistral-large-latest` | `AGORA_MISTRAL_MODEL` |
| synthèse (client bas niveau) | `SYNTHESIS_MODEL` | `mistral_client.py:27` | → `NAMING_MODEL` | `AGORA_MISTRAL_SYNTH_MODEL` |
| extraction (défaut backend) | `API_MODEL` | `claims/backend.py:27` | **`ministral-3b-latest`** | `AGORA_CLAIMS_API_MODEL` |
| extraction (défaut pipeline) | `DEFAULT_MODEL` | `claims/pipeline.py:41` | **`ministral-3:latest`** (Ollama) | — |
| LLM local | `MODEL` | `local_llm_client.py:43` | `google/gemma-4-12B-it` | `AGORA_LOCAL_LLM_MODEL` |
| embedder (pipeline) | `DEFAULT_EMBEDDER` | `claims/pipeline.py:47` | `arctic-l` | — |
| embedder (registre) | `DEFAULT_MODEL_ID` | `embed/embedder.py:41` | `Snowflake/arctic-l-v2.0` | — |
| embedder (servi) | `MODEL_ID` | `recluster.py:50` | **`nomic` codé en dur** | — |

### Les 6 pathologies concrètes (pas des impressions — des lignes)

1. **Trois défauts d'extraction contradictoires** : `mistral-large-latest`, `ministral-3b-latest`,
   `ministral-3:latest`. Le modèle obtenu dépend du **point d'entrée**, pas d'une décision.
   (`ClaimsCacheModelMismatch` existe précisément pour rattraper ça — c'est le symptôme, pas la cure.)
2. **`TRANSLATE_MODEL` dupliqué à l'identique** dans deux modules, lisant la même env.
3. **Cascades d'env à 2 niveaux avec replis DIVERGENTS** : `opinion` et `argmine` retombent tous
   deux sur `AGORA_ENRICH_MODEL`, mais leur littéral final diffère (`large` vs `small`). Poser
   `AGORA_ENRICH_MODEL` déplace donc **trois rôles d'un coup** — couplage non déclaré.
4. **`ABSTRACTION_CHAT_MODEL` en dur**, seul rôle non surchargeable.
5. **`recluster.MODEL_ID`** annonce nomic dans `/health` alors que le défaut est arctic (vrai
   par coïncidence aujourd'hui : les caches d'idées servis sont encore en nomic 768 d).
6. **Le client bas niveau porte une politique de modèle** (`NAMING_MODEL`, `SYNTHESIS_MODEL`) :
   c'est au RÔLE de décider, pas au transport HTTP.

### Fait dur, vérifié : **aucune** de ces 14 env vars n'est posée en prod

`systemctl --user cat agora-backend` ne pose que `AGORA_PUBLIC`, `AGORA_AUTOBUILD`,
`AGORA_CLAIMS_BACKEND` ; aucun script d'ops ne pose de `AGORA_*_MODEL`. **Toute cette
optionalité ne sert rien aujourd'hui** — on peut la simplifier sans casser d'ops.

## 2. Design proposé

`pipeline/profile.py` — un `@dataclass(frozen=True) PipelineProfile` avec **un champ par rôle**
(extraction, enrichissement, abstraction, opinion, argmine, traduction) **plus** la forme
(embedder, `fine_gamma`, `resolution`, `seed`, batch, workers) et un `name`.

Deux instances au départ : `FINAL` (qualité, ce qui tourne aujourd'hui) et `LIVE` (latence/coût).
Les modules lisent `profile.extract_model` ; les constantes locales disparaissent.

### Trois contraintes non négociables

**(a) Le profil DOIT entrer dans la signature de cache — c'est LE risque du chantier.**
Les caches sont clés par contenu **+ modèle** (`claims.json` porte `"model"` ; `titles.py:131`
hashe le modèle). Si `LIVE` et `FINAL` écrivent dans le même `backend/cache/<dataset>/`, la
synthèse live (petit modèle) **écrase silencieusement** les titres/insights de la synthèse
finale. Il faut un espace de cache par profil **ou** le nom du profil dans chaque clé.
> Précédent direct : la bascule nomic→arctic a laissé un seuil périmé en place et a rendu la
> corrélation des contributions inopérante pendant une semaine. Même classe d'erreur, en plus gros.
> **Un test de non-collision LIVE/FINAL est le critère d'acceptation n°1.**

**(b) Ne pas aplatir ce qui est légitimement hétérogène.** `abstraction` en `mistral-small` et
`argmine` en `mistral-small` ne sont pas des oublis : ce sont des choix commentés (« nommage/
regroupement léger ≠ extraction ») et, pour argmine, **validés au banc** (V-SELECT). Un profil
à un seul champ `model` détruirait des verdicts mesurés. → **un champ par rôle, toujours.**

**(c) Migration à iso-comportement.** Le refactor ne doit changer **aucun modèle effectif** :
`FINAL` reproduit exactement les valeurs de la colonne « valeur effective » ci-dessus. Sinon on
mélange un refactor et un changement de qualité, et plus rien n'est attribuable.
Witness : un build complet avant/après doit être **cache-HIT à 100 %** (zéro appel LLM).

## 3. Ce que ça débloque (le lien avec le live)

C'est le prérequis propre du dossier LIVE (`.agent/notes/LIVE_FEASIBILITY.md`) : « étage 1 =
LLM cheap, étage 2 = restructuration qualité » **est** la distinction `LIVE` / `FINAL`. Sans
profil, ces deux étages se partageraient les mêmes constantes globales et le même cache.

Ça rend aussi **tirable** le levier déjà mesuré et jamais promu : ministral-3b ≈ mistral-large
pour l'extraction (Δ +0,02 sur 191 avis appariés, ~150× moins cher,
`research/llm_regression/`). Aujourd'hui basculer `EXTRACT_MODEL` est un geste global et risqué ;
avec un profil `LIVE`, c'est un choix déclaré et isolé.

## 4. Tâche groomée

**But** : une source de vérité unique pour les modèles et la forme du pipeline, instanciable en
profils.
**Contraintes** : iso-comportement pour `FINAL` ; un champ par rôle ; le profil entre dans la
signature de cache ; zéro nom de corpus (règle de généricité inchangée).
**Acceptation** — résultats réels (branche `feat/pipeline-profile`) :

1. ✅ **Plus aucun littéral de modèle hors du profil.** Critère REFORMULÉ : je visais d'abord
   « aucune constante nommée `*MODEL*` ailleurs », ce qui portait sur les *noms*. Ce qui compte
   est que les **valeurs** ne soient plus dupliquées ; les modules gardent des alias
   **dérivés** (`EXTRACT_MODEL = profile.active().model_for("extract")`). Test :
   `test_aucun_litteral_de_modele_hors_du_profil`, avec trois exceptions justifiées (le profil
   lui-même, les modèles **Ollama** locaux, la **grille tarifaire** `cost.py`).
2. ⚠️ **« 100 % cache HIT » était un critère IMPOSSIBLE — le mien, pas celui du code.**
   `render_insight` est **explicitement sans cache** (`backend/insights.py`) : chaque rebuild
   régénère les synthèses. Un build ne peut donc jamais coûter zéro appel.
   **Ce qui a été prouvé à la place, et qui vaut mieux** — protocole à trois runs + contrôle :

   | Run | Code | Appels LLM |
   |---|---|--:|
   | 1 | refactor, cache tel quel | 136 |
   | 2 | refactor, cache chaud | **39** |
   | 3 | refactor, cache chaud | **39** (stable) |
   | **contrôle** | **`main` (sans refactor)**, cache chauffé par le refactor | **39** |

   Le contrôle est décisif : le code d'avant **retrouve intactes** les entrées écrites par le
   code d'après ⇒ les **clés de cache sont identiques** ⇒ iso-comportement démontré. Les 39
   appels résiduels sont le plancher **pré-existant** des insights non cachés (~55 s, ~0,12 $
   pour 19 thèmes ; ≈ 3× cela sur granddebat, 58 thèmes), indépendant de ce chantier.
3. ✅ `test_live_est_isole_de_final` + `test_tous_les_profils_non_defaut_sont_namespaces`
   (garde-fou pour les profils futurs, pas seulement pour `live`).
4. ✅ `/health` expose `profile` (rôles, embedder, forme) et le `model_id` **réel de chaque
   dataset**, lu dans son `meta.json` — un champ global mentait dès que deux datasets
   n'avaient pas été bâtis avec le même embedder.

**Un changement de comportement ASSUMÉ** : `pipeline/claims/backend.API_MODEL` passe de
`ministral-3b-latest` à la valeur du rôle `extract` (`mistral-large-latest`). C'était le
3ᵉ défaut d'extraction contradictoire ; un appelant qui omettait `model=` obtenait un modèle
différent de celui du pipeline. Le chemin servi n'est pas concerné (`build_analysis` passe
toujours `model=` explicitement). Effet de bord positif : un oubli ne provoque plus un
`ClaimsCacheModelMismatch` — le test correspondant provoque désormais la divergence
**explicitement**, ce qui teste le garde plutôt qu'un défaut incohérent.

**Dépendances** : aucune. Ne touche pas le chemin servi public (prod ne construit jamais).
**Note de contrat** : les env vars par rôle deviennent `AGORA_MODEL_<RÔLE>` (un seul niveau) et
`AGORA_PROFILE` sélectionne le profil. Aucune ancienne variable n'était posée en prod (vérifié).

**Reste ouvert** (hors périmètre, découvert ici) : les insights non cachés — ~39 appels par
rebuild sur le plus petit dataset. C'est aussi la couche que le pipeline live **diffère** ;
le sujet se traitera là.

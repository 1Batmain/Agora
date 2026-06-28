# PROTO — STANCE sur le SUJET DU CLUSTER vs cibles per-claim — VERDICT

**Branche** `work/stance-proto` · **R&D pur** (aucun fichier produit touché) ·
script `research/stance_proto.py` · chiffres bruts `research/stance_proto_results.json`.

Lancement :
```
MISTRAL_API_KEY=$(cat var/mistral.key) \
uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
python research/stance_proto.py
```
> Note worktree : ce worktree R&D n'embarque ni `var/` ni le cache `claims.json`. Le
> script lit la clé et `backend/cache/tiktok/claims.json` du **dépôt principal** en
> **lecture seule** (repli automatique). Aucun fichier produit n'est modifié.

## Hypothèse
La stance citoyenne s'**agrège proprement** si le **sujet** vient du **CLUSTER**
(canonique, un seul par thème) plutôt que des **cibles per-claim** (idiosyncrasiques,
en grande partie non agrégeables). Testé sur 2 macros TikTok à sujet clair, clusterisées
par le chemin de prod `build_live_tree('tiktok', k=défaut)` (10 macros, sélection des 2
par mots-clés — aucun contenu imposé).

## Prompt STANCE (mistral-small, batché, multilingue)
On donne **le sujet du cluster** + une contribution verbatim, et on classe la position
ENVERS le sujet :
- **favorable** : valorise / défend / minimise le sujet (le voit positivement, en veut plus) ;
- **defavorable** : dénonce / critique / veut limiter le sujet (le voit négativement) ;
- **nuance** : ambivalent, conditionnel, ou pas de position claire.
Sortie JSON stricte `{"results":[{"i","stance","justif"}]}`, justif ≤15 mots dans la
langue de la contribution. (Texte intégral du *system prompt* dans
`stance_proto_results.json → stance_prompt_system`.)

## Exemples (claim → stance | justif)

**Cluster ADDICTION** — sujet canonique (titre LLM) : **« Dépendance aux applications de vidéos courtes »**

| claim (extrait) | stance | justif |
|---|---|---|
| Sentiment de mal-être… regarder des vidéos courtes fait perdre productivité/temps | defavorable | Critique la dépendance et perte de temps |
| incapacité à s'extraire de son téléphone et à profiter de la vie réelle | defavorable | Incapacité à profiter de la vie réelle |
| Il est compliqué de se détacher des contenus qui tournent en boucle… | defavorable | Difficulté à se détacher |
| Dependance aux vidéos de courts formats et perte de concentration | defavorable | Dépendance et sentiment de manque |
| Quand je passe trop de temps sur l'application, j'ai le sentiment d'être vide | defavorable | Sentiment de vide et distraction |

**Cluster HARCÈLEMENT** — sujet canonique (titre LLM) : **« Harcèlement en ligne contre les filles »**

| claim (extrait) | stance | justif |
|---|---|---|
| contenus de haine envers certaines personnes et communautés | defavorable | Dénonce haine en ligne |
| ceux derrière leur téléphone utilisent des mots très blessants | defavorable | Critique mots blessants |
| Plus de liberté d'expression… j'ai reçu des messages « sale f… » | defavorable | Dénonce insultes politiques |
| Ma fille en mal-être… s'automutile, influencée en ligne | defavorable | Dénonce mal-être fille |
| Notre baby-sitter a publié une vidéo de notre fils à caractère raciste | defavorable | Dénonce vidéo raciste |

## Agrégats stance (par cluster, sur TOUS les membres)

| cluster | sujet | n | favorable | **defavorable** | nuance | dominante |
|---|---|--:|--:|--:|--:|--:|
| addiction | Dépendance aux applications de vidéos courtes | 160 | 0 | **157** | 3 | **98 %** |
| harcèlement | Harcèlement en ligne contre les filles | 305 | 4 | **276** | 25 | **90 %** |

→ Un **verdict net et lisible** par cluster : *« 98 % des contributions du thème
addiction dénoncent la dépendance »*, *« 90 % du thème harcèlement le condamnent »*.
Bruit attendu : les 4 « favorable » du harcèlement sont des erreurs LLM ponctuelles
(ex. un « doxxing » dénoncé mal étiqueté) — marginal (1,3 %), n'entame pas l'agrégat.

## Comparaison : les cibles per-claim des MÊMES avis (depuis `claims.json`)

| cluster | n claims | **cibles distinctes exploitables** | distinct / claims | cibles vs 1 sujet | inexploitables (déictiques/courtes) |
|---|--:|--:|--:|--:|--:|
| addiction | 301 | **217** | 0.72 | **217 : 1** | 15 % |
| harcèlement | 450 | **360** | 0.80 | **360 : 1** | 15 % |

**0.72–0.80 cible distincte par claim** : ~3 claims sur 4 introduisent une cible
**jamais réutilisée**. Échantillon brut de cibles d'un même cluster (harcèlement) :
> contenus de haine · mots très blessants · *cela* · liberté d'expression sur ses idées
> politiques · messages · doxxing · *(aucune)* · une vidéo de notre fils à caractère
> raciste · contenus · Désinformation · discours de haine et défis dangereux · ridiculiser
> des individus gratuitement…

Même les cibles « les plus fréquentes » d'un cluster ne convergent pas vers un sujet :
addiction = {`cette application`, `l'addiction`, `addiction`, `application`, `le mal être`,
`une dépendance psychologique chronique`} ; harcèlement = {`désinformation`, `contenus`,
`incitation à la haine`, `homophobie`, `mauvaises blagues pour brisé les couples`,
`des commentaires`}. C'est un **nuage d'objets d'argument**, pas un sujet agrégeable —
et 15 % sont en plus de purs déictiques/pronoms (`cela`, `ça`, `il`…) ou des mono-mots,
strictement inutilisables hors contexte de l'avis. *(Le 15 % minore la part « molle » :
beaucoup de cibles « exploitables » comme `messages`/`contenus` restent des fragments
génériques — la dispersion 217:1 / 360:1 est la mesure décisive.)*

## VERDICT

**OUI, nettement.** Le **sujet venu du cluster** rend la stance **agrégeable** :
2 sujets canoniques → 2 verdicts à dominante 90–98 %. Les **cibles per-claim** sur les
**mêmes** contributions **explosent** en 217 et 360 sujets distincts (≈ une cible neuve
tous les 1,3 claim) : **non agrégeables par construction** — on ne peut pas compter
« N pour / M contre » sur 217 objets différents.

| | sujet = CLUSTER | cible = per-claim |
|---|---|---|
| nombre de sujets / cluster | **1** | 217 et 360 |
| agrégat stance | net (98 % / 90 %) | impossible (1 sujet ≈ 1 claim) |
| stabilité hors-contexte | canonique | 15 % déictiques + fragments |

### Recommandation — **ACTER l'archi**
- **Sujet = titre du cluster** (canonique) ⇒ **axe de stance** servi à l'utilisateur :
  « sur *[sujet du thème]*, N défavorables / M favorables / K nuancés ».
- **Cible per-claim rétrogradée** : utile comme **signal de clustering** (pondération-cible
  α, cf. [[agora-sandbox-console]]) et comme **traçabilité** (ancrage verbatim
  [[agora-spans-anchor-textclean]]), **pas** comme sujet d'agrégation de stance.
- Étape suivante suggérée (hors proto) : un passe stance batché par macro au build, exposé
  comme métrique de thème ; mesurer la part d'erreurs LLM (ici ~1–10 % de « favorable »
  parasites) sur un échantillon annoté avant de la servir.

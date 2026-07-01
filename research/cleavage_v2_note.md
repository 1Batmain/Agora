# Objet de clivage v2 — conditionné sur le cluster + score de fit · verdict

**Branche** `work/cleavage-v2` · dataset `granddebat` (15 plus grosses feuilles) ·
modèle `mistral-small-latest` · seed 42 · encodeur `nomic-v2` (espace claims).
Repro : `research/cleavage_v2.py` (→ `cleavage_v2_results.json`) puis
`research/cleavage_v2_fit_probe.py` (sonde fit centroïde vs titre).

## Problème (Bob)
`derive_cleavage` (backend/build_opinion.py, v1) ne recevait PAS le titre du cluster et
demandait « la proposition la PLUS SAILLANTE ». Résultat : la cible dérive vers une
**facette bruyante** au lieu du centre du thème. Cas-test : thème
« Restaurer la confiance par l'écoute » → cible v1 **« cesser de mentir »** (une facette).

## v2 — 3 leviers testés
1. **Conditionner** sur `node.title` (titre court du thème, déjà caché par le build d'analyse).
2. **« central » > « saillant »** : résumer le débat du thème, pas le détail le plus bruyant.
3. **Score de fit** : embedder la proposition → cosinus vs **centroïde du cluster**, seuil ~0.3.

## VERDICT

| Levier | Verdict | Preuve |
|---|---|---|
| 1+2 Conditionner + « central » | **ADOPTÉ** | la cible se recentre sur le thème ; fit-titre meilleur **12/15**, sans régression de qualité |
| 3 fit vs **centroïde** | **REJETÉ** | discriminateur cassé : sur le cas Bob il classe la PIRE cible plus haut (v1 0.823 > v2 0.787) ; plage [0.76,0.91], seuil 0.3 jamais atteint |
| 3′ fit vs **TITRE** (remplaçant) | **ADOPTÉ** | `cos(cible, titre)` suit la centralité : v2 mieux 12/15, **correct sur Bob** (0.793 > 0.754) |

### Pourquoi le fit-centroïde est trompeur
Le centroïde est la moyenne du **sac de claims**, donc dominé par la facette la PLUS
VOCALE. `cos(cible, centroïde)` récompense donc exactement le biais « saillant » qu'on
cherche à fuir. Sur n0, la cible-facette v1 « cesser de mentir » est lexicalement plus
proche du centroïde (saturé de *mentir/honnêteté/discours*) que la cible centrale v2
« écoute active » → la métrique **inverse** le bon classement. Le bon signal de
représentativité est la proximité au **titre** (le sujet déclaré du thème), pas au centroïde.

## Tableau v1 / v2 / fit (extrait, 15 feuilles)

`cFit` = cos vs centroïde (rejeté) · `tAln` = cos vs titre (retenu = `cleavage_fit`).

| thème | titre | v1 (saillant) | v2 (central) | cFit v1→v2 | tAln v1→v2 |
|---|---|---|---|---|---|
| **n0** | Restaurer la confiance par l'écoute | **cesser de mentir et communiquer avec honnêteté** | **pratiquer une écoute active et sincère** | 0.823→0.787 ✗ | 0.754→**0.793** ✓ |
| n5 | Réforme du nombre de parlementaires | supprimer le Sénat | **réduire le nombre de parlementaires** | 0.909→0.911 | 0.736→**0.926** ✓ |
| n11 | Réduction des avantages des élus | réduire salaires des **députés** | réduire salaires des **élus** | 0.869→0.914 | 0.850→**0.930** ✓ |
| n13 | Limitation des mandats des élus | limiter à deux mandats consécutifs | limiter le nombre de mandats | 0.867→0.895 | 0.833→**0.929** ✓ |
| n18 | Débats publics citoyens organisés | débats publics avec les élus | débats publics citoyens réguliers | 0.900→0.906 | 0.870→**0.942** ✓ |
| n12 | Transparence des dépenses des élus | rendre publics comptes et décisions | rendre publics tous les frais | 0.833→0.823 | 0.819→**0.882** ✓ |
| n16 | Fonctionnement et rôle des députés | rendre publics les votes | rendre obligatoire la présence aux votes | 0.859→0.864 | 0.774→0.765 |
| n2 | Réorganisation des collectivités | renforcer l'autonomie locale | réorganiser communes/départements/régions | 0.859→0.823 | 0.858→0.855 |
| n14 | Réforme du système politique *(vague)* | limiter avantages + transparence | réformer le contrôle des conflits d'intérêts | 0.794→0.762 | 0.703→**0.717** ✓ |
| n3 | Tirage au sort des citoyens | tirer au sort pour décisions | *(identique)* | 0.871 | 0.901 |
| n6 | Privilèges des anciens présidents | supprimer avantages+privilèges | supprimer les privilèges | 0.843→0.827 | 0.954→**0.967** ✓ |

Récap : **mean fit-titre v1 = 0.822 → v2 = 0.862**, v2 meilleur **12/15**.
mean fit-centroïde v1 0.857 ≈ v2 0.854 (nul, et inversé sur le cas-clé).

## Le fit (titre) discrimine-t-il les mauvaises ?
Oui, **partiellement** : il sépare les cibles bien centrées (≥0.88 : n13, n11, n18, n5,
n6, n12) du cas vraiment diffus **n14 « Réforme du système politique »** (0.717, thème
fourre-tout). Seuil retenu `CLEAVAGE_FIT_LOW = 0.75` (env `AGORA_CLEAVAGE_FIT_LOW`) →
marque `cleavage_fit_low` (MARQUEUR d'audit, **on n'efface pas** la cible). Le ~0.3 du
brief est sans objet : il s'appliquait au fit-centroïde abandonné ; l'échelle titre est ~[0.70,0.97].

## Décision / implémentation
- **build_opinion.py** : `cleavage_system(title)` (conditionné + « central ») ;
  `analyse_leaf` récupère le titre via `title_for_node` (cache HIT, zéro LLM si déjà titré) ;
  `_attach_cleavage_fit` ajoute `cleavage_fit` = cos(cible, titre) + `cleavage_fit_low`.
- **Re-bake** : justifié (la v2 gagne sur le cas reporté, sans régression). Commande :
  `MISTRAL_API_KEY=$(cat var/mistral.key) uv run python -m backend.build_opinion --dataset granddebat`.
  Non lancé ici (écrirait l'`opinion.json` SERVI du dépôt principal) → laissé à la phase Run.

Lié : [[agora-opinion-target-verdict]] (la cible = proposition polaire à la feuille),
[[agora-stance-subject-verdict]] (la stance s'agrège sur le sujet du cluster).

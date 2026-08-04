# Scénario LIVE — rejouer les débats de l'Assemblée nationale

**Demande (Bob, 2026-08-04)** : établir un **scénario live** — un use case réel avec prises de
parole horodatées, rejouées, avec une synthèse mise à jour en direct. Nouveau **type** de
pipeline, réutilisant les modules existants. **Isolé** d'Agora pour ne rien casser. Nouvelle
page affichant la transcription entrante **et** la sortie du pipeline.
**Statut : GROOMÉ, non lancé.** Remplace le spike de mesure proposé dans `LIVE_FEASIBILITY.md`.

---

## 1. La source — vérifiée disponible

**Comptes rendus des séances publiques de l'Assemblée nationale**, XML, **Licence ouverte**.

```
https://data.assemblee-nationale.fr/static/openData/repository/17/vp/syceronbrut/syseron.xml.zip
HTTP 200 · 56 Mo · last-modified 2026-08-04 02:05 UTC
```

Contient : jours de séance, dates, numéros de séance, **thèmes de discussion**, **tous les
orateurs nommés** (députés et ministres), et le **texte des débats**.

Trois propriétés qui en font le bon corpus :
- **Rafraîchi quotidiennement** (modifié ce matin) → le même pipeline qui rejoue l'archive
  pourra consommer le flux réel. Le rejeu n'est pas un cul-de-sac, c'est l'étape 1 du produit.
- **Licence ouverte** → aucune contrainte de réutilisation.
- **Orateurs nommés** → l'axe « personne » que le produit actuel n'a pas.

Précédent utile : on ingère **déjà** de l'open data AN (le corpus TikTok en vient), et
`pipeline/collect/` sait découvrir et télécharger depuis ce portail.

### ✅ Reconnaissance faite (2026-08-04) — cf. `LIVE_DATA_RECON.md`

Corpus téléchargé et mesuré en entier. **Mieux que supposé** : chaque prise de parole porte un
**horodatage réel en secondes** (`stime`, 90,4 % de couverture) et un **id d'orateur stable**
(`id_acteur`, 90,1 %). On rejoue donc à la **cadence exacte de la séance**, sans simuler
d'horloge.

Rendement : **321 892 paragraphes → 94 709 tours substantiels** (~158/séance sur 601 séances),
à comparer aux 22 174 avis du Grand Débat. Cadence réelle : **un tour toutes les ~50 s** contre
quelques secondes de traitement ⇒ **~10× de marge sur le temps réel**.

## 2. Ce qu'on garde, ce qu'on enlève, ce qu'on ajoute

C'est exactement l'argument de Bob pour la classe `PipelineProfile` : composer des couches par
use case. Concrètement, pour le live :

| Couche | Batch (Agora) | Live | Pourquoi |
|---|:--:|:--:|---|
| ingestion | csv/jsonl | **XML → tours** | à écrire (cf. §4) |
| extraction de claims (verbatim ancré) | ✅ | ✅ | déjà **incrémentale** (cache par avis) |
| embedding local | ✅ | ✅ | quelques ms, aucun coût API |
| **assignation** au thème existant | ❌ | ✅ **nouveau** | plus proche centroïde — la carte ne bouge pas |
| re-clustering complet | ✅ | **déclenché** | des minutes : jamais à chaque tour |
| stance / clivage | ✅ | ✅ | **le seul chemin vers les positions** |
| titres · accroches · descriptions | ✅ | ❌ | LLM cher, inutile en direct |
| insights rédigés | ✅ | **différé** | à la pause / en fin de séance |
| arguments V-SELECT | ✅ | ❌ | batch |
| traduction FR | ✅ | ❌ | corpus mono-FR |
| **agrégation par orateur** | ❌ | ✅ **nouveau** | orateur × thème → stance |

Budget mesuré : ~1,45 claim par tour, stance à ~4 claims/s ⇒ **extraction + position d'une
intervention tiennent en quelques secondes**. Un député parle toutes les quelques minutes. Le
live tient largement — c'est *re-dessiner la carte* qui ne tient pas.

## 3. La simplification que le cadrage de Bob rend possible

Le point dur identifié dans `LIVE_FEASIBILITY.md` était : **comment garder l'identité des thèmes
à travers une repartition ?** (sans quoi la carte se remélange sous les yeux de l'utilisateur).

En **assignant** aux thèmes existants plutôt qu'en repartitionnant, on **contourne** le problème
au lieu de le résoudre : les thèmes sont **gelés** après amorçage, la carte est stable par
construction. On mesure alors la **dérive** (part des claims mal couverts par tout centroïde
existant) et on ne restructure que quand elle dépasse un seuil.

→ La question de l'appariement des thèmes redevient un problème de **second** temps, pas un
prérequis. C'est ce qui rend le scénario faisable maintenant.

**Amorçage (cold start)** : au tour 1 il n'y a aucun thème. Options — (a) bâtir en batch sur
les N premiers tours puis basculer en incrémental ; (b) amorcer sur une séance **antérieure**
du même texte de loi. (b) est plus réaliste (un débat parlementaire a un ordre du jour connu
d'avance) et donne une démo qui démarre déjà peuplée.

## 4. Ce qui est réellement neuf (à construire)

1. **Ingestion XML** — `pipeline/ingest` ne gère que `csv | jsonl`. Il faut un collecteur
   qui convertisse le XML AN en tours `{id, orateur, groupe, texte, seq, séance}`. Sortie
   jsonl → tout le reste de la chaîne le lit déjà.
2. **Filtre de parole procédurale** — ⚠️ **le risque de contenu n°1, à inspecter en premier**.
   Un compte rendu contient « La séance est ouverte », les mises aux voix, les rappels au
   règlement, les interruptions, « (Applaudissements sur les bancs…) ». Sans filtre, la synthèse
   se remplit de bruit de procédure. À regarder **sur les données réelles avant** de concevoir.
3. **Axe orateur** — `Idea` porte déjà `author_hash` et `ts`, mais **rien en aval ne s'en sert**
   (vérifié). Agrégation orateur × thème → stance à construire.
4. **Moteur de rejeu** — l'horloge : déroule les tours, à vitesse réglable, avec pause/reprise.
5. **Cache incrémental d'embeddings** — ⚠️ **prérequis dur** : `_emb_fingerprint`
   (`claims_endpoint.py:120`) hashe la concaténation de *tous* les claims ; un tour nouveau
   invalide **tout**. Inacceptable en live. À passer en clé par claim.
6. **Page live** — flux de transcription entrant à gauche, sortie du pipeline à droite
   (thèmes vivants, positions par orateur, compteurs), avec l'état qui évolue visiblement.

## 5. Isolation — ne rien casser

- **Nouveau module racine `live/`** (pipeline, moteur de rejeu, API). Réutilise `pipeline/*`
  par import ; **ne modifie pas** `backend/build_analysis.py` ni les endpoints servis.
- **Espace de cache séparé** — pas de partage de dossier avec les consultations servies.
  ⚠️ C'est la même exigence que pour `PipelineProfile` : deux profils qui écrivent au même
  endroit se corrompent en silence.
- **Route front distincte**, page à part. La Console actuelle n'est pas touchée.
- **Nouveau descripteur de type `transcript`**, à côté des consultations — pas de mélange dans
  `/api/datasets` (qui alimente la landing publique).

## 6. Risques à porter honnêtement

| Risque | Gravité | Ce qu'on en fait |
|---|---|---|
| **Positions ≠ embeddings** (NMI 0,04–0,06 mesuré) | **haute** | les positions **doivent** passer par la stance LLM ; ne jamais promettre un « clustering des positions » |
| ~~Parole procédurale polluant la synthèse~~ | ~~haute~~ → **retiré** | le corpus **type** la procédure (`code_grammaire`) : filtre à lire, pas à inventer. Cf. `LIVE_DATA_RECON.md` |
| Cible de stance **ne transfère pas** hors tiktok (verdict connu) | moyenne | re-valider la cible sur du transcript avant toute conclusion |
| Orateurs **nommés** vs invariant d'anonymisation | moyenne | légitime (registre public, Licence ouverte) mais **exception explicite et cantonnée au module `live/`**, jamais étendue aux consultations citoyennes |
| Un orateur parle 20 fois | moyenne | l'hypothèse « 1 avis = 1 personne » saute ; l'agrégation par orateur la remplace |
| **Discours rapporté** : un orateur cite la thèse adverse **pour la réfuter** → stance **inversée** | **haute** | ⚠️ **découvert à la sonde**, sans réponse connue dans le dépôt. Bloque toute promesse de « position par orateur ». Cf. `LIVE_PROBE_CLAIMS.md` |
| **~la moitié des claims ne portent pas sur le fond** (attaque personnelle, procédure, hors-sujet) | **haute** | le pré-filtre de pertinence (déjà validé, `relevance_calibration_note.md`) devient une pièce centrale, à remonter en amont |
| Une intervention est **coupée en plusieurs paragraphes** (36,9 % du corpus) | moyenne | **recollage** par orateur avant extraction — l'unité n'est pas le paragraphe |
| ~~Le débat est dialogique → sens hors contexte~~ | ~~à explorer~~ → **écarté** | mesuré : déixis 16 %, quasi toujours bénigne. **Ce n'était pas le bon risque** |

## 7. Ordre proposé

0. **`PipelineProfile`** (`PIPELINE_PROFILE.md`) — **prérequis**, confirmé par le cadrage : c'est
   ce qui permet de composer les couches par use case au lieu de dupliquer le pipeline.
1. **Reconnaissance des données** (cheap, décisif) : télécharger, ouvrir le XML, mesurer la part
   de parole procédurale, vérifier ce qu'on a vraiment par tour (orateur, groupe politique,
   texte, ordre, éventuelle heure). **Peut invalider ou réorienter tout le reste** → à faire
   avant de concevoir quoi que ce soit d'autre.
2. Ingestion XML → tours jsonl + filtre procédural.
3. Cache incrémental d'embeddings (le verrou n°5).
4. Boucle live : assignation + stance, sans re-clustering.
5. Page live (transcription + sortie).
6. Puis seulement : dérive, restructuration déclenchée, identité des thèmes.

**But** : une démo qui rejoue une séance réelle et montre une synthèse qui se construit sous les
yeux, sans toucher à l'Agora servi.
**Acceptation** : une séance rejouable de bout en bout ; la carte reste stable ; la position par
orateur est traçable jusqu'au verbatim ; zéro régression sur les endpoints existants.

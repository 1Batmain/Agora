# Reprise — état au 2026-10-16 (hivernage de la forge)

Écrit avant mise en sommeil. Vérifié : **tout est poussé**, aucun commit local orphelin,
aucune modification non commitée (les 5 branches locales sont identiques à `origin`).

---

## 1 · Où en est chaque branche

| Branche | Tête | État |
|---|---|---|
| `main` | `5a4f13a` | **ce que sert la prod**. Rien en attente dessus. |
| `fix/cve-python-anyio-setuptools` | `689efc2` | ✅ prête · **PR NON ouverte** — j'attends le go de Bat |
| `feat/pipeline-profile` | `31810c0` | chantier live, **volontairement non mergé** (consigne de Bat : pas de PR avant que ce soit propre) |
| `research/bench-embedders`, `release/cla-signatures` | — | antérieures, non touchées |

## 2 · Ce qui attend une DÉCISION (pas du travail, un arbitrage)

1. **Ouvrir la PR du correctif `anyio`** vers `main`. Le correctif est testé (246 tests +
   chemin servi exercé, fail-closed public vérifié). Dès qu'il est sur `main`, l'autodeploy
   le prend ; VPS-ADMIN attend juste le signal.
2. **`setuptools` PYSEC-2026-3447** — bloqué par un fait mesuré : `torch 2.12.1+cpu` déclare
   `setuptools<82`, donc le plancher à 83 force torch en 2.14.1. Torch est le moteur
   d'embedding → risque d'invalider l'analyse publiée. Protocole de validation exigé avant
   tout bump (équivalence des vecteurs puis ARI = 1 sur la partition) : détaillé dans
   `.agent/inbox/REPONSE-audit-2026-10-16.md`. **À arbitrer avec Bat, pas à prendre seul.**
3. **Nommer l'outil live.** Bat l'a demandé le 08/08 : le pipeline live devient un produit
   distinct d'Agora. Sans nom, la frontière de code reste floue.

## 3 · Ce qui attend du TRAVAIL, par ordre de rendement

1. **Pré-filtre de pertinence AVANT l'amorçage live.** C'est de loin le plus rentable :
   mesuré, 47 % des claims classés tombent sur des thèmes **procéduraux**, et le mode de
   panne est qu'un clivage bidon produit une position *confiante et fausse*. Tant que ce
   n'est pas fait, la sortie live n'est pas montrable. Cf. `LIVE_PIPELINE.md` §3 bis.
2. **Bencher le garde-fou « discours rapporté ».** Écrit, non validé. Sans lui, une position
   par orateur peut être publiée à l'envers — or c'est tout l'intérêt du cas AN.
3. **Restreindre `/api/health`.** Je dois ce correctif à VPS-ADMIN : j'ai élargi cet endpoint
   le 04/08 (profil actif + `model_id` par dataset) pour corriger un champ qui mentait, et ça
   expose gratuitement la cartographie des datasets. C'est dans mon périmètre.
4. **Cas d'usage Twitch** (demandé le 08/08, **rien d'écrit** — seulement discuté) : chat d'un
   live → un « super commentaire » toutes les ~15 s représentant les avis dominants.
   Points déjà établis dans l'échange, à ne pas re-découvrir :
   - l'extraction de claims **saute** (un message de chat EST déjà un claim) ;
   - à 15 s de cycle, budget = **un seul appel LLM par cycle** (embedder local → grouper →
     générer). Pas de stance, pas de clivage ;
   - lecture du chat **sans clé** via IRC anonyme (`justinfan`) ;
   - fenêtre glissante obligatoire (des centaines de messages/minute sur un gros stream) ;
   - ⚠️ **tension d'invariant** : le super-commentaire est GÉNÉRATIF, Agora est strictement
     EXTRACTIF. Le rapatriement dans Agora ne peut viser que les couches déjà génératives
     (titres, accroches, descriptions) — **jamais** claims, citations ni arguments V-SELECT.
   - **Deux questions restées sans réponse de Bat** : quelle chaîne (gaming / talk / politique
     change la nature du chat) et un seul super-commentaire par cycle ou 2–3 « voix » pour
     les courants distincts.
5. **Items d'audit différés** : `var/audit/deferred.md` (#3 c-TF-IDF sparse, #5 mémoïsation).

## 4 · Données hors git — ce qu'il faut savoir

Rien de précieux n'est perdu si `var/` et `data/` disparaissent :

- **Corpus AN** (`data/raw/an/`, gitignoré) : re-téléchargeable, commande dans
  `LIVE_PIPELINE.md` §5 (`syseron.xml.zip`, Licence ouverte, rafraîchi quotidiennement).
- **`var/live-cache/`** : instantanés d'un rejeu, reproductibles en relançant `live.replay`.
- **`var/rnd-cache/`** (140 Mo) : caches R&D `granddebat-mix` et `xstance-large`, rescapés du
  worktree `agora-fix`. `granddebat-mix` contient une extraction LLM complète — **coûteuse à
  refaire**. C'est le seul artefact hors git qui vaut une archive.
- **`backend/cache/*/`** : les caches servis ; leurs fichiers d'ENTRÉE (`embeddings.npy`,
  `ideas.jsonl`, `meta.json`) sont **suivis en git**, les dérivés non.

⚠️ Piège à ne pas réapprendre : les `embeddings.npy` sous `backend/cache/` ne sont **pas**
couverts par `.gitignore` (seuls les dérivés le sont). Un `git add -A` sur un dossier
contenant un cache R&D embarque des dizaines de Mo — ça s'est produit une fois, rattrapé par
le refus de GitHub.

## 5 · Reprise de l'infra (d'après VPS-ADMIN)

Services arrêtés et *disabled* → `systemctl --user enable --now <service>`. Funnels 443/8443
retirés, configuration d'exposition sauvegardée dans `forge/_hivernage-20261016/infra/`.
La surveillance reste active (sentinelle horaire, auditd, bilan, sauvegardes).

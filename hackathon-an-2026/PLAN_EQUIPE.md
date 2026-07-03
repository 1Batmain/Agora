# Plan équipe — Hackathon Assemblée nationale (jour J)

> **Équipe de 5 · 1 journée · objectif : démo impeccable + différenciant crédible.**
> Ce plan part des deux audits (`research/audit_capacites_2026-07.md` produit,
> `research/audit_code_2026-07.md` code) et de l'état servi en live
> (`https://forge.tail0b8aa8.ts.net/api/datasets`). Règle d'or : **on ne casse pas
> ce qui marche**. La partie (1) recense ce socle ; la (2) priorise ; la (3) découpe
> en 5 lanes étanches ; la (4) fixe les règles de merge.

---

## 1. État des lieux

### 1.a — Ce qui est FAIT et SOLIDE (à ne PAS toucher, à SAVOIR pitcher)

Ces points sont **prouvés** (gold externe, juges aveugles, vérif live) — ils sont la
matière du pitch, pas du travail restant.

- **Pipeline validé — témoin Grand Débat 14/14, 0 mismatch.** Couverture 100 % des
  sous-thèmes de la synthèse officielle OpinionWay, alignement 4,57–4,93/5
  (`research/granddebat_witness_note.md`). C'est notre meilleure carte : *les thèmes
  émergents collent au réel, mesuré contre un gold indépendant*.
- **Corpus complet 22k re-coupé `sauce_magique`.** `granddebat-complet` est servi à
  **28 384 contributions** (vérifié `/api/datasets`), re-coupe 21→37 macros appliquée
  au corpus entier SANS re-extraction (commits `da3ea17`, `3e4bf49`). Façade top1
  99,9 %→14,1 %, témoin 14/14 maintenu, 0 mismatch (`research/` verdicts).
- **Traçabilité verbatim 100 %.** Gate dur `is_verbatim` : 14 680/14 680 claims sur
  3 corpus. Chaque claim est une sous-chaîne exacte, `spans` servis en live,
  surlignage cliquable jusqu'à la phrase (`AvisDetail.tsx`).
- **Opinion / stance corrigée et honnête.** Stance calibrée vs gold x-stance
  (0,79 sur décidés), bande de confiance « high » = 81 %. Métriques anti-mésusage :
  « consensus » → « cohésion sémantique — PAS un accord d'opinion » (`strings.ts`).
- **Coûts honnêtes.** Endpoint `/cost` live (0,064 $ Grand Débat analyse). Comparatif
  d'ordre de grandeur assumé : **~2 000–2 500 $ vs 3,25 M€** pour le traitement officiel
  (`hackathon-an-2026/COMPARATIF_GRANDDEBAT.md`). ⚠️ voir manque coût ci-dessous.
- **Install 1 commande + CI/CD + démo publique.** `scripts/setup.sh` (uv + npm +
  caches), `make dev`, CI pytest+build sur PR, auto-deploy VPS sur push `main`,
  mode public `AGORA_PUBLIC=1` fail-closed **vérifié live** (`/recluster`→403,
  `/docs`→404). Zéro fuite de secret sur 518 commits.

### 1.b — Ce qui MANQUE (tiré des 2 audits)

| # | Manque | Source | Gravité pour la démo |
|---|--------|--------|----------------------|
| a | **Aucun export / rapport** (PDF/CSV/HTML) — le livrable attendu par un commanditaire public n'existe pas | capacités B.1 | **Fort différenciant manquant** |
| b | **Pas de human-in-the-loop** : l'analyste ne peut ni renommer, ni fusionner, ni exclure un claim (flags passifs seulement) — exigé mot pour mot par le défi (« auditer chaque cluster ») | capacités B.1 | **Fort différenciant manquant** |
| c | **Coût affiché incomplet** : `/cost` omet l'extraction mistral-large (~facteur 100) — dangereux si on pitche le coût | capacités A.7 / B.2 | **Risque pitch** |
| d | **Bundle ~725 kB** : three.js chargé statiquement → page lente au 1er chargement de démo | code Top-10 #2 / QW1 | **Risque démo** |
| e | **Lectures publiques sans rate-limit** : DoS CPU/mémoire anonyme (`/avis_list` fold-scan O(N) sur 16 Mo) | code Top-10 #4 / QW3 | **Risque démo publique** |
| f | **`/avis_list` scanne 22k avis + matérialise tous les matchs avant paginer** | code Top-10 #1 | **Risque démo (latence recherche)** |
| g | **Stabilité échantillon non mesurée** : le témoin valide 3 000, pas le re-tirage | capacités A.2 | Crédibilité jury |
| h | **PII / RGPD** : masquage regex-only (noms/adresses non couverts), `text` brut committé dans `ideas.jsonl` | capacités A.8 / B.1 | Bloquant vente, **pas** bloquant démo |
| i | **UX** : bandeau échantillonnage peu visible, avertissement « pas un sondage » absent des %, clé JSON `consensus` non renommée au serve-time | capacités A.6 / reco 1 | Polish jury |

**Décision de cadrage :** (h) RGPD est **hors périmètre jour J** (effort M–L, non
démontrable en démo, risque de casser les caches). On le **mentionne comme roadmap
assumée** dans le rapport (lane B) — c'est plus honnête et plus fort que de le bâcler.

---

## 2. Priorisation pour UNE journée à 5

### P0 — Démo impeccable (non négociable, à finir avant midi)
Tout ce qui fait qu'un jury voit une app **rapide, stable, publique** sans plantage.
- **Les 3 quick-wins code** (`audit_code §6`) : lazy-load three.js (d), cache
  `read_analysis` + skip sans `theme_id` (f partiel), rate-limit lectures (e).
- **Coût complet dans `/cost`** (c) : sinon on ne pitche pas le chiffre.
- **UX honnêteté bout-en-bout** (i) : bandeau échantillon, « pas un sondage »,
  renommer `consensus` au serve-time.

### P1 — Différenciant (l'après-midi, ce qui gagne le hackathon)
Ce qui transforme « moteur d'analyse » en « outil » — et répond au défi mot pour mot.
- **Rapport / export** (a) : bouton → CSV par thème + rapport HTML imprimable
  (synthèse + thèmes + citations verbatim + **page méthode & limites** auto-incluse).
- **Vue analyste human-in-the-loop** (b) : renommer / fusionner / exclure un claim,
  avec journal des modifications — brancher sur les flags existants.

### P2 — Si le temps le reste (fin de journée, opportuniste)
- **Stabilité échantillon** (g) : 2 tirages de 3 000, mesurer le recouvrement des
  thèmes → une phrase chiffrée dans le rapport.
- `/avis_list` : mémoïsation `_fold(text)` par `(dataset, mtime)` (f complet).
- Front perf : `useMemo(segments)`, `React.memo(AvisCard)`.

> **Règle de priorisation :** une lane ne démarre P1 qu'après avoir livré **sa part de
> P0**. Personne ne touche à P2 tant qu'un P1 différenciant n'est pas *démo-able*.

---

## 3. Découpage en 5 lanes indépendantes

Principe : **1 personne = 1 lane**, fichiers partagés minimisés, contrat d'interface
explicite. Les deux zones de contention connues — `backend/server.py` (routes) et
`frontend/src/redesign/RedesignApp.tsx` — sont **réparties par zones nommées** pour
éviter les conflits (voir §4). Chaque lane est autonome sur son premier commit.

---

### Lane 1 — Quick-wins & perf (P0)
**Objectif :** app rapide et bornée pour la démo. Les 3 quick-wins de l'audit code.
**Personne :** profil back+front à l'aise partout.
**Fichiers touchés :**
- `frontend/src/redesign/RedesignApp.tsx:17,459` (lazy `Density3D` + `<Suspense>`) —
  *seule modif front de cette lane, zone « imports/3D »*.
- `backend/analysis_store.py:170` (cache mtime `read_analysis`, servir une **copie**).
- `backend/server.py:583` (skip `read_analysis` dans `/avis_list` sans `theme_id`).
- `backend/server.py:415-594` + `backend/auth.py:85` (`Depends(rate_limit)` sur
  `/analysis`, `/avis_list`, `/avis`, `/citations`, `/insights`, `/opinion`).
**Definition of Done :** bundle initial nettement < 725 kB (three.js en chunk séparé,
vérifié `npm run build`) ; carte/explorateur ne re-parsent plus `analysis.json` à
chaque requête ; les 6 endpoints de lecture renvoient 429 sous martelage ; CI verte ;
aucune régression visuelle sur la 3D (elle charge à l'ouverture).
**Premier commit :** `perf(front): lazy-load Density3D (three.js) hors bundle initial`
puis un commit par quick-win.

---

### Lane 2 — Rapport & export (P1 différenciant)
**Objectif :** produire le livrable qui manque — CSV par thème + rapport HTML
imprimable, **méthode & limites incluses**. C'est LE manque qui transforme le moteur
en outil (capacités reco 2).
**Personne :** profil back Python + un peu de templating.
**Fichiers touchés (surtout NOUVEAUX → zéro conflit) :**
- **Nouveau** `backend/report.py` : lit `/analysis` + `/opinion` + `/insights` +
  `/citations` (fonctions déjà en place dans `analysis_store.py`), rend CSV et HTML.
- **Nouveau** `backend/templates/report.html` (Jinja/f-string, imprimable navigateur —
  pas de dépendance lourde type weasyprint le jour J).
- `backend/server.py` — **zone « routes export »** (ajouter `/export/csv`,
  `/export/report.html` à la **fin** du fichier, après les routes existantes).
- Front : **nouveau** `frontend/src/redesign/ExportButton.tsx` + 1 point de montage
  dans `RedesignApp.tsx` (**zone « barre d'actions », en bas du header**).
**Contrat d'interface :** `GET /export/csv?dataset=<name>` → `text/csv` ;
`GET /export/report.html?dataset=<name>` → HTML autoportant (CSS inline). Les deux
sont des **lectures** → soumis au rate-limit de Lane 1 (coordonner : ajouter les 2
routes à la liste `Depends(rate_limit)`).
**Definition of Done :** depuis la démo, 1 clic télécharge un CSV (thème, n, %, stance,
claims représentatifs) et 1 clic ouvre un rapport HTML imprimable contenant synthèse +
thèmes + citations verbatim + **page « méthode & limites »** (échantillon, couverture,
fidélité verbatim 100 %, `sovereign:false`). CI verte.
**Premier commit :** `feat(export): endpoint /export/csv par thème`

---

### Lane 3 — Vue analyste (human-in-the-loop) (P1 différenciant)
**Objectif :** répondre au défi mot pour mot (« validation humaine : auditer chaque
cluster ») — renommer / fusionner / exclure, avec journal. Brancher sur les flags
existants (capacités B.1).
**Personne :** profil full-stack (write-through JSON + UI d'édition).
**Fichiers touchés :**
- **Nouveau** `backend/analyst.py` : write-through sur `analysis.json` (renommage
  titre), exclusion de claim (liste d'exclusion en marge, **jamais** de mutation
  destructive du cache), **journal** `var/analyst_journal.jsonl` (qui/quoi/quand).
- `backend/server.py` — **zone « routes analyste »** (`POST /analyst/rename`,
  `/analyst/exclude`, `GET /analyst/journal`), gardées **COMPUTE/écriture** →
  403 en public (réutiliser le pattern d'`auth.py`, comme `/flag`).
- Front : **nouveau** `frontend/src/redesign/AnalystView.tsx` + entrée de nav dans
  `RedesignApp.tsx` (**zone « nav/onglets »**).
- S'appuie sur les flags existants (`/flags`) comme embryon — les *brancher*, pas les
  réinventer.
**Contrat d'interface :** `POST /analyst/rename {dataset, theme_id, new_title}` →
`{ok, journal_id}` ; `POST /analyst/exclude {dataset, claim_id}` → idem ;
`GET /analyst/journal?dataset=` → liste. **Le journal est exporté par Lane 2** dans le
rapport → contrat : Lane 2 lit `GET /analyst/journal` si présent (dégrader proprement
si absent).
**Definition of Done :** en dev, renommer un thème persiste et apparaît au reload ;
exclure un claim le retire des vues sans casser la traçabilité ; journal consultable ;
**403 en mode public** (vérifié) ; CI verte (ajouter un test `test_analyst.py` calqué
sur les tests d'auth existants).
**Premier commit :** `feat(analyst): write-through renommage de thème + journal`

---

### Lane 4 — UX & polish démo (P0 puis polish)
**Objectif :** boucler l'histoire honnête jusqu'au bout de l'UI (capacités reco 1) +
rendre la démo lisible pour un jury non-technique.
**Personne :** profil front + sensibilité produit/design.
**Fichiers touchés :**
- `frontend/src/redesign/strings.ts` : renommer/clarifier les libellés, avertissement
  « ceci n'est pas un sondage » sur les stats d'opinion.
- **Bandeau échantillonnage visible** (« 3 000 avis analysés sur 28 384 ») : monté
  dans `ConsultationOverview.tsx` depuis `dataset_stats` (déjà servi).
- Avertissement « pas un sondage » branché dans le composant d'opinion (là où
  s'affiche le « % favorable »).
- `IndicesDashboard.tsx:50` : corriger `isPct` (compte 0/1 affiché « 100 % »).
- **Backend, 1 ligne partagée avec prudence :** renommer la clé JSON `consensus` →
  `cohesion` **au serve-time** dans `analysis_payload` (`backend/analysis.py`,
  **zone « payload/theme_dict »**) — coordonner avec le front qui la lit.
**Contrat d'interface :** si la clé JSON servie change (`consensus`→`cohesion`), Lane 4
met à jour **et** le backend **et** `contract.ts` **et** les lecteurs front dans le
même commit (sinon garder l'alias `consensus` en doublon transitoire).
**Definition of Done :** un jury voit d'emblée « échantillon 3 000/28 384 », ne peut
pas lire un « % favorable » sans l'avertissement, aucun « 100 % » faux ; captures avant
/après dans la PR ; CI verte.
**Premier commit :** `feat(ux): bandeau échantillonnage visible sur l'aperçu`

---

### Lane 5 — Data & robustesse (P0 coût + P2 stabilité)
**Objectif :** rendre le pitch coût défendable (c) et poser la preuve de stabilité (g).
**Personne :** profil data/pipeline.
**Fichiers touchés :**
- **Coût complet (P0)** : tracer la phase extraction mistral-large dans `cost.json`
  au bake (`backend/build_*.py` où le coût analyse est déjà écrit ; source pricing
  unique déjà existante). `/cost` doit alors afficher extraction **+** analyse.
  ⚠️ Si re-bake impossible le jour J (pas de clé/temps), **écrire le coût extraction en
  constante documentée** dans `cost.py` à partir de `research/audit_cost.md`
  (~9–15 $/build) plutôt que de laisser le facteur ~100 caché.
- **Stabilité échantillon (P2)** : script `research/sample_stability.py` — 2 tirages de
  3 000, mesurer le recouvrement des macros → **une phrase chiffrée** livrée à Lane 2
  pour la page « méthode & limites ». Verdict écrit (`research/sample_stability_note.md`,
  format one-off habituel).
- **Hygiène caches (rappel §4)** : ne PAS committer les `.npy`/`.jsonl`/`meta.json`
  modifiés locaux (déjà en `git status`) — le deploy fait `reset --hard`.
**Contrat d'interface :** `/cost` conserve sa forme, ajoute les clés `extraction` et
`total` (Lane 4 peut les afficher). Le chiffre de stabilité est un **texte** transmis à
Lane 2 (pas de couplage code).
**Definition of Done :** `/cost` live montre un coût **total** honnête (extraction +
analyse) ; verdict stabilité écrit avec un chiffre de recouvrement ; caches locaux non
committés ; CI verte.
**Premier commit :** `fix(cost): inclure la phase extraction dans /cost (coût total honnête)`

---

### Carte des contentions (qui touche quoi)

| Fichier partagé | Lanes | Règle |
|---|---|---|
| `backend/server.py` | 1 (rate-limit routes existantes), 2 (routes export en **fin**), 3 (routes analyste en **fin**) | Ajouter les nouvelles routes **à la fin**, une lane après l'autre ; Lane 1 pose le `Depends` en premier, 2 & 3 rebasent dessus |
| `RedesignApp.tsx` | 1 (imports/3D), 2 (barre d'actions), 3 (nav/onglets), 4 (aucune si possible) | Zones nommées disjointes ; petits commits ; rebaser souvent |
| `backend/analysis.py` | 4 (renommage clé `consensus`) | Lane 4 seule ; commit atomique back+front+contract |
| `strings.ts` / `contract.ts` | 4 | Lane 4 propriétaire |

---

## 4. Règles (non négociables)

1. **Branche + PR vers `main`.** Une branche par lane
   (`lane1-perf`, `lane2-export`, …). **Jamais** de push direct sur `main` — le deploy
   fait `reset --hard origin/main`, tout état local non poussé est perdu.
2. **CI verte obligatoire** (pytest + build) avant merge. Une PR rouge ne merge pas.
3. **Review-gate** : l'architecte (ou un pair) relit avant merge ; merge en **ordre de
   dépendance** — Lane 1 (rate-limit) d'abord, puis 2/3 qui rebasent, 4/5 en parallèle.
4. **Caches non commités.** Les `.npy`, `.jsonl`, `meta.json` (déjà modifiés dans
   `git status`) **ne se committent pas**. Ne rien ajouter sous `data/` ni `var/`
   (gitignorés — secrets et données brutes).
5. **On ne casse pas ce qui marche.** Pipeline, témoin 14/14, verbatim, opinion :
   **intouchables**. Toute modif du payload servi met à jour `contract.ts` dans le même
   commit.
6. **Généricité** : zéro nom de corpus en dur (Agora doit marcher sur des consultations
   inconnues). Tester au moins sur `granddebat-complet` **et** `tiktok`.
7. **Public fail-closed** : toute nouvelle route d'écriture (Lane 3) doit renvoyer 403
   en `AGORA_PUBLIC=1`, sur le modèle de `/flag` (`auth.py`).

---

*Établi le 2026-07-03 à partir de `research/audit_capacites_2026-07.md`,
`research/audit_code_2026-07.md` et de l'état live `/api/datasets`. Priorité absolue :
une démo rapide, stable et honnête ; le différenciant se joue sur le rapport (Lane 2)
et la vue analyste (Lane 3), les deux manques que le défi AN nomme explicitement.*

# 🌙 PLAN DE NUIT — exécution autonome (validé avec Bob)

> **Règle d'or : du FINI, du VÉRIFIÉ. App propre au réveil, ZÉRO debugging matinal.**
> Chaque lane gatée + testée avant merge ; toute lane à risque de ne pas aboutir = terminée OU revertée (pas de demi-mesure).
> **Pas de paramètre/knob `naming`** (inutile — Bob). Labels de cluster = c-TF-IDF passif, pas un réglage.

---

## LANE 1 — Unifier TOUS les datasets sur le pipeline claims+cible v3 (propreté)
**But** : tiktok (déjà v3), **granddebat (3000)** et **xstance (3000)** servent tous une analyse **claims + cible** cohérente.
- Ré-extraire granddebat + xstance avec l'extraction **v3** (claims multi-spans + **cible verbatim orientée stance**, **batché**),
  **extraction = mistral-large** (`AGORA_EXTRACT_MODEL`), **enrichissement = mistral-small** (`AGORA_ENRICH_MODEL`).
- **Audit propreté** : claims+cible = SEUL chemin de traitement servi ; pas de résidu d'ancien pipeline.
- **Acceptance** : `/datasets` = 3 ; pour chacun `/analysis` (status ready) + `/avis` au format claim-v2 (claims[]{spans,target}) ;
  **100% verbatim** (claims ET cibles) ; couverture cible loguée par dataset.

## LANE 2 — BAC À SABLE « CONSOLE DE MIXAGE » (la pièce maîtresse) — knobs + recluster live + decision-trace
**But** : sentir/régler l'effet des paramètres **en live** (« je tourne le fader, je vois »).

### 2a. Pondération-cible dans l'embedding (le knob α)
- Embedding d'un claim = `normalize( α·emb(target) + (1−α)·emb(claim_text) )` si cible présente, **sinon `emb(claim_text)`** (repli gracieux).
- → le build doit **embedder AUSSI les cibles** (cache `target_emb` aligné aux claims ; cible absente → pas de contribution).
- α=0 → clustering par claim (actuel) ; α↑ → orienté aspect (doit rapprocher les 3 « addiction »).

### 2b. Endpoint recluster RAPIDE (sans LLM)
- `POST /sandbox {dataset, alpha, k, resolution, coarsen_mult, tau_mult}` → recluster sur **embeddings cachés** :
  blend(α) → graphe kNN(k) → Leiden(resolution) → subdivision variance-adaptative (τ × tau_mult) → coarsening (×coarsen_mult).
  **Aucun LLM.** Labels = **c-TF-IDF** (passif). Renvoie : clusters {id, n_claims, n_avis, keywords, qq claims}, hiérarchie,
  + **decision-trace** (cf. 2d). Objectif **~1 s** pour 3000 claims (vectoriser le blend, kNN/Leiden rapides). Débounce côté front.

### 2c. UI console de mixage (mode analyste)
- Faders/knobs **résolution · α (poids cible) · coarsening (×μ+σ) · τ (×) · k**, esthétique board sombre, valeurs en direct.
- Au mouvement (débounced) → appel `/sandbox` → la **carte se réorganise** (d3, transitions douces) + nb clusters/tailles.
- **PAS de knob naming.** Pas de titres/insights LLM ici (mots-clés suffisent).

### 2d. Decision-trace (debug — « verre, pas boîte noire »)
- Pour une **paire** de clusters : `sim(centroïdes)` vs **seuil coarsening μ+σ** vs `min(cohésions)` → fusionnés/pas + pourquoi.
- Pour un **nœud** : `dispersion` vs **τ** → subdivisé/pas.
- **k plus proches voisins** d'un cluster (centroïdes) / d'un claim. Démo intégrée sur le cas **addiction** (n17/n18/n20).
- Affiché dans un panneau à côté de la console.

**Acceptance LANE 2** : `npm run build` propre ; bouger un fader → recluster < ~1-2 s, carte se met à jour ; le decision-trace
explique une fusion/non-fusion avec des chiffres ; α rapproche visiblement les addiction quand on monte. Endpoint testé.

---

## VÉRIF FINALE (avant de « finir ») — smoke test complet
- Backend :8010 up ; `/datasets`=3 ; `/analysis` + `/avis` (claims+cible) OK sur **tiktok, granddebat, xstance** ; `/sandbox` répond.
- Front :5180 up, build tsc propre ; console fonctionne (faders → recluster live) ; vue avis (claims surlignés + cibles soulignées) OK.
- **Si un point échoue → réparer ou reverter la lane fautive ; ne pas laisser l'app cassée.**

## HORS SCOPE NUIT (groomé, pas exécuté — trop R&D / risque)
- Stabilisation fine de la granularité variance-adaptative (mais la **console permet de la régler à la main** via les knobs).
- Distillation vers un petit modèle d'extraction (optimisation, plus tard).
- Sujet émergent (clustering des cibles) au-delà du knob α ; stance pour/contre ; soumission citoyenne (E3/E4).
- Load-anim (retiré, gadget).

## TRANSPARENCE COÛT
- Ré-extraire granddebat+xstance via mistral-large (~6000 avis, batché) = coût API réel — assumé (claims+cible partout).

---

## EXTRACT v4 — cible OBLIGATOIRE (antécédent in-avis OU question), question ingérée (Bob, 2026-06-24)
**Décision** : option 1 (durcir + question), PAS de reformulation (l'option 2 casserait le gate verbatim). Voir [[agora-claim-pipeline-v3]].
- **Ingérer la question/consigne** de chaque consultation : tiktok + granddebat = question GLOBALE (descriptor) ;
  xstance = question PAR LIGNE (colonne existante). La rendre dispo à l'extraction par avis.
- **Prompt durci** : cible **obligatoire**. Ordre de résolution : (1) **antécédent DANS l'avis** (multi-span : inclure le
  span du sujet — ex. « Les vidéos courtes … Elles doivent être interdites » → cible « Les vidéos courtes ») ; (2) sinon
  **la question**. Few-shot pour les 2 cas. Garde sélectivité/regroupement/verbatim.
- **`align_spans`** : valide la cible contre **(avis ∪ question)**. La cible porte sa **source** (`"avis"|"question"`) + offsets.
  Les **parts du claim restent TOUJOURS dans l'avis** (gate verbatim dur inchangé).
- **Provenance `/avis`** : `target:{start,end,source}`. Front : souligne si source=avis ; si source=question, afficher comme label « sujet ».
- Ré-extraire les 3 datasets + mesurer la **chute du sans-cible**. Note `research/extract_v4_note.md`.

## PERSISTANCE des réglages (après extract-v4) — endpoint /analysis/apply
- `POST /analysis/apply {dataset, params:{alpha,k,resolution,coarsen_mult,tau_mult}}` → rebuild l'analyse AVEC ces params
  (clustering + enrichissement LLM titres/insights) + **persiste** → /analysis sert ensuite ces réglages.
- Stocker les params choisis (par dataset) ; bouton « Sauvegarder » du front (console-integrate) s'y branche.
- À FAIRE après le merge d'extract-v4 (backend, conflit sinon).

## SUPPRIMER le « jouet live » — COMPLET (Bob, 2026-06-24) — après toolbox-potards + extract-v4
Retirer entièrement la feature live (jamais aboutie, incrémental dégradé, demo = mock factice) :
- Front : `LiveView.tsx`, `liveStream.ts`, `mock.ts` (live), boutons « Rejouer live/démo », route/état live dans `RedesignApp`.
- Backend : endpoint SSE `/stream`, `backend/state.py` (AnalysisState incrémental), refs build-live.
- Vérifier qu'aucune autre partie ne dépend de ces symboles ; build front + backend propres après suppression.
- (Lane E E1-E4 reste groomée pour plus tard si on reprend le live un jour, mais le code actuel dégage.)

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

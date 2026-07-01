# EXTRACT v3 — cible orientée STANCE (quasi-obligatoire) + BATCHING

> Branche `work/extract-v3`. Évolue l'extraction claim-v2 **sans changer le modèle** :
> (1) la **cible** devient l'**objet de prise de position** (ce sur quoi on est POUR/CONTRE),
> verbatim et **quasi-obligatoire** ; (2) l'extraction passe à **N avis/appel** (batching)
> pour la vitesse/coût. **Gate verbatim DUR conservé** (claims ET cibles = sous-chaînes
> exactes, validées PAR AVIS). Modèle `Claim{text, spans[], target}` **intact**.

## 1. Cible orientée STANCE + quasi-obligatoire (`CLAIM_SYS`)
- `target` redéfinie : non plus « l'aspect dont parle le claim » mais **l'OBJET sur lequel
  le citoyen prend PARTI** — ce qu'on peut être POUR ou CONTRE (« les vidéos », « le temps
  passé sur l'écran », « la modération »…), toujours **portion VERBATIM de l'avis**.
- **Quasi-obligatoire** : chaque claim DOIT porter une cible (« un claim = une POSITION SUR
  UN OBJET »). Si l'objet d'une position claire n'est pas pointable → le passage est trop
  vague/narratif → **on ne l'extrait pas** (sélectivité plutôt que claim sans cible).
  `target=null` n'est plus qu'un **dernier recours** (grief réel à objet implicite).
- Few-shots **stance** ajoutés (« j'aime les vidéos parce qu'elles me font rire » → cible
  « les vidéos », POUR ; « le temps passé sur l'écran me dégoûte » → cible « le temps passé
  sur l'écran », CONTRE). Acquis conservés : sélectivité, regroupement, objet+position,
  multi-spans, verbatim strict.

## 2. Batching N avis/appel (`pipeline/claims/extract.py`)
- `batch_claim_prompt(texts)` : avis **numérotés** (`=== AVIS #k ===`) ; consigne `MODE LOT`
  ajoutée au system (mêmes règles, appliquées à chaque avis indépendamment ; verbatim **par
  avis**). Réponse JSON **clée par numéro** : `{"1": {"claims":[…]}, "2": {"claims":[]}, …}`.
- `parse_batch_claims(raw, n)` : remap → specs par avis. **Tolérant** aux variantes
  (valeur = liste directe ; objet enveloppé sous une clé ; liste positionnelle). Un avis
  **absent / mal formé** reste `None`.
- `extract_claims(…, batch_size=N)` : groupe N avis (défaut `BATCH_SIZE=8`, env
  `AGORA_CLAIMS_BATCH`). `align_spans` valide **toujours PAR AVIS** (zéro contamination
  inter-avis : une part d'un autre avis ne s'ancre pas → rejetée). **Repli robuste** : tout
  avis `None` dans la réponse du lot est **ré-extrait SEUL** (chemin mono-avis) → aucun avis
  perdu. `batch_size<=1` rejoue le chemin v2.
- `max_tokens` propagé par appel (API : `max_tokens` ; Mac/Ollama : `num_predict` + `num_ctx`
  agrandi) : budget = `BATCH_TOKENS_PER_AVIS(400) × taille`, borné à 8192 → pas de troncature.
- Validations : `selftest_extractive` (unitaire claim-v2) **OK** ; tests batch ad-hoc (parse
  des 6 formes + bout-en-bout batch **+ repli mono-avis** verbatim) **OK**.

## 3. Ré-extraction tiktok (mistral-large, batché) + mesures

### Échantillon contrôle — BATCH vs 1-par-1 (40 premiers avis, même modèle)
Vérifie que le batching **ne dégrade pas** la qualité :

| Mode | appels | s/avis | claims | verbatim | **cible** | cibles verbatim |
|------|-------:|-------:|-------:|---------:|----------:|----------------:|
| BATCH (n=8) | 5 | 2.00 | 91 | **100 %** | **93.4 %** | 100 % |
| SINGLE (1/appel) | 40 | 2.60 | 103 | 100 % | 89.3 % | 100 % |

→ batching **non dégradant** : verbatim identique (100 %), **couverture cible même
supérieure** (sélectivité plus nette), 8× moins d'appels.

### Corpus complet — 1604 avis (`backend.scripts.reextract_v3 --dataset tiktok --batch 8`)
| Métrique | claim-v2 (avant) | **EXTRACT v3 (après)** |
|----------|-----------------:|-----------------------:|
| claims | 3024 (1.88/avis) | **3027 (1.89/avis)** |
| **verbatim claims** | 100 % | **100 % (3027/3027)** |
| **couverture cible** | 85.2 % (2576) | **87.7 % (2655)** |
| **cibles verbatim** | 100 % | **100 % (2655/2655)** |
| sans cible | 448 (14.8 %) | **372 (12.3 %)** |
| multi-spans | 57 | 31 |
| appels LLM | **1604** (1/avis) | **201** (8/appel) |
| temps extraction | ~60 min | **~40.5 min** (2433 s, 1.52 s/avis) |
| erreurs (429/5xx) | absorbées par backoff | **0** |

- **Verbatim : 100 %** (claims ET cibles) — gate dur tenu sur tout le corpus.
- **Cible : 85.2 % → 87.7 %** (les sans-cible passent de ~15 % à 12.3 %, −18 % relatif).
  Gain net mais **modéré au corpus** : la queue d'avis très courts/vagues (549 avis pris en
  entier par le repli, dont ~177 portent désormais une cible) tire la moyenne ; sur des avis
  plus riches (échantillon n=40) la couverture monte à **93 %**. Pousser vers ~100 %
  forcerait des cibles sur du texte sans objet net → écartées par le gate verbatim de toute
  façon : **87.7 % est une couverture HONNÊTE** (cibles réelles, pas inventées).
- **Vitesse/coût : 1604 → 201 appels (−87 %)**, ~60 → ~40.5 min (−33 % wall-clock), **0
  erreur** (vs flux de 429 absorbés en v2). Le gain wall-clock < gain en appels car
  mistral-large facture la latence à la sortie (un lot produit plus par appel) ; le gros
  bénéfice est la **chute des requêtes** (moins de rate-limit, coût/orchestration réduits).
- **Régression mineure assumée** : multi-spans 57 → 31 (le mode lot privilégie des claims
  contigus). Le multi-span reste un cas-limite ; claims/avis et sélectivité inchangés.

### Échantillon qualitatif — cibles « stance-ready » (corpus v3)
Les cibles pointent désormais l'**objet d'une position** (verbatim) :
- `tiktok:29` → « du contenue qui pousse à la culture du vide », « une boucle d'addiction ».
- `tiktok:48` → « des contenus de haine envers certaines personnes et communautés ».
- `tiktok:58` → « des contenus qui tournent en boucle », « Envie d'y retourner régulièrement
  et grande perte de temps ».
- `tiktok:65` → « liberté d'expression sur ses idées politique » (POUR) vs messages haineux reçus.
- `tiktok:73` → « les reels d'instagram », « une addiction ».

## Acceptance
- [x] Couverture cible **améliorée** (85.2 → 87.7 % corpus ; 93 % sur avis riches), sans-cible
      réduits de ~15 % → 12.3 %.
- [x] **100 % verbatim** (claims ET cibles), gate dur PAR AVIS conservé.
- [x] Extraction **sensiblement plus rapide** : 1604 → 201 appels, ~60 → ~40.5 min, 0 erreur.
- [x] Modèle **claim-v2 intact** (`Claim{text, spans[], target}`, selftest OK) ; `claims.json`
      tiktok ré-écrit (mistral-large) → réutilisable tel quel par un rebuild.
```
uv run python -m backend.scripts.sample_extract_v3 --dataset tiktok --n 40 --batch 8   # contrôle batch/single
uv run python -m backend.scripts.reextract_v3      --dataset tiktok --batch 8          # ré-extraction + métriques
```

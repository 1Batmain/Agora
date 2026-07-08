# V-EXTRACT — argmining par extraction verbatim + ablation embedding (VERDICT R&D, 2026-07-07)

**Statut : RESEARCH, zéro modif pipeline.** Corpus `lutte-contre-les-fausses-informations`.
Suite de `argmine_verbatim_note.md`. Idée (Bob) : la cible établie, EXTRAIRE les arguments
pour/contre comme des SPANS verbatim resserrés (même gate que les claims), puis les embedder pour
des analyses plus fines. Fork B validé : extraction DEPUIS les claims sélectionnés (V-SELECT),
composable et ancré. Harnais : `research/argmine_extract.py`.

## 1. Extraction verbatim (fork B)

- **100 % verbatim** (gate `align_spans` + `Claim.is_verbatim` contre l'avis — rejet si pas
  sous-chaîne). Le repli gracieux (claim entier) reste verbatim.
- **Resserrement 71-79 %** des claims, MAIS **compression médiane 0.93-0.99** (span/claim) : sur
  ce corpus, les claims sont **déjà de la taille d'un argument** → l'extraction ne raccourcit
  quasi pas. Le problème « span trop court pour un embedding représentatif » **ne se manifeste
  pas ici**.

## 2. Ablation embedding (espace INTER-ARGUMENTS ; le support argument↔claim reste RAW)

Métrique : paires quasi-doublons INTRA-(thème,stance) à cos ≥ 0.85 (ce que le dedup fusionnerait).
Moins de fausses paires = arguments distincts mieux séparés.

| espace argument | fausses paires doublons | fidélité cos(vec, span) |
|---|---:|---|
| **raw** (span brut) | **0–1** | 1.0 (réf.) |
| target-context (`cible + span`) | 10–25 | — |
| reflet LLM uniforme | 11 | **0.78 médiane / 0.68 min** |

## 3. Verdict

- **RAW GAGNE — servir ET embedder le span verbatim brut** (espace inter-arguments). C'est la
  représentation la PLUS discriminante ; les arguments distincts restent distincts.
- **Les DEUX enrichissements BACKFIRENT**, pour la MÊME cause mécanique : ils injectent une
  **référence à la cible COMMUNE** dans chaque vecteur → composante partagée qui domine le cosinus
  → **effondrement des arguments distincts en faux doublons**. Preuve reflet : tout reflet d'un
  groupe commence par « L'extrait exprime une position favorable à [CIBLE recopiée], car… » — le
  préfixe identique écrase le signal propre. `target-context` fait pareil (préfixe cible constant).
- **Le reflet DÉRIVE en plus du verbatim** (fidélité 0.78 médiane) → à écarter aussi pour toute
  tâche de rattachement.
- **Escalade résolue** (règle Bob « reflet uniquement si 0-LLM échoue, uniforme ») : le 0-LLM
  (target-context) échoue → on a testé le reflet uniforme → **il échoue aussi**. On RESTE à raw.

## 4. Réserves (à mesurer ailleurs avant de généraliser)

- **Ce corpus ne stresse pas la prémisse du reflet** (compression ~0.99 : spans déjà longs). La
  valeur du reflet sur un corpus VERBEUX (granddebat : avis-paragraphes → spans courts elliptiques)
  reste ouverte — mais y serait le fixe correct un reflet qui **développe le contenu PROPRE du
  span SANS recopier la cible** (sinon même effondrement). Non testable ici (seul ce dataset a les
  caches analysis servis).
- **Variance de `vselect`** : mistral-large T=0 n'est pas déterministe (34 puis 17 args entre deux
  runs) → seeder/stabiliser la sélection avant tout bench répété.

## 5. Recommandation

**Adopter V-SELECT + V-EXTRACT (fork B), embedding RAW.** L'argument servi = span verbatim
resserré (ou claim entier en repli) ; le vecteur = ce même span brut, dans l'espace claims. Pas
d'enrichissement cible-référencé. Le raffinage V-EXTRACT apporte la finesse d'unité sans coût de
distinctness. Support argument↔claim = raw (comparable aux claims). Détail schéma : réécrire
l'étape 1 de `backend/build_arguments.py`, garder `back_match`/rollup/`arguments.json`, ajouter
`verbatim:true` + `spans` + contrôle CI « 100 % sous-chaînes ».

## Artefacts

`argmine_extract.py` · `argmine_extract_results.json`.

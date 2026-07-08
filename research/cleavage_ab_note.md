# Stance cible (a) vs (b) — TEST DE TRANSFERT sur granddebat-style (2026-07-07)

**Statut : RESEARCH, zéro modif pipeline.** Rejoue le verdict `stance_target_ab_verdict.md`
(variante (b) « mots-clés distinctifs + contraste » gagne 7-2 sur TIKTOK) sur un corpus
granddebat-style (`lutte-contre-les-fausses-informations`) — le verdict RÉSERVAIT explicitement
« n=1 corpus, rien ne se transpose sans re-mesure ». Différence (a)→(b) = **prompt seul**
(cadrage des mots-clés + phrase de contraste), dérivé sur ENTRÉES IDENTIQUES.

## Mesures

**Fit objectif** (`cleavage_ab.py`, cos(cible,titre) nomic-v2, seuil fit_low 0.75) :

| | fit médian | fit moyen | fit_low | par thème |
|---|---:|---:|---:|---|
| (a) actuel | 0.818 | 0.812 | 4/24 | (a) mieux **13** |
| (b) distinctif | 0.817 | 0.809 | 4/24 | (b) mieux **8** · nul 3 |

**14/24 feuilles : cibles (a) et (b) IDENTIQUES** — sur ce corpus, le changement de prompt ne
change souvent RIEN.

**Panel aveugle** (`cleavage_ab_panel.py`, protocole stance_target_ab, 10 paires non-triviales,
3 passes) : **(b) 6 – 4 (a)**. À comparer aux **7-2** de tiktok.

## Verdict

- **(b) NE TRANSFÈRE PAS clairement** sur ce corpus. Fit plat-à-négatif, panel 6-4 (n=10, non
  significatif), et cibles identiques sur 14/24 thèmes. Net ≈ +2 thèmes sur 24 — marginal.
- **(b) reste sûr** (jamais pire en agrégat) → adoption **low-risk mais low-impact** ici ; à ne
  PAS présenter comme un gain établi hors tiktok. La réserve du verdict d'origine est **confirmée
  par la mesure**.
- **Le vrai levier stance est ailleurs** (cf. `stance_large_bench.md` + rapport lane) :
  1. **`large_noabst` gaté sur cible dérivée réelle** (jamais mesuré hors gold à cible explicite) ;
  2. **calibrer `MIN_ENGAGEMENT=0.35`** (33 % des feuilles sortent en `impur` sur ce corpus) ;
  3. qualité de la **cible dérivée** (5 `fit_low`) — cause racine de l'abstention.

## Artefacts

`cleavage_ab.py` · `cleavage_ab_results.json` · `cleavage_ab_deriv.<ds>.json` (dérivations
cachées) · `cleavage_ab_panel.py` · `cleavage_ab_panel_votes.json`.

# Verdict — moteur d'ABSTRACTION : macros par étiquette canonique + affectation embedding (2026-07-18)

**But.** Au-dessus de la couche plate (γ, pic de modularité), construire une couche macro qui
(1) FUSIONNE les thèmes redondants (les ~5 « addiction ») et (2) NE soude PAS les sujets
distincts. S'appuie sur `synthesis_embed_note.md` (l'étiquette canonique rapproche les
redondants dans l'espace).

## Ce qui marche, ce qui coince (tiktok, 9 thèmes plats)

Étiquette canonique LLM par thème → puis former les macros. Trois méthodes de formation :

| méthode | macros | verdict |
|---|---|---|
| clustering embedding (pic de modularité) | 2 | **trop grossier** — dégénère sur 9 points, fourre-tout |
| regroupement LLM libre | 5 (intuitifs) | **pas une partition** — re-dédouble les thèmes AMBIGUS (« cyberharcèlement + dangers ») même contraint |
| fusion géométrique μ+σ | 3 | partition, mais lumpe encore (impact réseaux ↔ harcèlement) |
| **LLM propose les catégories + affectation par embedding** | **4, propres** | **GAGNANT** : partition garantie + noms intuitifs |

## La conception retenue

1. Couche plate γ → **étiquette canonique** LLM par thème (3-6 mots, catégorie générique).
2. Le LLM **propose un petit jeu de CATÉGORIES macro** (sa force : nommer l'abstrait).
3. **Affectation par embedding** : chaque thème → sa catégorie la plus proche (produit scalaire).
   → partition STRICTE garantie (un thème = un macro), pas la double-assignation du LLM libre.
4. Catégories non utilisées → tombent.

Résultat tiktok : 4 macros — Cyberharcèlement (1018) · Addiction numérique (795) · Algorithmes
et manipulation (225) · Image de soi (123). **Les 5 addictions fusionnent**, les distincts
restent séparés.

## Réserves / à faire

- La qualité macro hérite de la qualité des thèmes PLATS : un thème générique (« Impact des
  réseaux sociaux ») se place mal — c'est un thème plat faible, pas un défaut de l'abstraction.
- Étape (2) « proposer les catégories » : à câbler proprement (LLM sur la liste des étiquettes).
- Récursif : rejouer 1-4 sur les macros si leur nombre reste grand.
- Valider hors tiktok (Grand Débat complet — multi-thèmes).

Protos : `research/abstraction_proto.py` (+ variantes inline LLM/μ+σ/hybride).

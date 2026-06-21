# Banc de segmentation sémantique — rapport

*Jeu : `gold.json` — N=32 (16 mono, 16 multi). Embeddings : `nomic-v2`. Seed fixé, CPU.*

## 0. Faisabilité des token-embeddings

**nomic-v2 : OUI.** Token-embeddings récupérés via `SentenceTransformer.encode(text, output_value='token_embeddings')` → `last_hidden_state` `[n_tokens, 768]` AVANT pooling. Préfixe doc `'search_document: '`, `trust_remote_code=True`. Alignement token→offset char exact via `offset_mapping` (tokens spéciaux + préfixe retirés). **Aucun repli e5/bge nécessaire.**

## 1. Méthode

- **Unité** = mot (suite de non-espaces, langue-agnostique). Vecteur-mot = moyenne des token-embeddings du mot, L2-normalisé. Fenêtre glissante W = moyenne des vecteurs-mots.
- **Segmenteurs** : (1) *TextTiling-cosine* — minima locaux de cos(bloc-gauche, bloc-droite) sous `mu_bloc - c.sigma_bloc` ; (2) *Centroïde live* — coupe quand `cos(mot, centroïde courant)` < `mu_nouveaute - alpha.sigma` ; (3) *Change-point* — `ruptures` PELT/rbf, pénalité balayée.
- **Seuils dérivés ET calibrés GLOBALEMENT** (mu/sigma poolés sur TOUS les avis, pas par-document) : un seuil purement relatif à un avis ne peut jamais s'abstenir sur un mono cohérent (il coupe toujours au point le moins pire). Coefficients sans dimension, aucun magic-number absolu. `min_seg=3` mots.

- **Métriques** : Pk & WindowDiff (↓, sur multi), F1 des frontières (tol ±1 mot, micro sur multi), **taux de faux-positifs mono** (fraction de mono produisant ≥1 coupe — métrique clé). **Objectif de sélection = F1 GLOBAL** (frontières sur mono+multi : toute coupe d'un mono est un faux positif → le segmenteur « ne rien couper » n'est pas favorisé).

## 2. Meilleure config par segmenteur

| method | W | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| texttiling | 3 | 0.4713 | 0.5395 | 0.3051 | 0.2195 | 0.5 | 0.875 | 2.562 | 0.18 |
| centroid_live | 8 | 0.3425 | 0.3462 | 0.2143 | 0.3 | 0.1667 | 0.3125 | 0.438 | 0.1714 |
| changepoint | 8 | 0.2414 | 0.2414 | 0.5143 | 0.5294 | 0.5 | 0.625 | 0.75 | 0.383 |

*(Pk/WindowDiff = moyenne sur les 16 multi ; mono_FP/mono_cuts sur les 16 mono.)*

## 3. Top 12 configurations (toutes méthodes)

| method | W | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| changepoint | 8 | 0.2414 | 0.2414 | 0.5143 | 0.5294 | 0.5 | 0.625 | 0.75 | 0.383 |
| changepoint | 12 | 0.3429 | 0.3429 | 0.4286 | 0.375 | 0.5 | 1.0 | 1.562 | 0.2687 |
| changepoint | 5 | 0.4165 | 0.4246 | 0.383 | 0.3103 | 0.5 | 1.0 | 1.562 | 0.25 |
| changepoint | 8 | 0.5555 | 0.5706 | 0.381 | 0.2667 | 0.6667 | 1.0 | 2.5 | 0.233 |
| changepoint | 3 | 0.5051 | 0.5761 | 0.2857 | 0.2 | 0.5 | 1.0 | 2.125 | 0.1856 |
| changepoint | 12 | 0.6414 | 0.7404 | 0.3182 | 0.2 | 0.7778 | 1.0 | 4.125 | 0.1818 |
| texttiling | 3 | 0.4713 | 0.5395 | 0.3051 | 0.2195 | 0.5 | 0.875 | 2.562 | 0.18 |
| texttiling | 3 | 0.4871 | 0.575 | 0.3288 | 0.2182 | 0.6667 | 1.0 | 3.812 | 0.1791 |
| changepoint | 5 | 0.6584 | 0.8915 | 0.313 | 0.1856 | 1.0 | 1.0 | 5.625 | 0.1756 |
| centroid_live | 8 | 0.3425 | 0.3462 | 0.2143 | 0.3 | 0.1667 | 0.3125 | 0.438 | 0.1714 |
| changepoint | 12 | 0.601 | 0.6114 | 0.2769 | 0.1915 | 0.5 | 1.0 | 2.562 | 0.1698 |
| texttiling | 5 | 0.4564 | 0.4713 | 0.25 | 0.2273 | 0.2778 | 0.625 | 1.25 | 0.1667 |

## 4. Gagnant

**`changepoint` · W=8 · pen=3.0** — F1 global=0.383 ; F1 multi=0.514 (P=0.529, R=0.500), Pk=0.241, WindowDiff=0.241, faux-positifs mono=0.625 (0.75 coupe/mono).

## 5. Exemples (avis multi → frontières)

**multi-01** (thèmes : addiction, sante_mentale)

- gold : Je passe beaucoup trop de temps à scroller le soir et je n'arrive pas à lâcher mon téléphone. ⟂ Du coup je dors mal, je suis épuisé et je me sens de plus en plus déprimé au quotidien.
- prédit : Je passe beaucoup trop de temps à scroller le ⟂ soir et je n'arrive pas à lâcher mon téléphone. Du coup je dors mal, je suis épuisé et je me ⟂ sens de plus en plus déprimé au quotidien.

**multi-02** (thèmes : harcelement, image_corps)

- gold : Ma fille subit des moqueries et des commentaires méchants à chaque publication. ⟂ Résultat elle complexe énormément sur son corps et se compare sans arrêt aux filles parfaites qu'elle voit.
- prédit : Ma fille subit des moqueries et des commentaires méchants à chaque ⟂ publication. Résultat elle complexe énormément sur son corps et se compare sans arrêt aux filles parfaites qu'elle voit.

**multi-03** (thèmes : algorithme, sante_mentale)

- gold : L'algorithme pousse en boucle des vidéos de plus en plus sombres dès qu'on en regarde une. ⟂ À la longue ça entretient le mal-être et ça plonge vraiment dans l'anxiété.
- prédit : L'algorithme pousse en boucle des vidéos de plus en plus sombres dès qu'on en ⟂ regarde une. À la longue ça entretient le mal-être et ça plonge vraiment dans l'anxiété.

## 6. Limites

- **N=32, jeu synthétique** : les multi sont des avis construits par concaténation de segments mono-thème. Frontières nettes par construction → borne SUPÉRIEURE optimiste vs avis naturels où les transitions sont graduelles.
- Registre unique (consultation TikTok FR) — pas de garantie cross-domaine.
- Seuils dérivés par config mais grille discrète (W∈[3, 5, 8, 12]).

- *gold `_doc` : Vérité terrain de SEGMENTATION SÉMANTIQUE, labellisée par l'architecte. Registre = consultation TikTok (bien-être des jeunes en ligne, FR). mono = 1 thème (aucune frontière interne attendue → teste le…*

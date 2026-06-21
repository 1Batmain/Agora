# Banc de segmentation sémantique — rapport

*Jeu : `gold_large.json` — N=305 (104 mono, 201 multi). Embeddings : `nomic-v2`. Seed fixé, CPU.*

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
| texttiling | 8 | 0.4423 | 0.4653 | 0.2537 | 0.2092 | 0.3221 | 0.7692 | 1.346 | 0.2103 |
| centroid_live | 5 | 0.5558 | 0.6161 | 0.2181 | 0.1474 | 0.4195 | 0.9327 | 2.327 | 0.1765 |
| changepoint | 8 | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.76 | 0.384 |

*(Pk/WindowDiff = moyenne sur les 201 multi ; mono_FP/mono_cuts sur les 104 mono.)*

## 3. Top 12 configurations (toutes méthodes)

| method | W | Pk | WindowDiff | F1_multi | P | R | mono_FP | mono_cuts | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| changepoint | 8 | 0.2815 | 0.282 | 0.4423 | 0.4545 | 0.4307 | 0.7019 | 0.76 | 0.384 |
| changepoint | 12 | 0.3788 | 0.3793 | 0.3857 | 0.3209 | 0.4831 | 1.0 | 1.394 | 0.317 |
| changepoint | 5 | 0.3947 | 0.4063 | 0.3674 | 0.3039 | 0.4644 | 0.9712 | 1.452 | 0.3002 |
| changepoint | 8 | 0.5742 | 0.6069 | 0.3458 | 0.2349 | 0.6554 | 1.0 | 2.442 | 0.2765 |
| changepoint | 12 | 0.5994 | 0.6189 | 0.3244 | 0.2204 | 0.6142 | 1.0 | 2.433 | 0.2595 |
| texttiling | 8 | 0.4423 | 0.4653 | 0.2537 | 0.2092 | 0.3221 | 0.7692 | 1.346 | 0.2103 |
| changepoint | 3 | 0.5024 | 0.5619 | 0.2619 | 0.1824 | 0.4644 | 0.9712 | 2.394 | 0.2074 |
| texttiling | 5 | 0.5147 | 0.5799 | 0.2542 | 0.176 | 0.4569 | 0.9135 | 2.115 | 0.2068 |
| changepoint | 8 | 0.6441 | 0.8756 | 0.2674 | 0.16 | 0.8127 | 1.0 | 4.635 | 0.2062 |
| changepoint | 12 | 0.6427 | 0.8071 | 0.2556 | 0.1571 | 0.6854 | 1.0 | 4.0 | 0.1981 |
| changepoint | 5 | 0.645 | 0.9335 | 0.2531 | 0.148 | 0.8727 | 1.0 | 5.433 | 0.1937 |
| texttiling | 3 | 0.5665 | 0.7057 | 0.2341 | 0.1466 | 0.5805 | 0.9519 | 3.26 | 0.1864 |

## 4. Gagnant

**`changepoint` · W=8 · pen=3.0** — F1 global=0.384 ; F1 multi=0.442 (P=0.455, R=0.431), Pk=0.281, WindowDiff=0.282, faux-positifs mono=0.702 (0.76 coupe/mono).

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

## 6. Limites — verdict honnête

- **La segmentation par embeddings reste MÉDIOCRE sur des transitions naturelles.** Même la meilleure config (`changepoint` W=8) ne récupère que **R=0.43** des frontières gold (soit **~57% de frontières ratées**) pour une précision P=0.45, et **sur-coupe** les mono (70% des mono reçoivent ≥1 coupe parasite, 0.76 coupe/mono). Le signal token-level capte mal les virages de thème quand la transition n'est pas lexicalement marquée.

- **Jeu (N=305) : multi = concaténation de segments mono-thème.** Frontières nettes par construction → ces chiffres sont déjà une **borne optimiste** ; sur des avis vraiment continus, attendre pire.

- **Implication pour la prod** : avant de câbler un segmenteur, soit relever le rappel (signal plus riche : phrases/clauses, modèle supervisé, marqueurs discursifs), soit assumer qu'on découpe surtout les avis franchement multi-thèmes et qu'on tolère la sur-coupe des mono en aval (dédup/agrégation thématique).

- Registre unique (consultation TikTok FR) — pas de garantie cross-domaine ; seuils dérivés par config mais grille discrète (W∈[3, 5, 8, 12]).

- *gold `_doc` : Vérité terrain de SEGMENTATION SÉMANTIQUE étendue (~300 avis), labellisée à la main. Registre = consultation TikTok (bien-être des jeunes en ligne, FR). mono = 1 thème (aucune frontière interne attend…*

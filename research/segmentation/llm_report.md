# LLM (Mistral) comme segmenteur & extracteur de thèmes — rapport

*Jeu : `gold_large.json` — N=305 (104 mono, 201 multi). Modèle : `mistral-small-latest`, température 0.0 (quasi-déterministe ; ±0.01 de F1 entre runs), JSON mode. Clé via `mistral_client.load_api_key` (jamais loggée).*

## Scorecard — bat-il l'attention réglé-main (F1 0.769) ?

| Approche | F1_multi | Pk↓ | WindowDiff↓ | P | R | mono_FP↓ | F1_global |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Mistral (frontières)** | 0.5124 | 0.2505 | 0.2505 | 0.9688 | 0.3483 | 0.0 | 0.5124 |
| Attention (réglé-main) | 0.769 | 0.1493 | 0.1563 | 0.8955 | 0.6742 | 0.1442 | — |
| Change-point (cosinus) | 0.44 | 0.2815 | 0.282 | 0.4545 | 0.4307 | 0.7019 | 0.384 |

**Verdict frontières : NON** — Mistral F1_multi=0.512 vs attention 0.769 (-0.257).

- **Échec = SOUS-segmentation, pas sur-coupe.** Précision quasi parfaite (**P=0.969** — quand le LLM coupe, il a raison), mais rappel faible (**R=0.348**) : il prédit en moyenne **0.48** coupe/multi contre **1.33** attendues, et ne coupe PAS DU TOUT sur **119/201** multi (59%). Les transitions du gold (« rédigées pour glisser naturellement ») sont vues comme un seul flux cohérent.
- **Abstention mono PARFAITE** : **0%** de faux positifs (0.00 coupe/mono) — strictement mieux que l'attention (14%) et le change-point (70%). Le LLM ne coupe jamais un avis mono-thème cohérent ; c'est le miroir exact de sa sous-segmentation.

## Récupération des THÈMES (le vrai but — l'attention ne sait pas faire)

Multi-label, choix fermé sur 8 thèmes, vs l'ensemble des `seg_themes` du gold (305 avis couverts).

| Granularité | P | R | F1 |
| --- | --- | --- | --- |
| micro | 0.8807 | 0.9808 | 0.928 |
| macro | 0.9008 | 0.9802 | 0.9346 |

**Exactitude d'ENSEMBLE** (tous les thèmes d'un avis, ni plus ni moins) : **73%** des avis.

### F1 par thème

| thème | P | R | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| desinformation | 1.0 | 1.0 | 1.0 | 57 | 0 | 0 |
| harcelement | 0.986 | 0.986 | 0.986 | 72 | 1 | 1 |
| image_corps | 0.958 | 1.0 | 0.979 | 69 | 3 | 0 |
| addiction | 0.899 | 1.0 | 0.947 | 71 | 8 | 0 |
| algorithme | 0.97 | 0.901 | 0.934 | 64 | 2 | 7 |
| contenus_choquants | 0.912 | 0.954 | 0.932 | 62 | 6 | 3 |
| sante_mentale | 0.811 | 1.0 | 0.896 | 99 | 23 | 0 |
| enfants | 0.67 | 1.0 | 0.802 | 67 | 33 | 0 |

## Exemples (avis multi → frontières & thèmes)

**multi-01** — gold thèmes : addiction, sante_mentale

- gold frontières : Je passe beaucoup trop de temps à scroller le soir et je n'arrive pas à lâcher mon téléphone. ⟂ Du coup je dors mal, je suis épuisé et je me sens de plus en plus déprimé au quotidien.
- Mistral frontières : Je passe beaucoup trop de temps à scroller le soir et je n'arrive pas à lâcher mon téléphone. ⟂ Du coup je dors mal, je suis épuisé et je me sens de plus en plus déprimé au quotidien.
- Mistral thèmes : addiction, sante_mentale

**multi-02** — gold thèmes : harcelement, image_corps

- gold frontières : Ma fille subit des moqueries et des commentaires méchants à chaque publication. ⟂ Résultat elle complexe énormément sur son corps et se compare sans arrêt aux filles parfaites qu'elle voit.
- Mistral frontières : Ma fille subit des moqueries et des commentaires méchants à chaque publication. ⟂ Résultat elle complexe énormément sur son corps et se compare sans arrêt aux filles parfaites qu'elle voit.
- Mistral thèmes : harcelement, image_corps

**multi-03** — gold thèmes : algorithme, sante_mentale

- gold frontières : L'algorithme pousse en boucle des vidéos de plus en plus sombres dès qu'on en regarde une. ⟂ À la longue ça entretient le mal-être et ça plonge vraiment dans l'anxiété.
- Mistral frontières : L'algorithme pousse en boucle des vidéos de plus en plus sombres dès qu'on en regarde une. ⟂ À la longue ça entretient le mal-être et ça plonge vraiment dans l'anxiété.
- Mistral thèmes : algorithme, sante_mentale

## Coût, latence, confidentialité — honnêteté

- **Coût d'un run à froid (sans cache)** : **77 appels** `mistral-small-latest`, ~**70s** cumulés (~0.90s/appel), ~**229,502** caractères de prompts envoyés. C'est le coût réel facturé pour évaluer les 305 avis.
- **Ce run** : 0 appels réels + 77 servis par le cache disque `.cache/llm/` (0 erreurs) — le cache rend les relances gratuites et déterministes.
- **Batching** : frontières 6 avis/appel, thèmes 12 avis/appel (réduit le nb d'appels d'un facteur ~6/12).
- **Destinataire** : `api.mistral.ai` (UE).
- **Ce qui part à l'API** : le **texte intégral des avis citoyens** (données potentiellement sensibles) est transmis à Mistral. L'attention et le change-point tournent **100% en local** (aucune donnée ne sort). C'est le compromis central : le LLM est plus capable mais externalise la donnée et coûte par appel.
- **Déterminisme** : température 0.0 + JSON mode ; reproductible modulo variations serveur. Cache → relances stables et gratuites.

## Verdict

- **Frontières : le LLM NE BAT PAS l'attention réglé-main** (0.512 vs 0.769). L'attention locale reste devant et gratuite. Mais l'attention ne produit QUE des frontières.

- **Thèmes : le LLM récupère l'ensemble des thèmes à micro-F1=0.928** (exact-set 73%) — capacité que NI l'attention NI le change-point n'ont. Si le but produit est « quels thèmes dans cet avis », c'est la mesure qui compte, et le LLM la sert directement sans pipeline d'embeddings ni seuils réglés à la main.

- **Compromis** : capacité multi-thème immédiate et langue-agnostique, contre coût/latence par appel et sortie des données vers l'API. Pour de la segmentation de frontières pure et locale, l'attention reste préférable ; pour l'extraction thématique, le LLM est l'outil direct.

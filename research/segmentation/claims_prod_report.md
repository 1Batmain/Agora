# Pipeline CLAIMS en PROD — thèmes ÉMERGENTS d'une vraie consultation

*Source : `ideas.jsonl` — **40 avis citoyens réels** (consultation TikTok, open data Assemblée nationale ; réponses libres FR sur le mal-être / le harcèlement). **AUCUN gold, AUCUNE taxonomie** : les thèmes émergent du bas. Extraction : `ministral-3:latest` (Mac, souverain), pensée coupée. Embed : `nomic-v2`. Clustering k-NN+Leiden, défauts DÉRIVÉS (k=8, seuil=0.535), résolution 1.0 → **5 thèmes**, modularité 0.546.*

⚠️ **Run PARTIEL** (`--limit`) — aperçu, pas la consultation entière.

**154 claims atomiques** extraites (3.85/avis) puis clusterisées. Chaque thème ci-dessous est une préoccupation DÉCOUVERTE — personne ne l'a écrite dans une liste. Tri par **poids social × consensus** (les préoccupations les plus partagées ET cohérentes d'abord). Le poids = somme des poids d'avis (near-dups d'ingest cumulés sur leur représentant).

## Carte des thèmes émergents

| # | thème (c-TF-IDF) | claims | avis | poids | consensus | diversité |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | comportement · sensation · souffre | 43 | 19 | 43.0 | 0.612 | 0.999 |
| 2 | application · temps · sociaux | 39 | 18 | 39.0 | 0.649 | 0.987 |
| 3 | ligne · haine · plateformes | 40 | 16 | 40.0 | 0.604 | 0.999 |
| 4 | instagram · utilisation · reels | 24 | 9 | 24.0 | 0.651 | 0.996 |
| 5 | vidéos · courtes · vidéo | 8 | 4 | 8.0 | 0.649 | 1.0 |

*consensus = cosinus moyen intra-thème (haut = même intention) ; diversité = 1 − densité de quasi-doublons (haut = mêmes idées, formulations variées).*

## Détail — claims représentatives par thème

### 1. comportement · sensation · souffre

*43 claims · 19 avis · poids 43.0 · consensus 0.612 · diversité 0.999*  
mots-clés : _comportement, sensation, souffre, intense, troubles, personnelle_

- Cette pratique peut générer des sentiments de frustration ou d'insatisfaction personnelle
- Elle exprime une impression persistante d’étourdissement physique et mental.
- La personne ressent une dépendance intense à quelque chose.
- Il éprouve un sentiment de perte de maîtrise ou de contrôle sur son utilisation.

### 2. application · temps · sociaux

*39 claims · 18 avis · poids 39.0 · consensus 0.649 · diversité 0.987*  
mots-clés : _application, temps, sociaux, réseaux, difficile, plateformes_

- L’utilisateur ressent une dépendance à l’activité en question.
- L'utilisation excessive de ces plateformes génère un sentiment de perte de temps inutile.
- Ces plateformes créent une dépendance forte, même chez celles et ceux qui ne les trouvent pas agréables.
- Les utilisateurs ressentent une dépendance sans accompagnement pour en sortir.

### 3. ligne · haine · plateformes

*40 claims · 16 avis · poids 40.0 · consensus 0.604 · diversité 0.999*  
mots-clés : _ligne, haine, plateformes, mineurs, contenu, individus_

- Les contenus en ligne peuvent propager de la haine envers certains groupes spécifiques comme les Juifs
- Certains contenus visent à ridiculiser gratuitement des individus.
- La personne concernée a subi des allusions ou des contenus problématiques dans ses publications en ligne.
- Les échanges sur la plateforme incluent souvent des propos violents ou haineux.

### 4. instagram · utilisation · reels

*24 claims · 9 avis · poids 24.0 · consensus 0.651 · diversité 0.996*  
mots-clés : _instagram, utilisation, reels, utilisateurs, contenu, plateformes_

- L’utilisation de TikTok peut créer une dépendance très rapide.
- TikTok contribue à induire un état de mal-être chez ses utilisateurs via son algorithme addictif.
- Les échanges avec des personnes habituées à TikTok deviennent plus compliqués en raison de cette dépendance aux informations ou opinions diffusées sur la plateforme.
- L’utilisation de TikTok a déclenché des troubles du bien-être chez une adolescente.

### 5. vidéos · courtes · vidéo

*8 claims · 4 avis · poids 8.0 · consensus 0.649 · diversité 1.0*  
mots-clés : _vidéos, courtes, vidéo, répétée, résolution, meurtre_

- Les vidéos disponibles peuvent contribuer à une détérioration du bien-être mental des individus.
- Une utilisation intensive de vidéos courtes peut créer une dépendance, comme l’a été mon expérience au lycée.
- Les vidéos courtes ne procurent aucune valeur éducative ou informative notable.
- Certaines vidéos encouragent ou normalisent les comportements autodestructeurs comme les scarifications.

## Coût & souveraineté

- **Extraction ministral (Mac)** : 40 appels réels + 0 cache, ~98s (~2444 ms/avis), 3,798 tokens, 0 erreurs. Embed + clustering : local, négligeable.
- **Souverain** : la donnée citoyenne ne quitte jamais le réseau privé (`http://mac-local:11434`, Tailscale). Coût marginal ~0 € (vs ~2-4 €/consultation en API).

## Lecture

- **Les thèmes ont émergé sans aucune taxonomie.** Sur le banc gold (8 thèmes connus), claims→cluster reconstruisait une bijection 8↔8 à micro-F1 0.784 ; ICI, sans gold, il produit directement la carte des 5 préoccupations de la consultation — c'est le mode d'emploi réel.

- **Granularité réglable** : la résolution Leiden fixe le nombre de thèmes (basse = quelques grands thèmes, haute = sous-facettes fines). Aucun nombre n'est imposé : on choisit la lentille selon l'usage (synthèse vs exploration fine).

- **Tri par poids × consensus** : remonte ce que BEAUCOUP de citoyens disent de façon COHÉRENTE — le signal d'opinion partagée, pas l'anecdote isolée. La diversité distingue « 100 personnes, 100 formulations » d'un copier-coller viral.

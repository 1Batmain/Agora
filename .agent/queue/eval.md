# Lane eval — banc de mesure (eval-as-truth)

Owns: `eval/`. Arbitre les choix par la mesure, pas l'intuition (Playbook §5).

## T-E1 · Banc Leiden vs HDBSCAN sur x-stance
- Goal : x-stance porte des labels FAVOR/AGAINST par question → scorer la qualité
  de clustering (NMI / ARI / purity vs labels) + silhouette intra-modèle.
- Accept : scorecard par approche, par question.
- Deps : nlp T-N3 (sur batch statique).

## T-E2 · Stabilité (bootstrap)
- Goal : N runs Leiden/HDBSCAN sur ré-échantillons → accord inter-runs (clusters
  qui survivent). Anticipe "clusters qui changent selon paramètres".
- Accept : indice de stabilité par approche.
- Deps : T-E1.

## T-E3 · Coût
- Goal : latence + nb d'appels embeddings (+ LLM si naming) par approche.
- Accept : coût mesuré à côté de l'accuracy (décision = nombre).
- Deps : T-E1.

## T-E4 · Scorecard
- Goal : rapport unique accuracy × stabilité × coût ; honnêteté (taille échantillon,
  non-couvert). Mesure CHAQUE changement futur.
- Accept : `eval/report.md` régénérable.
- Deps : T-E1..3.

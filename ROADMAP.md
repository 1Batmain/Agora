# Agora — Roadmap

> Cap fixé collectivement (juillet 2026). Trois phases séquentielles : **consolider →
> approfondir → ouvrir**. Chaque phase livre de la valeur utilisable ; on ne passe à la
> suivante que quand la précédente tient. Les contributions sont bienvenues sur toutes
> les phases — voir [CONTRIBUTING.md](CONTRIBUTING.md).

## Phase 1 — Optimiser l'existant (UX) 🎨
*Objectif : une expérience irréprochable sur ce qu'Agora sait déjà faire.*

- [ ] **Landing page** : refonte visuelle ambitieuse — l'entrée doit donner envie
      (direction artistique, animations sobres, storytelling du pipeline)
- [ ] **Page explorateur** : ergonomie des filtres (thèmes/sentiment/recherche),
      lisibilité des surlignages, carte de stats de claim, pagination fluide
- [ ] Activer la **recherche instantanée** (moteur DuckDB déjà mergé — générer les
      `.duckdb` par dataset au build)
- [ ] Cohérence visuelle générale (typographie, densité, mobile)
- [ ] Accessibilité : premier passage RGAA sur les parcours clés

## Phase 2 — Approfondir l'analyse 🔬
*Objectif : la meilleure lecture d'opinion possible sur les consultations existantes
(datasets Assemblée nationale). Chaque amélioration validée par la mesure (gold ou
panel aveugle) avant d'être servie.*

- [ ] **Stance analysis** : améliorer la classification (le banc gold x-stance et le
      protocole de validation existent — accuracy servie : 0,79)
- [ ] **Argument mining** : refonte pour respecter l'invariant verbatim (extraits
      exacts, traçables) — puis intégration UI (arguments pour/contre par thème)
- [ ] **Stabilité d'échantillonnage** : mesurer la sensibilité des thèmes au re-tirage
      (la preuve de robustesse qui manque)
- [ ] **Vue analyste** (human-in-the-loop) : renommer/fusionner/corriger les thèmes,
      à partir des signalements
- [ ] Export d'un **rapport de synthèse** (HTML) par consultation

## Phase 3 — Ouvrir la plateforme 🌍
*Objectif : Agora devient une place d'expression — on y OUVRE des consultations, on
collecte, on analyse en continu.*

- [ ] **Ouverture de consultations** : création d'une consultation (question, période,
      contexte) depuis l'interface
- [ ] **Collecte de témoignages** : soumission citoyenne robuste (la brique `/submit`
      existe en mode collecte)
- [ ] **Anonymisation RGPD by-design** : PII masquée À L'INGESTION (regex + NER —
      la brique existe en recherche), registre des traitements, droit à l'effacement
- [ ] **Synthèse vivante** : analyse incrémentale pendant que la consultation est
      ouverte (thèmes qui émergent au fil de l'eau)
- [ ] **Synthèse officielle de clôture** : rapport final figé, daté, traçable —
      le livrable institutionnel
- [ ] Montée en charge (l'analyse du Grand Débat complet — 22 174 avis pour ~25 $ —
      donne l'ordre de grandeur : viser 100k+)

## Principes transverses (non négociables, toutes phases)
**Verbatim** (extraits exacts, zéro paraphrase) · **Traçabilité** (du thème à la phrase
du citoyen) · **Généricité** (zéro corpus en dur) · **Honnêteté** (limites affichées,
métriques non trompeuses, « ceci n'est pas un sondage ») · **Validation avant adoption**
(gold, panel aveugle, verdict écrit).

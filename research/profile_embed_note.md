# Verdict — moteur B : embedder un PROFIL de thème, pas une étiquette courte (2026-07-18)

**Contexte.** Moteur d'abstraction B (le retenu) : couche plate → profil LLM par thème →
ré-embedding → clustering = macros. Question de tuning (Bob) : QUOI embedder ? étiquette courte
vs profil complet, sous la contrainte des 512 tokens (fenêtre nomic-v2).

## Mesure (tiktok, 9 thèmes plats)

| ce qu'on embedde | cosinus des paires proches | effet |
|---|---|---|
| étiquette canonique courte (3-6 mots) | **+1.00** (identiques !) | SUR-collapse : « faire »/« vidéos »/« algorithme » reçoivent la MÊME étiquette (« Réseaux sociaux ») → fusion aveugle, **précision perdue** |
| **profil** (3-5 phrases, sujet canonique + angles, ≤300 mots) | +0.66 max, étalé | **distinction préservée** — même sujet proche, mais pas écrasé |

→ **Le profil est le bon compromis** : canonique dans son ouverture (les redondants se
rejoignent) mais fidèle dans le détail (la précision monte jusqu'aux couches abstraites). Câblé
dans `pipeline/cluster/abstraction.py::_profile`.

## Réserve — le vrai juge est un corpus MULTI-thèmes

Sur 9 thèmes (tiktok, mono-sujet), le clustering des profils (`flat_partition`, pic de
modularité) **dégénère** au grossier (2-4 macros) — trop peu de points pour une géométrie
propre, et pas de vraie structure macro à trouver (tout parle de TikTok). Le moteur B ne peut
se valider que sur un corpus à DIZAINES de thèmes et VRAIS domaines distincts → **Grand Débat
complet** (4 thèmes officiels). C'est le prochain test.

## Levier restant à tuner
Le CONTEXTE feed au LLM pour un profil fidèle (nombre/diversité des claims représentatifs,
structure du profil). Repro : `research/profile_embed_test.py`.

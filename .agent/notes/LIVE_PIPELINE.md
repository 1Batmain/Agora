# Pipeline LIVE — ce qui est construit, ce qui est mesuré, ce qui reste faux

**2026-08-04/05.** Implémentation du scénario live (`LIVE_SCENARIO.md`) : rejouer une séance
de l'Assemblée nationale et construire une synthèse au fil de l'eau. Branche
`feat/pipeline-profile`, **non mergée** (chantier expérimental).

---

## 1. Ce qui existe

| Module | Rôle | LLM ? |
|---|---|:--:|
| `live/transcript.py` | comptes rendus XML → tours nommés/horodatés | non |
| `live/state.py` | thèmes gelés, assignation, dérive, positions par orateur | non |
| `live/pipeline.py` | amorçage + traitement d'un tour | oui |
| `live/replay.py` | horloge de rejeu, instantanés | — |
| `live/server.py` + `page.html` | affichage (lit les instantanés, ne calcule rien) | non |
| `pipeline/stance.py` | stance + objet de clivage, **partagé** batch/live | oui |
| `pipeline/embed/vector_store.py` | cache d'embeddings adressé par contenu | non |
| `pipeline/profile.py` | quels modèles, quelles couches | — |

**Isolation vérifiée par test** : aucun module de `live/` n'importe `backend/`. Le pipeline
live compose des couches de `pipeline/` ; il ne peut pas casser l'Agora servi. Cache dédié
(`var/live-cache/`), serveur séparé (port distinct), profil `live` avec son propre namespace.

## 2. L'architecture, en une phrase

> **Amorcer une fois, geler les thèmes, puis assigner au lieu de repartitionner.**

Assigner (plus proche centroïde) contourne le problème dur — garder l'identité des thèmes à
travers une repartition — au lieu de le résoudre. La carte ne bouge pas sous les yeux du
spectateur. Le prix est la **dérive**, qu'on mesure au lieu de la subir.

Le seuil de couverture est **dérivé** de la distribution des similarités intra-thème de
l'amorçage (percentile bas), jamais fixé en dur : un littéral se périme silencieusement à la
première bascule d'embedder — vécu avec le seuil de corrélation des contributions.

## 3. Ce que le réel a corrigé

Trois choses que la conception sur papier avait fausses. Toutes découvertes en exécutant, pas
en réfléchissant.

### 3.1 ⚠️ Le budget par tour était sous-estimé d'un ordre de grandeur

J'avais budgété **1,45 claim par tour** — le ratio mesuré sur les contributions citoyennes.
Sur l'amorçage réel : **16,9 claims par tour**. La raison est évidente après coup : un avis
citoyen fait quelques centaines de caractères, un tour de parole parlementaire fait 1 121
caractères de médiane et jusqu'à **20 655**. Le nombre de claims suit la LONGUEUR, pas le
nombre de tours.

Conséquence : le coût par tour est ~10× ce que j'annonçais. La marge sur le temps réel existe
toujours (un tour toutes les ~50 s), mais elle est de l'ordre de 1–2×, pas de 10×.
**Le chiffre de `LIVE_SCENARIO.md` §2 est à corriger.**

### 3.2 Les tours longs devaient être découpés

`extract_claims` groupe 8 avis par appel. Sans découpage, un lot d'amorçage dépassait
**100 000 caractères d'entrée** pour un plafond de 3 200 tokens de sortie : lenteur extrême et
troncature silencieuse du JSON. Les tours sont donc fragmentés à ≤ 1 800 c **aux frontières de
phrase**, puis les claims sont recollés sous l'id du tour. Sans risque pour l'invariant
verbatim (l'ancrage est PAR AVIS, donc un fragment est un contexte d'ancrage valide).

### 3.3 ⚠️ Un taux de dérive brut ne s'interprète pas

Premier rejeu : **40 % de claims non couverts**. Lu brut, ça réclame une restructuration.
En regardant les claims concernés :

> « il est fort de café de nous intenter un procès en comédie »
> « le sous-amendement no 42160 à remplacer "annuelle" par "chaque année" »
> « vous crachez aujourd'hui au visage des 8 000 manifestants »

Ce n'est pas un sujet nouveau : c'est le **théâtre parlementaire** — exactement les ~50 % de
claims hors-fond identifiés à la sonde (`LIVE_PROBE_CLAIMS.md`). La dérive expose donc
maintenant un `baseline` (le bruit de fond attendu par construction du seuil) et un `excess`.
**Seul l'excès porte un signal de dérive thématique.**

## 4. Ce qui reste FAUX ou non validé — à ne pas oublier

| Point | État |
|---|---|
| **Discours rapporté** | garde-fou de prompt ÉCRIT (`REPORTED_SPEECH_GUARD`) mais **non validé par un banc**. Un orateur qui cite l'adversaire pour le réfuter peut encore voir sa position inversée. **Rien ne doit être publié comme « position de X » sur cette base.** |
| **Profil `live`** | `validated=False`. Seul `extract` (ministral-3b) repose sur une mesure ; `opinion`, `enrich`… sont des choix par analogie de taille, non benchés. |
| **Qualité des thèmes d'amorçage** | γ=3.0 sur 338 claims donne 21 thèmes, dont plusieurs purement procéduraux (« obstruction · honte · volonté »). Le pré-filtre de pertinence n'est **pas** appliqué en amont de l'amorçage — il devrait l'être. |
| **Objets de clivage** | certains sont bons (« abroger la réforme des retraites de 2023 »), d'autres absurdes parce que dérivés d'un thème procédural (« organiser l'obstruction à la volonté du peuple »). Conséquence directe du point ci-dessus. |
| **Amorçage à froid** | fait sur les N premiers tours de la même séance. L'option plus réaliste (amorcer sur une séance ANTÉRIEURE du même texte) n'est pas implémentée. |
| **Pas de restructuration** | l'indicateur de dérive existe ; l'action qu'il devrait déclencher n'est pas écrite. |

## 5. Lancer

```bash
# 1. corpus (Licence ouverte, ~54 Mo) → data/raw/an/ (gitignoré)
curl -sL -o /tmp/an.zip \
  https://data.assemblee-nationale.fr/static/openData/repository/17/vp/syceronbrut/syseron.xml.zip
unzip -j /tmp/an.zip 'xml/compteRendu/*' -d data/raw/an/

# 2. rejeu (×60 = 1 min de séance par seconde ; --speed 0 = aussi vite que possible)
MISTRAL_API_KEY=$(cat var/mistral.key) uv run --extra contender --extra embed-contender \
  --extra faiss python -u -m live.replay data/raw/an/CRSANR5L17S2025O1N055.xml \
  --speed 60 --bootstrap 30

# 3. page (dans un autre terminal) → http://127.0.0.1:8020
uv run --extra serve uvicorn live.server:app --port 8020
```

`--no-stance` fait un essai à blanc (assignation seule, aucun appel de position).

## 6. Prochaines marches

1. **Pré-filtre de pertinence AVANT l'amorçage** — c'est ce qui empoisonne le plus la sortie
   aujourd'hui (thèmes et clivages procéduraux).
2. **Bencher le garde-fou discours rapporté** — sans quoi l'axe « position par orateur »,
   qui est l'intérêt du cas AN, reste inutilisable.
3. **Amorcer sur une séance antérieure** du même texte (démarrage à froid réaliste).
4. **Déclencher une restructuration** sur `excess`, et traiter alors l'appariement des thèmes
   avant/après — le problème contourné jusqu'ici.

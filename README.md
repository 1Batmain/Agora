# Agora — outil d'analyse des consultations citoyennes


[![Démo live](https://img.shields.io/badge/D%C3%A9mo_live-forge.tail0b8aa8.ts.net-2ea44f?style=for-the-badge)](https://forge.tail0b8aa8.ts.net)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/Front-React_+_Vite-61dafb?logo=react&logoColor=black)
![Souverain](https://img.shields.io/badge/embeddings-100%25_local-6f42c1)
![PR-only](https://img.shields.io/badge/main-prot%C3%A9g%C3%A9e_(PR)-orange)

> ### 🚀 Démo publique en ligne : **<https://forge.tail0b8aa8.ts.net>**
> Explorez une vraie consultation — la carte des thèmes, les synthèses, et chaque
> témoignage citoyen avec ses extraits surlignés. *(serveur de test, mode public en lecture seule)*

---

## Le problème

Quand des dizaines de milliers de citoyens répondent à une consultation, personne ne lit
tout. On résume — et en résumant, on **trahit** : on reformule, on lisse, on choisit
d'avance les cases. Le citoyen ne se reconnaît plus dans la synthèse.

**Agora fait l'inverse.** Les thèmes ne sont pas imposés : ils **émergent** des
contributions elles-mêmes. Rien n'est reformulé — chaque affirmation reste le **verbatim
exact** de la personne, traçable jusqu'à son témoignage d'origine. Et tout tourne **en
local** : les embeddings ne quittent jamais la machine (souveraineté des données).

Trois invariants tenus de bout en bout :

| | |
|---|---|
| 🔍 **Fidèle** | zéro reformulation — les extraits sont des sous-chaînes exactes du texte citoyen |
| 🧭 **Traçable** | chaque thème → ses claims → l'avis complet, surligné, d'où ils viennent |
| 🔒 **Souverain** | embeddings `nomic` locaux, aucun texte citoyen envoyé à un cloud tiers |

---

## Le pipeline, en clair

Du texte brut à la carte des opinions, sans jamais perdre le fil vers la source :

```
Contributions citoyennes
        │
        ▼
 ① CLAIMS      extraction VERBATIM (multi-spans) — la question de la consultation
               sert de cadre, aucune paraphrase ; chaque claim = un morceau exact du texte
        │
        ▼
 ② EMBEDDINGS  chaque claim vectorisé en LOCAL (nomic-embed) — souverain, hors-ligne
        │
        ▼
 ③ GRAPHE k-NN chaque claim relié à ses plus proches voisins (cosinus)
        │
        ▼
 ④ LEIDEN      détection de communautés → les THÈMES émergent (aucune taxo imposée)
        │
        ▼
 ⑤ ENRICH      titres + synthèses Markdown des thèmes (LLM, sur le cluster — pas sur le citoyen)
        │
        ▼
 ⑥ OPINION     objet de clivage + stance (pour / contre / nuancé) + confiance calibrée
        │
        ▼
 Carte navigable : thèmes → sous-thèmes → témoignages surlignés
```

Le front rend la **carte des thèmes** (bulles d3-pack), un **explorateur de témoignages**
(chaque avis inline, extraits colorés à la couleur de leur thème), les **synthèses** par
niveau, et la **répartition d'opinion** par thème.

---

## Les datasets

Agora est **générique** : rien n'est codé en dur pour un corpus. Il tourne tel quel sur
plusieurs consultations de référence, servies depuis leur cache précalculé :

| Dataset | Contenu |
|---|---|
| **tiktok** | Consultation citoyenne FR sur les réseaux sociaux / TikTok |
| **granddebat** | Grand Débat National 2019 (axe Démocratie & citoyenneté) |
| **republique-numerique** | Consultation « République numérique » |
| **xstance** | Corpus multilingue DE/FR/IT — **12 enfants** par thématique (santé, économie, éducation…) |

> La consultation **x-stance** illustre la hiérarchie *mère → enfants* : une consultation
> peut se découper en sous-consultations, chacune servie par son id.

---

## Installation & lancement en local

**En une commande** (recommandé) :
```bash
git clone git@github.com:1Batmain/Analyse-des-consultations-citoyennes.git
cd Analyse-des-consultations-citoyennes
./scripts/setup.sh     # deps back+front · caches d'analyse (release GitHub) · secrets locaux
make dev               # backend :8010 + front :5180 → http://localhost:5180
```
Prérequis : [`uv`](https://docs.astral.sh/uv/) (Python 3.11) et Node 18+. La clé Mistral est
**optionnelle** — le front et la **lecture des caches** marchent sans ; elle ne sert qu'à
*construire* de nouvelles analyses. Workflow de contribution : [`CONTRIBUTING.md`](CONTRIBUTING.md).

<details><summary><b>Lancement manuel</b> (sans <code>make</code>)</summary>

Backend — API FastAPI sur `:8010`, sert **uniquement le cache précalculé** (`/datasets`,
`/analysis`, `/insights`, `/citations`, `/opinion`, `/avis_list`, `/cost`) — aucun calcul
lourd à la requête :
```bash
uv run --extra contender --extra embed-contender --extra faiss --extra serve \
  uvicorn backend.server:app --host 0.0.0.0 --port 8010
```
Frontend — Vite sur `:5180` (proxifie `/api/*` vers `:8010`) :
```bash
cd frontend && npm install && npm run dev
```
</details>

### Mode public (exposition Internet) — *fail-closed*
Avant d'exposer le serveur, activez le durcissement via **trois variables d'env** :

```bash
export AGORA_PUBLIC=1        # inverse la posture : endpoints coûteux/mutants REFUSÉS sans token
export AGORA_API_TOKEN="…"   # jeton requis pour tout endpoint protégé (comparaison à temps constant)
export AGORA_HASH_SALT="$(python -c 'import secrets; print(secrets.token_hex(32))')"  # sel d'anonymisation (≥32)
```

En mode public : les endpoints de **compute/build** (`/recluster`, `/density`, `/build`)
et les **mutations** (`/submit`, `/flag`) renvoient `403`, et **aucun build n'est jamais
déclenché** par une lecture (zéro extraction LLM en prod publique). Seules les lectures de
cache restent ouvertes.

---

## Tests

Suite de régression + **intégration bout-à-bout** in-process (TestClient, sans réseau ni
build LLM). Les tests qui exigent une analyse précalculée **se skippent proprement** si le
cache d'un dataset est absent — jamais d'échec parasite, jamais de build déclenché.

```bash
make test    # ou : uv run --extra embed-contender --extra faiss --extra serve --with pytest pytest -q
```

`backend/tests/test_integration.py` suit les vrais parcours : landing (`/datasets`),
exploration d'une consultation prête (`/analysis` → thème réel → `/insights` /
`/citations` / `/avis_list` / `/opinion`), et la posture **mode public fail-closed**.

---

## Collaborer

`main` est **protégée** : tout passe par **Pull Request**, jamais de push direct.

- **Une PR par lot de travail**, revue avant merge.
- **Invariants non négociables** — à ne jamais casser dans une contribution :
  - **verbatim** : les extraits restent des sous-chaînes exactes du texte citoyen (pas de paraphrase) ;
  - **souverain** : les embeddings restent locaux, aucun texte citoyen vers un cloud tiers ;
  - **générique** : rien de spécifique à un corpus en dur — tout dérivé des données.
- **Contexte agent centralisé** dans [`.agent/`](.agent/README.md) : onboarding, conventions,
  notes de R&D et ledger des tâches — tout agent (ou humain) y trouve ses repères d'emblée.

---

<sub>Agora — analyse fidèle, traçable et souveraine des consultations citoyennes. Démo : <https://forge.tail0b8aa8.ts.net></sub>

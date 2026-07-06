# Agora

[![Démo live](https://img.shields.io/badge/D%C3%A9mo_live-forge.tail0b8aa8.ts.net-2ea44f?style=for-the-badge)](https://forge.tail0b8aa8.ts.net)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/Front-React_+_Vite-61dafb?logo=react&logoColor=black)
![Souverain](https://img.shields.io/badge/embeddings-100%25_local-6f42c1)
![PR-only](https://img.shields.io/badge/main-prot%C3%A9g%C3%A9e_(PR)-orange)

## En bref

**Agora veut améliorer les échanges entre les citoyens et les acteurs de la démocratie —
et restaurer un lien de confiance.**

Quand des milliers de personnes répondent à une consultation publique, leur parole mérite
mieux qu'un résumé qui lisse et reformule. Agora analyse l'intégralité des contributions
et en fait **émerger les grands thèmes** — automatiquement, rapidement, à coût marginal —
sans jamais trahir la parole d'origine :

| | |
|---|---|
| 🔍 **Fidèle** | zéro reformulation — chaque extrait est une sous-chaîne exacte du texte citoyen |
| 🧭 **Traçable** | chaque thème → ses idées → le témoignage complet, surligné, d'où elles viennent |
| 🔒 **Souverain** | embeddings locaux, modèles ouverts — portable en local de bout en bout |
| ⚖️ **Honnête** | limites affichées (échantillons, incertitudes) — « ceci n'est pas un sondage » |

L'ambition, phase après phase ([ROADMAP](ROADMAP.md)) : des consultations mieux restituées
aujourd'hui, une **place d'expression** demain — où l'on ouvre une consultation, où chacun
témoigne, et où la synthèse vit sous les yeux de tous.

👉 **Démo en ligne : <https://forge.tail0b8aa8.ts.net>**

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
 ⑤ ENRICH      titres + synthèses des thèmes (LLM, sur le cluster — pas sur le citoyen)
        │
        ▼
 ⑥ OPINION     objet de clivage + sentiment (positif / négatif / neutre) + confiance
        │
        ▼
 Carte navigable : thèmes → sous-thèmes → témoignages surlignés
```

## Collaborer

Le projet est ouvert aux contributions — `main` est protégée, tout passe par Pull Request.

```bash
git clone git@github.com:1Batmain/Agora.git
cd Agora
./scripts/setup.sh     # deps back+front · caches d'analyse · secrets locaux
make dev               # backend :8010 + front :5180 → http://localhost:5180
```

- Workflow, prérequis et repères : [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Feuille de route et chantiers ouverts : [`ROADMAP.md`](ROADMAP.md)
- Contexte pour agents et humains : [`.agent/README.md`](.agent/README.md)
- **Invariants non négociables** : verbatim (extraits exacts), souveraineté (embeddings
  locaux), généricité (zéro corpus en dur), honnêteté (limites affichées).

---

<sub>Agora — analyse fidèle, traçable et souveraine des consultations citoyennes.</sub>

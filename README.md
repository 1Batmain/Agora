# Agora — retranscrire la parole citoyenne

> ### 🚀 Démo publique en ligne : **<https://forge.tail0b8aa8.ts.net>**
> Testez la méthode Agora sur des témoignages publics recueillis lors de différentes consultations citoyennes.

---

## Améliorer les échanges entre les citoyens et les acteurs de la démocratie

- **En simplifiant le recours à l'avis des citoyens**, grâce à l'automatisation complète de la phase d'analyse des témoignages recueillis ;
- **En permettant de naviguer dans de larges recueils de témoignages** par thématiques, et de développer chacune d'elles jusqu'au verbatim exact qui la compose.

---

**Quatre engagements, tenus de bout en bout :**

| | |
|---|---|
| 🔍 **Fidèle** | zéro reformulation — chaque extrait est une sous-chaîne exacte du texte citoyen |
| 🧭 **Traçable** | chaque thème émane des idées extraites du témoignage complet |
| 🔒 **Souverain** | embeddings locaux, modèles ouverts  |

L'ambition, phase après phase ([ROADMAP](ROADMAP.md)) : des consultations mieux restituées
aujourd'hui, une **place d'expression** demain — où l'on ouvre une consultation, où chacun
témoigne, et où la synthèse vit sous les yeux de tous.

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

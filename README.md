# Agora - retranscrire la parole citoyenne

> ###  Démo publique en ligne : **<https://forge.tail0b8aa8.ts.net>**
> Testez la méthode agora sur des témoignanges publics receuillis lors de différentes consultations citoyennes

---

**Agora propose un moyen d'améliorer les échanges entre les citoyens et les différents acteurs de la démocratie**
-> En simplifiant le recours à l'avis des citoyens grâce à l'automatisant complète de la phase d'analyse des témoignages receuillis.
-> En permetant de naviguer dans de larges recceuils de témoignanges par thématiques, et de développer chacune d'elles jusqau'au verbatim exact qui les compose.

---

| | |
|---|---|
| 🔍 **Fidèle** | zéro reformulation — chaque extrait est une sous-chaîne exacte du texte citoyen |
| 🧭 **Traçable** | chaque thème → ses idées → le témoignage complet, surligné, d'où elles viennent |
| 🔒 **Souverain** | embeddings locaux, modèles ouverts — portable en local de bout en bout |
| ⚖️ **Honnête** | limites affichées (échantillons, incertitudes) — « ceci n'est pas un sondage » |

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

# Émergence d'arguments par densité — proto R&D sur corpus fourni (2026-07-07)

**Statut : RESEARCH, zéro modif pipeline.** Teste l'idée de Bob : les arguments doivent
ÉMERGER de la densité de la donnée (même système que les thèmes, un cran plus fin), au lieu
d'être sélectionnés par un LLM (biais « meilleur d'un mauvais lot »). Corpus **republique-
numerique** (3000 avis → **4175 claims**, 162 feuilles) — buildé en DEV (`emerge_build.py`,
extraction ~2 h, isolé, zéro touche prod). Émergence : `emerge_proto.py`.

## 1. Le bon primitif = celui des THÈMES (kNN cosinus + Leiden), pas HDBSCAN

- **HDBSCAN euclidien sur embeddings 768-dim → 66 % de bruit** + 0 feuille mono-cluster : un
  ARTEFACT de la densité en haute dimension, PAS un verdict. Écarté.
- **`_subdivide` (le primitif de subdivision des thèmes)** = kNN cosinus + Leiden, PARTITION
  propre (0 bruit), communautés cohésives par construction. C'est littéralement « le même
  système que les thématiques », appliqué à l'intérieur d'une feuille. → **à utiliser**.

## 2. Mécaniquement, l'émergence MARCHE

162 feuilles → **388 communautés d'arguments**, toutes cohésives (dispersion 0.0-0.21,
médiane 0.15 ; tailles 5-29, médiane 9). 122 feuilles se subdivisent (≥2), 40 homogènes. Zéro
bruit. Là où le corpus EST argumentatif, des arguments distincts émergent proprement — ex.
**n80 (code source)** : « d'accord mais seulement si libre et non open source » / « l'accès au
code source, oui » / « comment garantir que le code publié est celui déployé ».

## 3. Mais « dense » ≠ « argument substantiel » — trois régimes observés

L'émergence surface des amas denses, dont la NATURE dépend du corpus. Sur republique-numerique
(consultation d'**édition d'un TEXTE DE LOI**, article par article — pas de l'opinion-clivage) :

- **(a) vrais arguments** distincts (n80 code source, n43 handicap) — l'idée tient ;
- **(b) méta-commentaires éditoriaux** sur le texte (n37 : « hors sujet », « pas assez
  développé », « proposition concise et claire, merci ! ») — denses mais PAS des arguments ;
- **(c) sur-découpes de consensus** (n154 transparence : 2 communautés disant toutes deux
  « la transparence est importante ») — le primitif thème FORCE ≥2 (monte la résolution) même
  quand la feuille est homogène → faux clivage.

## 4. Verdict

- **Émergence par densité (primitif thème) : validée mécaniquement.** Propre, non biaisée,
  fail-closed en principe. Corrige V-SELECT (plus de « meilleur d'un mauvais lot »).
- **Deux manques à combler** avant d'en faire un livrable :
  1. **Substance** : un amas dense peut être du méta-commentaire. La densité ne distingue pas
     « argument » de « bavardage cohésif » → il FAUT un filtre de substance (un jugement LLM
     LÉGER par communauté : « est-ce un argument sur le fond ? », sans rien rédiger — garde
     l'anti-biais car la communauté a émergé de la donnée AVANT le LLM).
  2. **Sur-découpe** : ne pas forcer ≥2 (garder `None`/consensus plus souvent) — abaisser la
     résolution argument ou exiger une vraie SÉPARATION inter-communautés, pas juste une
     partition possible.
- **Affichage** : le médoïde est souvent fade/générique (n2, n35) → prendre le membre le plus
  NET du cluster (ou LLM choisit un membre — non biaisé, le cluster préexiste), pas le médoïde.
- **Cible bottom-up** : les embeddings **intriquent sujet et stance** — les communautés sont
  des sous-sujets/nuances, PAS des pôles pour/contre nets (n80 : conditions/variations, pas
  oui-vs-non). Lire une cible débattable directement sur l'axe des clusters n'est **pas** fiable
  → un jugement de stance reste nécessaire pour la polarité.

## 5. Réserve corpus — la plus importante

**republique-numerique est un mauvais banc pour le clivage** : c'est de l'édition législative
collaborative, pas de l'opinion. D'où la forte proportion de méta-commentaires. Le vrai test de
l'émergence d'arguments pour/contre demande un corpus **opinion-clivage** : **tiktok** (1621,
débat addiction, ~1 h de build) ou **granddebat** (22174 — mais ~15 h d'extraction à ce rythme,
hors session). Recommandation : rejouer l'émergence sur **tiktok** avant de conclure.

## Artefacts

`emerge_build.py` (build fondation, dumpe `emerge_cache/<ds>/`) · `emerge_proto.py` (émergence) ·
`emerge_proto_republique-numerique.json`.

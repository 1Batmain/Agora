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

## 6. Test sur corpus opinion-clivage (tiktok) + raffinements — LE verdict

Buildé **tiktok** (1604 avis → 2361 claims, 134 feuilles) en DEV. Ajouté (`emerge_refined.py`) :
**fusion des sur-découpes** (centroïdes ≥0.85) + **filtre de substance LLM** (is_argument +
membre le plus net, sans rédiger) + **affichage membre-net** (pas le médoïde).

Résultats qui tranchent :

- **Fusion 0.85 → tiktok s'effondre à 0 clivage (1 argument/feuille).** Le pour et le contre
  d'un thème partagent le vocabulaire → centroïdes proches → fusionnés. La fusion, censée réparer
  la sur-découpe, DÉTRUIT le clivage réel.
- **Sans fusion → 237 « arguments », mais ce sont des NUANCES d'un même point**, pas des
  arguments distincts (n1 : 4 communautés = « mal-être / tristesse / addiction / vide » = UN
  argument éclaté). Et **le pour/contre ne se sépare pas** (n16 : « tiktok a de bons côtés » sort
  comme 1 amas parmi des négatifs, pas un pôle net).
- **Filtre de substance : keep tiktok 25 % vs repnum 5 %** — il discrimine DIRECTIONNELLEMENT
  (opinion > législatif), mais les décisions sont **incohérentes à la marge** : il garde
  « mal-être dû au contenu triste » et jette « mal-être dû au temps passé » et « harcèlement dans
  les commentaires » — arguments équivalents. Le 25 % n'est pas « 25 % d'arguments », c'est du
  bruit de filtre.

### Verdict FINAL sur l'émergence

**L'émergence bottom-up des arguments (et *a fortiori* de la cible) ne fonctionne pas, pour une
raison FONDAMENTALE, pas un défaut de réglage :**

1. **Les embeddings encodent le SUJET, pas la STANCE ni la structure argumentative.** Prouvé :
   les clusters sont des sous-facettes de sujet ; pour/contre ne se sépare jamais ; fusionner
   ou non ne donne ni arguments distincts ni pôles.
2. **« Argument » est mal défini SANS cible.** Sur tiktok (témoignages de vécu) comme sur repnum
   (édition de loi), il n'y a pas de proposition débattable préalable → « quel témoignage est un
   argument pour/contre » est arbitraire, et le filtre LLM comme le clustering flanchent pour
   CETTE raison.

→ **L'ordre du pipeline ACTUEL est le bon** : dériver la cible D'ABORD (une proposition
débattable), PUIS classer la stance pour/contre, PUIS miner les arguments. L'émergence ne peut pas
bootstrapper la cible car le signal (débat/stance) n'est pas dans les embeddings. L'idée de Bob
était une hypothèse légitime ; **la mesure la réfute proprement** — et confirme que le vrai levier
est le **raffinement VERBATIM À L'INTÉRIEUR de cet ordre** (V-SELECT / V-EXTRACT,
`argmine_verbatim_note.md` / `argmine_extract_note.md`), pas le remplacement de l'ordre.

Corollaire réutilisable : tester la valeur d'un corpus AVANT d'y miner des arguments — pas de
cible dérivable / trop peu d'arguments substantiels → « positions pas assez développées » (le
garde-fou front de Bob, qui reste juste).

## Artefacts

`emerge_build.py` (build fondation) · `emerge_proto.py` (émergence brute) ·
`emerge_refined.py` (fusion + filtre substance) · `emerge_proto_republique-numerique.json` ·
`emerge_refined_{republique-numerique,tiktok}*.json`.

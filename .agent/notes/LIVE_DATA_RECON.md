# Reconnaissance — comptes rendus AN (corpus du scénario live)

**Fait le 2026-08-04.** Corpus téléchargé et mesuré **en entier** (601 séances). Objectif :
savoir ce qu'on a vraiment par tour de parole avant de concevoir le pipeline live.
Verdict : **le corpus est nettement meilleur que ce qu'on avait supposé** — deux des risques
identifiés dans `LIVE_SCENARIO.md` sont annulés par les données elles-mêmes.

Source : `syseron.xml.zip` (54 Mo → **311 Mo**, 601 fichiers `xml/compteRendu/*.xml`),
Licence ouverte, législature 17, sessions 2024-2026, rafraîchi quotidiennement.

---

## 1. Ce que porte un tour de parole

Chaque `<paragraphe>` est une prise de parole, avec en attributs :

| Attribut | Exemple | Ce que ça nous donne |
|---|---|---|
| **`stime`** (sur `<texte>`) | `617.82`, `7212.80` | ⏱️ **horodatage réel en secondes** depuis le début de séance, monotone. **90,4 %** de couverture |
| **`id_acteur`** | `PA721908` | 🧑 identifiant **stable** d'orateur (joignable à la base des acteurs AN : nom, groupe). **90,1 %** |
| **`code_grammaire`** | `PAROLE_GENERALE`, `SUSP_SEANCE_2_1` | 🏷️ **type de prise de parole** — le filtre procédural, gratuit |
| **`ordre_absolu_seance`** | `17` | ordre strict dans la séance |
| **`roledebat`** | `president` | rôle (13,9 % des paragraphes = le perchoir) |
| **`code_style`** | `Info Italiques` | didascalies « (Applaudissements.) » (9,0 %) |
| `<orateurs>` | `Mme la présidente 721908` | libellé affiché + id |

Métadonnées de séance : date, numéro, session, législature, président de séance, sommaire,
**intitulés des points** de l'ordre du jour (= le sujet débattu, déjà segmenté).

## 2. Les deux risques annulés

**⚠️→✅ « Pas d'horodatage »** — faux. `stime` donne un **timecode réel en secondes** (celui du
flux vidéo), à 90 % de couverture. On avait prévu de fabriquer une horloge de rejeu : on peut
**rejouer à la cadence exacte de la séance réelle**, ou l'accélérer d'un facteur choisi. C'est
mieux qu'un ordre seul.

**⚠️→✅ « La parole procédurale va polluer la synthèse »** — le corpus la **type explicitement**.
`code_grammaire` distingue la vraie parole des scrutins, suspensions, adoptions, rejets,
annonces. Le filtre n'est pas à inventer : il est à *lire*.

Distribution sur 321 892 paragraphes :

| Préfixe | Part | Nature | Garder ? |
|---|--:|---|:--:|
| `PAROLE` | 47,5 % | prise de parole de fond (médiane **234 c**, 53 % ≥ 200 c) | ✅ |
| `INTERRUPTION` | 22,4 % | « Très bien ! », « Oui ! », (Protestations.) — médiane **27 c** | ❌ |
| `DISC` | 11,6 % | discussion générale, amendements (médiane 186 c) | ✅ |
| `SCRUT` | 9,8 % | mises aux voix, résultats de scrutin | ❌ |
| `RAP` | 1,1 % | rapporteur (médiane 172 c) | ✅ |
| `SUSP` / `ADOP` / `REJET` / `ANN` / `FIN` … | ~6 % | procédure pure | ❌ |

Le contraste est net : les catégories à garder ont une **médiane de 172–234 caractères**, celles
à jeter **27–64**. La séparation est franche, pas un réglage délicat.

## 3. Rendement du filtre

Filtre appliqué : `code_grammaire ∈ {PAROLE, DISC, RAP}` **et** `roledebat ≠ president`
**et** `code_style ≠ Info Italiques` **et** `longueur ≥ 200 c`.

```
321 892 paragraphes  →  94 709 tours substantiels  (29,4 %)
                        ~158 tours par séance, sur 601 séances
```

**~95 000 prises de parole exploitables**, nommées et horodatées. À titre de comparaison, notre
plus gros corpus actuel (Grand Débat) fait 22 174 avis.

## 4. La cadence — le chiffre qui décide de la faisabilité

Sur les séances les plus denses : **~330 tours étalés sur ~5 h** ⇒ **un tour toutes les ~50
secondes**.

Notre budget mesuré (extraction + stance d'une intervention) est de **quelques secondes**.

→ **~10× de marge sur le temps réel.** On peut rejouer une séance à vitesse réelle sans jamais
prendre de retard, et même accélérer ×10 en restant à flot. La faisabilité temporelle du live
n'est pas serrée : elle est confortable.

## 5. Séances candidates pour la démo

Les meilleures ne sont pas les plus grosses, mais les plus **clivantes** — il faut de
l'opposition réelle pour que la stance ait quelque chose à montrer.

| Séance | Tours | Orateurs | Durée | Pourquoi |
|---|--:|--:|--:|---|
| **Abrogation de la retraite à 64 ans** (28 nov. 2024) | 331 | **72** | 4,9 h | ⭐ clivage massif et connu, beaucoup d'orateurs → l'axe « position par orateur » s'y voit |
| **Impôt sur le patrimoine des ultrariches** (20 fév. 2025) | **375** | 39 | 5,0 h | ⭐ le plus de tours ; sujet fortement polarisé, un seul thème tenu |
| Restaurer un système de retraite plus juste (31 oct. 2024) | 324 | 48 | 4,0 h | même famille — utile comme **amorçage** de la séance retraite |
| Lutte contre l'antisémitisme dans l'enseignement supérieur (2 juil. 2026) | 341 | 48 | 5,2 h | clivant, registre différent |
| *Questions au gouvernement* (plusieurs) | ~325 | **70–92** | 6,0 h | ⚠️ **change de sujet toutes les 2 min** — mauvais pour une synthèse d'un débat, mais **excellent banc de dérive thématique** plus tard |

**Recommandation** : « **Abrogation de la retraite à 64 ans** », amorcée sur « Restaurer un
système de retraite plus juste » (même sujet, séance antérieure) — ce qui résout proprement le
démarrage à froid décrit dans `LIVE_SCENARIO.md` §3.

## 6. Ce qui reste à vérifier (non bloquant)

- **Groupe politique par orateur** : `id_acteur` est stable, mais le groupe (LFI, RN, EPR…) vit
  dans un **autre** jeu de données AN (acteurs/organes). À joindre — c'est ce qui permettrait un
  affichage par groupe en plus de par personne.
- **Sens hors contexte** : une réplique de 200 c qui répond à l'orateur précédent peut être
  inintelligible seule. L'extraction verbatim survit, la *synthèse* moins. À regarder sur les
  claims réels, pas à supposer.
- **Ingestion XML** : `pipeline/ingest` ne gère que `csv | jsonl` → un collecteur XML → jsonl
  reste à écrire (confirmé, inchangé).

## 7. Conséquence sur le plan

Les étapes 1 (reconnaissance) est **faite**. Le risque de contenu n°1 est **retiré** de la table
des risques de `LIVE_SCENARIO.md`, et l'horloge de rejeu se simplifie (elle lit `stime` au lieu
de simuler). Les vrais risques restants sont inchangés et tous en aval :
**positions ≠ embeddings** (la stance doit passer par le LLM), **cible de stance à revalider**
hors tiktok, et **cache incrémental d'embeddings** (`_emb_fingerprint`, verrou dur).

Corpus de travail : `scratchpad/xml/compteRendu/` (601 fichiers). En cas de go, il ira dans
`data/raw/` (gitignoré), alimenté par un collecteur — pas commité.

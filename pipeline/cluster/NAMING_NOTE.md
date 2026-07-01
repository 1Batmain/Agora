# NAMING_NOTE — Fix nommage générique (c-TF-IDF + mots-vides corpus-dérivés)

## Problème
La consultation porte sur TikTok → « tiktok » sature le corpus → l'ancien TF-IDF le
remontait comme label de plusieurs macros (`tiktok · tok · tik`, `interdit · tik · tok`,
`tiktok · rend triste · algorithme`), masquant le vrai sujet. Les clusters sous-jacents
étaient cohérents : c'est le **nommage** qui échouait.

## Contrainte directrice : ZÉRO hardcoding de domaine
L'outil tourne sur des centaines de consultations (sujets et langues variés). Aucun mot
de domaine n'est codé en dur (`grep -i "tiktok\|vélo\|retraite" naming.py` → seulement
docstring/commentaires). Les mots-saturants sont **dérivés des statistiques du corpus**.
TikTok n'est qu'un cas de test.

## Méthode (3 mécanismes, du levier principal au complément)

### 1. Mots-vides de DOMAINE — cutoff `max_df` au niveau DOCUMENT (levier principal)
On mesure la **document-frequency (DF)** de chaque terme sur TOUT le corpus
(**un avis = un document** — pas une présence par cluster, qui raterait un terme massif
mais concentré : « tiktok » n'est que dans 18/47 sous-clusters, mais avec une TF énorme).
Un terme dont la DF dépasse un **seuil dérivé** est un mot-vide *de ce corpus*.

Dérivation du seuil (pas de magic number) :
- candidats = termes de **contenu** (hors mots-outils) avec DF aberrante haute (> μ + 3σ
  de la distribution des DF — règle 3-sigma standard) ;
- on isole le **plateau saturant** de la tête via le plus grand **écart relatif** (gap)
  entre termes consécutifs : seuls les termes au-dessus du gap sont retirés. Ainsi un mot
  de contenu modérément fréquent (`vidéos`, `addiction`, `algorithme`…) reste disponible.

Corpus-relatif, langue-agnostique, zéro liste en dur.

### 2. Mots-vides FONCTIONNELS (linguistique, pas domaine)
Mots-outils (le, la, et / the, and / der, die…) chargés **dynamiquement** depuis
`stopwordsiso` (50+ langues), pris en **union** (aucune détection de langue). Repli
statistique si la lib manque : token **court ET fréquent** (faible longueur + haute DF),
signal universel de mot-outil.

### 3. c-TF-IDF (class-based, complément)
Chaque cluster = un document (concat de ses avis). Pondération BERTopic
`ctfidf(t,c) = tf(t,c) · log(1 + A / f(t))` : fait remonter le terme **distinctif** d'un
cluster et écrase ce qui est commun à tous. Complément « soft » du retrait « hard ».

L'API de `naming.py` est inchangée (`name_clusters(cluster_docs, top_k, label_k)`),
utilisée telle quelle par `build.py`, `hierarchy.py` (macro + sous) et le backend
`/recluster`. Le set de mots-vides global est dérivé une fois et partagé entre niveaux.

## Résultats sur NOTRE corpus (TikTok, 1597 avis, 55 thèmes)

**Seuil de saturation DÉRIVÉ** : `domain_cutoff_df ≈ 0.191` → « terme présent dans
> 19.1 % des avis = mot-vide de domaine ». Saturants **auto-détectés** (sans être codés) :

| terme    | DF    | détecté comme saturant |
|----------|-------|------------------------|
| `tiktok` | 0.244 | ✅ (le vrai mot-de-domaine, jamais écrit dans le code) |
| `temps`  | 0.191 | ✅ (descripteur ubiquitaire de ce corpus) |

> Note : un cutoff fixe « 25-30 % » raterait `tiktok` (24.4 %). Le seuil **dérivé** des
> données s'y adapte et l'attrape — c'est l'intérêt de ne pas hardcoder la valeur.

**Taux de mot-saturant dans les labels : 60 % → 0 %** (33/55 → 0/55).

**Avant / après (macros — le vrai sujet émerge) :**

| macro | AVANT | APRÈS |
|------:|-------|-------|
| 0 | baisse · perdu temps · culpabilité après | sentiment · perte · culpabilité |
| 1 | faux compte · collège · menaces | fille · **harcèlement · haine** |
| 2 | **tiktok · tok · tik** | application · faire · passer |
| 3 | rapeur · voile · video | vidéos · contenus · vidéo |
| 4 | interdit · **tik · tok** | réseaux · fille · sociaux |
| 5 | **tiktok** · rend triste · algorithme | **algorithme** · vidéos · triste |
| 6 | application · appli · addiction | application · **addiction** · appli |
| 7 | parfait · comparer · grosse | **corps** · parfait · **comparaison** |

« tiktok / tik / tok » ont disparu des labels ; émergent l'exposition des jeunes au
harcèlement, l'image du corps / comparaison, l'addiction, l'algorithme.

## Preuve de GÉNÉRICITÉ (test vélo — autre sujet, aucun code spécifique)
Mini-corpus factice de 30 phrases sur le **vélo** passé au même code :
- `domain_cutoff_df = 1.0`, saturant auto-détecté = **`vélo` (DF 1.0)** ;
- `vélo` **retiré** des labels ; émergent les vrais distinctifs (`cargo · acheter`,
  `chambre · adaptés`…).

Le mécanisme attrape « vélo » exactement comme « tiktok » — **sans aucun mot codé**.

## Reproductibilité
Naming = texte pur, **aucun ré-embed**, aucun aléa (seed sans objet). Le graph est
relabelé en réutilisant les clusters existants :

```
uv run python -m pipeline.cluster.relabel_graph frontend/public/graph.json frontend/public/graph.json
```

`meta.naming` du graph trace la méthode, le seuil dérivé et les saturants détectés.

## Fichiers
- `pipeline/cluster/naming.py` — c-TF-IDF + mots-vides corpus-dérivés (cœur).
- `pipeline/cluster/build.py` — dérive le set global une fois, le partage macro + sous.
- `pipeline/cluster/relabel_graph.py` — relabel d'un `graph.json` sans ré-embed.
- `pyproject.toml` — ajout dépendance linguistique `stopwordsiso` (repli si absent).

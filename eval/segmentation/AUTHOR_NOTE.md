# Gold de segmentation — note de rédaction (`gold_large.json`)

Vérité terrain **étendue** pour la segmentation sémantique des avis, écrite à la main.
Registre = consultation citoyenne type TikTok (bien-être des jeunes en ligne, FR,
1ʳᵉ personne, style témoignage réel). Remplace/étend `gold.json` (32 items) sans le
contredire : les **32 items d'origine sont repris à l'identique** (mêmes `id`).

## Stats (auto-vérifiées)
- **305 avis** au total : **104 mono** + **201 multi**.
- multi : **135 à 2 segments** + **66 à 3 segments** (~33 % à 3 thèmes).
- **267 frontières** gold (jointures inter-segments), **468 segments** labellisés.
- Couverture par thème (occurrences mono + segments multi) :
  harcelement 73 · addiction 71 · algorithme 71 · image_corps 69 · enfants 67 ·
  contenus_choquants 65 · desinformation 57 · sante_mentale 99.
  (`sante_mentale` domine car c'est le thème « conséquence » naturel en fin de chaîne.)

## Taxonomie (8 thèmes, identique à `gold.json`)
harcelement · addiction · image_corps · algorithme · contenus_choquants · enfants ·
desinformation · sante_mentale.

## Format (identique à `gold.json`)
`{ "_doc", "join":" ", "taxonomy":{…}, "items":[…] }`
- mono : `{"id":"mono-NNN","type":"mono","theme":<t>,"text":…}` — **aucune** frontière
  interne attendue (teste les faux positifs : un avis cohérent ne doit pas être coupé).
- multi : `{"id":"multi-NNN","type":"multi","segments":[{"theme":<t>,"text":…},…]}`.
  Le **texte complet** = `" ".join(segment.text)` ; les **frontières gold** tombent aux
  offsets cumulés des jointures.

## Méthode
- Chaque segment est **dominé par UN seul thème**. Une phrase ambiguë est tranchée vers
  le thème dominant, ou scindée. Labellisation honnête (pas de sur-segmentation).
- Les multi se lisent comme **UN témoignage qui glisse** d'un thème à l'autre, pas comme
  une concaténation mécanique : les segments 2+ ouvrent sur des connecteurs naturels
  (« et du coup », « résultat », « en plus », « forcément », « par contre »…) ou sur une
  transition **sémantique sans ponctuation marquée**, pour forcer une détection de sens
  et non lexicale. La frontière = là où le **thème dominant change**.
- Variété recherchée : points de vue (ado, parent, prof, grand-parent, étudiant·e),
  longueurs (court ↔ long), netteté de transition (explicite ↔ douce), vocabulaire ;
  toutes les paires de thèmes plausibles sont couvertes, ainsi que des chaînes à 3 thèmes.
- Grammaire volontairement parfois relâchée (« y a », élisions, oralité) comme de vrais avis.

## Auto-vérification (passée avant commit)
JSON parse OK · tous les `theme` ∈ taxonomie · chaque multi ≥ 2 segments ·
**aucun texte de segment dupliqué** (mono + multi confondus) · aucun couple de segments
adjacents de même thème · total ≥ 300 · IDs uniques · 32 items de `gold.json` préservés.

## Caveats
- Corpus **synthétique** : plausible et représentatif du registre, mais ce ne sont pas de
  vrais verbatims de consultation. À utiliser comme référence de segmentation, pas comme
  échantillon statistique d'opinions réelles.
- Les frontières gold supposent le `join` à **un seul espace** ; tout autre assemblage
  décalerait les offsets.
- Léger déséquilibre assumé en faveur de `sante_mentale` (thème de conséquence) et
  `harcelement`/`addiction` (thèmes d'amorce les plus fréquents).
- Le découpage en « thème dominant par segment » reste un choix d'auteur : certaines
  transitions douces admettent une frontière à ±1 phrase.

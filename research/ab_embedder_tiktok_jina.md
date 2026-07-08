# A/B embedder — dataset FR servi (pas de gold : métriques INTERNES + inspection)

> Dataset **tiktok** · 2511 claims · clustering de PRODUCTION (derive_defaults → knn → Leiden → macro-forest) · embed en mémoire (caches servis intacts).

> ⚠️ Sans vérité terrain, silhouette/modularité sont INDICATIVES (piège e5). La cohérence NPMI (fr) et l'inspection humaine priment.

## Scorecard (interne)

| Métrique | sens | nomic-v2 | jina-v3 |
|---|:--:|:--:|:--:|
| Cohérence NPMI (fr) | ↑ | -0.272 | -0.201 |
| Silhouette (cosine) | ↑ | 0.075 | 0.077 |
| Modularité (Leiden) | ↑ | 0.628 | 0.666 |
| Stabilité (ARI boot) | ↑ | 0.656 | 0.684 |
| # clusters fins | · | 12 | 12 |
| # macro-thèmes | · | 12 | 12 |
| k dérivé | · | 13 | 13 |
| seuil dérivé | · | 0.61 | 0.461 |
| dimension | · | 768 | 1024 |

**Accord des partitions** NMI(nomic, arctic) = **0.519** (1 = clusters identiques ; bas = les deux voient des thèmes différents).


## Macro-thèmes — nomic-v2 (12 macros, 12 fins)

- **instagram · faire · application** (536 claims) — _instagram, faire, application, passer, addictif, passe_
    - « Tiktok est une plateforme qui montre principalement du contenue qui pousse à la culture du vide »
    - « Attention à ne pas trop se focaliser sur tiktok, les reels d'instagram sont la même chose. Tout aussi addictif et tout a… »
- **haine · commentaires · contenu** (473 claims) — _haine, commentaires, contenu, harcèlement, contenus, images_
    - « la plupart du contenu peut même parfois être très choquant , les propos sont violents et haineux ou enferme l'utilisateu… »
    - « Souvent suite à des contenus de haine envers certaines personnes et communautés (notamment antisémites, islamophobes et … »
- **sentiment · addiction · dépendance** (300 claims) — _sentiment, addiction, dépendance, anxiété, dépression, procrastination_
    - « Cela peut entrainer une dépendance, ce qui fut mon cas lorsque j'étais au lycée »
    - « Sensation d'addiction et de perte de contrôle »
- **perte · culpabilité · perdre** (268 claims) — _perte, culpabilité, perdre, impression, sentiment, journée_
    - « incapacité à s'extraire de son téléphone et à profiter de la vie réelle »
    - « de la culpabilité de passer trop de temps dessus »
- **vidéos · vidéo · regarder** (226 claims) — _vidéos, vidéo, regarder, voir, choquantes, videos_
    - « regarder des vidéos courtes pendant 1h fait perdre notre productivité, notre temps (précieux), ne nous apprend absolumen… »
    - « doxxing suite au post d'une vidéo avec laquelle un influenceur masculiniste n'était pas d'accord »
- **réseaux · sociaux · influenceurs** (209 claims) — _réseaux, sociaux, influenceurs, jeunes, enfants, plateformes_
    - « Sans prendre de recul sur notre rapport aux réseaux sociaux, il semble peu probable de sortir de cette addiction »
    - « la plateforme crée une boucle d'addiction et après c'est dur de s'en sortir , à part supprimer il n' y a pas grand chose… »
- **application · appli · désinstallé** (154 claims) — _application, appli, désinstallé, désinstaller, heures, supprimé_
    - « j'avais fermé mon compte à l'époque et je n'ai plus jamais eu de problèmes »
    - « Quand je passe trop de temps sur l'application, j'ai le sentiment d'être vide et de me distraire du monde qui m'entoure »
- **fille · collège · photo** (135 claims) — _fille, collège, photo, fils, classe, amie_
    - « Ma fille en mal être suite a divorce c est automutile avec TS , elle a influencé son frère qui lui avait simplement comm… »
    - « Inscription au compte de ma fille par des personnes mal intentionnées et allusions dans des publications »

## Macro-thèmes — jina-v3 (12 macros, 12 fins)

- **instagram · triste · application** (448 claims) — _instagram, triste, application, faire, passer, addictif_
    - « Tiktok est une plateforme qui montre principalement du contenue qui pousse à la culture du vide »
    - « Attention à ne pas trop se focaliser sur tiktok, les reels d'instagram sont la même chose. Tout aussi addictif et tout a… »
- **harcèlement · commentaires · haine** (379 claims) — _harcèlement, commentaires, haine, insultes, messages, reçu_
    - « Souvent suite à des contenus de haine envers certaines personnes et communautés (notamment antisémites, islamophobes et … »
    - « Très souvent ceux qui sont derrière leur téléphone utilisent des mots très blessant et se croisent absolument tout permi… »
- **fille · enfants · suicide** (362 claims) — _fille, enfants, suicide, réseaux, contrôle, sociaux_
    - « Ma fille en mal être suite a divorce c est automutile avec TS , elle a influencé son frère qui lui avait simplement comm… »
    - « Mon fils a accidentellement été exposé à un live TikTok de Jordan Tules alors qu'il cherchait simplement à regarder ses … »
- **vidéos · contenu · contenus** (349 claims) — _vidéos, contenu, contenus, vidéo, images, voir_
    - « regarder des vidéos courtes pendant 1h fait perdre notre productivité, notre temps (précieux), ne nous apprend absolumen… »
    - « la plupart du contenu peut même parfois être très choquant , les propos sont violents et haineux ou enferme l'utilisateu… »
- **culpabilité · perdre · sentiment** (284 claims) — _culpabilité, perdre, sentiment, impression, procrastination, perte_
    - « de la culpabilité de passer trop de temps dessus »
    - « j'avais fermé mon compte à l'époque et je n'ai plus jamais eu de problèmes »
- **application · appli · désinstaller** (183 claims) — _application, appli, désinstaller, heures, détacher, addiction_
    - « Sans prendre de recul sur notre rapport aux réseaux sociaux, il semble peu probable de sortir de cette addiction »
    - « la plateforme crée une boucle d'addiction et après c'est dur de s'en sortir , à part supprimer il n' y a pas grand chose… »
- **corps · comparaison · parfait** (155 claims) — _corps, comparaison, parfait, comparer, influenceurs, filles_
    - « Comparaison à autrui »
    - « En se comparant aux physiques des personnes qui se mettent en scène, on se sent dénigré »
- **désinformation · informations · fausses** (133 claims) — _désinformation, informations, fausses, angoisse, ordre, politiques_
    - « angoisse de voir mes pensées arriver »
    - « je peux être irritée par ce que je vois (consumérisme, idioties en tout genre, plagiat infini) »

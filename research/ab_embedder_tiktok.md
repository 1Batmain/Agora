# A/B embedder — dataset FR servi (pas de gold : métriques INTERNES + inspection)

> Dataset **tiktok** · 2511 claims · clustering de PRODUCTION (derive_defaults → knn → Leiden → macro-forest) · embed en mémoire (caches servis intacts).

> ⚠️ Sans vérité terrain, silhouette/modularité sont INDICATIVES (piège e5). La cohérence NPMI (fr) et l'inspection humaine priment.

## Scorecard (interne)

| Métrique | sens | nomic-v2 | arctic-l |
|---|:--:|:--:|:--:|
| Cohérence NPMI (fr) | ↑ | -0.272 | -0.255 |
| Silhouette (cosine) | ↑ | 0.075 | 0.09 |
| Modularité (Leiden) | ↑ | 0.628 | 0.632 |
| Stabilité (ARI boot) | ↑ | 0.656 | 0.669 |
| # clusters fins | · | 12 | 12 |
| # macro-thèmes | · | 12 | 12 |
| k dérivé | · | 13 | 13 |
| seuil dérivé | · | 0.61 | 0.391 |
| dimension | · | 768 | 1024 |

**Accord des partitions** NMI(nomic, arctic) = **0.608** (1 = clusters identiques ; bas = les deux voient des thèmes différents).


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

## Macro-thèmes — arctic-l (12 macros, 12 fins)

- **faire · application · instagram** (534 claims) — _faire, application, instagram, vie, passer, vidéos_
    - « Tiktok est une plateforme qui montre principalement du contenue qui pousse à la culture du vide »
    - « Attention à ne pas trop se focaliser sur tiktok, les reels d'instagram sont la même chose. Tout aussi addictif et tout a… »
- **haine · commentaires · contenu** (428 claims) — _haine, commentaires, contenu, contenus, harcèlement, désinformation_
    - « la plupart du contenu peut même parfois être très choquant , les propos sont violents et haineux ou enferme l'utilisateu… »
    - « Souvent suite à des contenus de haine envers certaines personnes et communautés (notamment antisémites, islamophobes et … »
- **sentiment · perte · addiction** (390 claims) — _sentiment, perte, addiction, culpabilité, perdre, impression_
    - « Cela peut entrainer une dépendance, ce qui fut mon cas lorsque j'étais au lycée »
    - « Sensation d'addiction et de perte de contrôle »
- **vidéos · vidéo · contenus** (321 claims) — _vidéos, vidéo, contenus, voir, choquantes, contenu_
    - « regarder des vidéos courtes pendant 1h fait perdre notre productivité, notre temps (précieux), ne nous apprend absolumen… »
    - « doxxing suite au post d'une vidéo avec laquelle un influenceur masculiniste n'était pas d'accord »
- **application · scroller · appli** (230 claims) — _application, scroller, appli, heures, arrêter, sentiment_
    - « la plateforme crée une boucle d'addiction et après c'est dur de s'en sortir , à part supprimer il n' y a pas grand chose… »
    - « incapacité à s'extraire de son téléphone et à profiter de la vie réelle »
- **corps · comparaison · influenceurs** (213 claims) — _corps, comparaison, influenceurs, vie, parfait, filles_
    - « Comparaison à autrui »
    - « Ma fille en mal être suite a divorce c est automutile avec TS , elle a influencé son frère qui lui avait simplement comm… »
- **réseaux · sociaux · enfants** (198 claims) — _réseaux, sociaux, enfants, contrôle, jeunes, réseau_
    - « Sans prendre de recul sur notre rapport aux réseaux sociaux, il semble peu probable de sortir de cette addiction »
    - « J'y vois un vrai danger pour les plus jeunes, sans compter les problèmes de désinformation »
- **fille · collège · photo** (139 claims) — _fille, collège, photo, harcèlement, fils, compte_
    - « Inscription au compte de ma fille par des personnes mal intentionnées et allusions dans des publications »
    - « Grosse moquerie sous des commentaires tiktok »

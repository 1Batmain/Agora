# Audit capacités & manques — Agora (juillet 2026)

**Auditeur** : audit produit+technique indépendant, mandat « honnêteté avant pitch ».
**Méthode** : (1) tests live de l'API publique `https://forge.tail0b8aa8.ts.net/api` (tous les endpoints, curl, payloads inspectés — *rien n'est cru sur parole*) ; (2) lecture du code (`backend/`, `pipeline/`, `frontend/src/redesign/`) avec absences confirmées par grep ; (3) dépouillement exhaustif des verdicts R&D (`research/*.md`, mémoire projet) en distinguant validation **aveugle** / **gold externe** / **auto-déclarée** ; (4) comparaison à l'état de l'art (Pol.is, Talk to the City, Make.org, Decidim) et aux besoins d'un commanditaire public.

---

## Récapitulatif exécutif

| # | 5 forces PROUVÉES | Preuve |
|---|---|---|
| 1 | **Traçabilité verbatim totale** : chaque claim est une sous-chaîne exacte de l'avis, spans servis en live, surlignage cliquable jusqu'à la phrase du citoyen | Gate dur, 100 % sur 14 680 claims (3 corpus) ; vérifié live sur `/avis/{id}` |
| 2 | **Thèmes émergents fidèles au réel** : sur le Grand Débat, couverture 14/14 des sous-thèmes de la synthèse officielle OpinionWay, 0 contresens | Gold externe ; alignement 4.93/5 (v1), 4.57/5 (v2 re-extrait) |
| 3 | **Extraction v2 validée en AVEUGLE** (panel de 3 juges, A/B anonymisé, 42 avis stratifiés) : sur-segmentation corrigée sans perte de thème | 75 % des décidés pour v2, 35/42 unanimes, complétude 37/42 tie |
| 4 | **Stance calibrée contre un gold externe adverse** : 0.79 sur les décidés (x-stance DE/FR/IT), et la confiance auto-déclarée est *réellement* calibrée (bande « high » = 81 %, abstention 2,4 %) | 3 000 items gold, benchmark hostile = plancher |
| 5 | **Honnêteté structurelle + coût dérisoire** : métriques anti-fantômes (« consensus » relabellisé « cohésion sémantique — PAS un accord d'opinion »), `sovereign:false` affiché, coût LLM mesuré (~0,06–0,08 $ la phase analyse), install 1 commande, mode public fail-closed vérifié live | Contrat de métriques appliqué (vérifié `strings.ts` + payload live) |

| # | 5 manques BLOQUANTS pour se vendre | Effort |
|---|---|---|
| 1 | **RGPD inexistant** : zéro mention légale/DPO/rétention/effacement ; masquage PII regex-only (noms/adresses non couverts) ; **texte brut committé dans l'historique git** | M |
| 2 | **Aucun export ni rapport** (PDF/CSV/DOCX) : le livrable attendu par tout commanditaire public n'existe pas | S–M |
| 3 | **Complétude d'analyse** : 11 % du Grand Débat (3 000/28 384), 4 % de x-stance — le passage à l'échelle réelle n'est pas démontré | M–L |
| 4 | **Pas de human-in-the-loop** : l'analyste ne peut ni renommer, ni fusionner, ni corriger un thème (flags passifs seulement) — exigé explicitement par le défi AN (« auditer chaque cluster ») | M |
| 5 | **Zéro représentativité** : aucune donnée socio-démo, aucune pondération, pas de détection de campagnes organisées (au-delà du near-dup cos>0.95) — « 96 % favorables » n'est pas défendable sans dire *qui* | M–L |

---

# Volet A — Ce que l'outil sait VRAIMENT faire aujourd'hui

Chaque capacité : est-elle servie en prod ? validée comment ? avec quelles limites franches.

## A.1 Extraction de claims verbatim (le socle) — SERVIE, VALIDÉE, la meilleure carte du produit

**Servi** (vérifié live) : `/avis/{id}` renvoie l'avis + chaque claim avec `spans` (offsets caractères), `target`, `leaf_id`, `theme_title`, `stance`, `stance_confidence`, `stance_justif`. Le front surligne les extraits à la couleur du thème (`AvisDetail.tsx`).

**Validation** — la plus solide du projet :
- **Fidélité verbatim 100 %**, mesurée exhaustivement par gate dur (`is_verbatim`), pas sur échantillon : 14 680/14 680 claims sur les 3 corpus unifiés (`research/unify_note.md`), 5 842/5 842 sur repnum.
- **Extraction v2 validée en AVEUGLE** (`research/v2_quality_note.md`) : panel de 3 juges, A/B anonymisé, ordre randomisé, 42 avis stratifiés. La v2 corrige la sur-segmentation (75 % des décidés, 35/42 unanimes 3-0) **sans perdre de thème** (complétude : 37/42 égalité ; 19/22 sur les avis multi-thèmes). C'est le standard méthodologique du projet : le gain avait d'abord été validé en non-aveugle, puis re-testé en aveugle avant adoption.
- Le prompt « relâché B » (complétude multi-thèmes) validé par juge neutre : +15 % de rappel (+16 % Grand Débat), verbatim 98,8 % (`research/extract_ab_note.md`).

**Limites franches** :
- **Sur-fusion résiduelle** : ~2/42 avis sévèrement sur-fusionnés (6 claims → 1), ~14 % d'avis touchés — connu, filtrable en aval, non filtré aujourd'hui.
- **Le bruit est le prix du rappel** : 3 tentatives de nettoyage au prompt ont échoué (`research/extract_b2_note.md`) — le bruit modéré de B est assumé, non séparable.
- L'ancrage est sur `text_clean` (PII masquée), **pas sur le brut** — cohérent, mais dépend de la qualité du masquage (voir A.8).
- **L'extraction passe par l'API Mistral** (mistral-large, UE) : le texte citoyen sort de la machine. La souveraineté revendiquée est donc **partielle** — et le produit le dit lui-même (`params.sovereign: false`, `data_note: "Données envoyées à l'API Mistral (UE)"` dans le payload live — honnêteté vérifiée). Les alternatives locales (ministral-3b : 0.934, MLP/nomic : 0.939, qui *battent* Mistral 0.928 sur les thèmes) sont benchmarkées mais **pas le chemin servi**.

## A.2 Cartographie thématique émergente — SERVIE, VALIDÉE contre un gold officiel, avec deux astérisques

**Servi** (vérifié live) : `/analysis` renvoie 355 thèmes sur 5 niveaux (19 macros, 266 feuilles) pour le Grand Débat, avec titre, hook, description, poids, mots-clés, claims représentatifs, et les **paramètres dérivés affichés** (k=14, seuils, critères de fusion — transparence algorithmique réelle). Zéro nom de corpus en dur (contrat de généricité).

**Validation** :
- **Témoin Grand Débat officiel** (gold externe = synthèse OpinionWay) : **couverture 14/14 = 100 %** des sous-thèmes officiels de l'axe Démocratie & citoyenneté, alignement 4.93/5, **0 mismatch** (`research/granddebat_witness_note.md`). Maintenu après re-extraction v2 : 14/14, 0 mismatch, alignement 4.57 (−0.36 = recadrages de titres, pas de perte de thème).
- **Contre-exemple assumé** (repnum) : sur un corpus mono-domaine convergent, 1 macro capte **92 % des voix** — les axes officiels ne réapparaissent qu'au niveau 1 (`research/repnum_note.md`). L'outil sur-concentre au macro quand le corpus est homogène ; c'est documenté, pas caché.
- **Réglages bornés par la mesure** : k-NN (k=12–13 optimal, monter k dégrade tout), poids d'arêtes (cosinus brut optimal), norme d'embedding (artefact de longueur, zéro signal) — trois résultats négatifs propres qui ferment les fausses pistes.
- Choix de nomic-v2 validé contre gold multilingue : regroupe par thème (NMI 0.407) et pas par langue (NMI 0.008), là où e5-small fait l'inverse (0.812 langue) avec de *meilleures* métriques internes — le piège a été identifié et évité.

**Limites franches** :
1. **Échantillonnage** : 3 000/28 384 pour le Grand Débat (11 %), 3 000/67 271 pour x-stance (4 %). Le témoin officiel valide l'échantillon de 3 000, **pas** le corpus entier. Aucune analyse de sensibilité à l'échantillon (re-tirage) n'a été faite.
- 2. **Transfert non validé sur TikTok** : tous les bancs à gold externe utilisent xstance/repnum/granddebat. Le corpus TikTok — le dataset du défi, servi à 100 % — **n'a aucun gold** ; l'embedding, le clustering et les seuils n'y sont pas validés (limite récurrente et reconnue dans les notes).
3. **Généricité affirmée, pas exécutée** : `research/genericity_audit.md` identifie 5 points « casse sur un autre corpus » (naming FR/latin-only → labels vides en cyrillique/CJK, déduits du regex, **jamais exécutés**). « Marche sur des centaines de consultations » est un contrat de conception, pas un fait mesuré.

## A.3 Synthèses LLM + citations — SERVIES, mais validées seulement par ricochet

**Servi** (vérifié live) : `/insights` (global + par thème, markdown, précalculé/caché), `/citations` (claims triés par proximité au centroïde, avec `avis_id` → remontée au verbatim possible).

**Validation** : indirecte. Le témoin officiel valide que les *thèmes* couvrent le réel ; l'alignement 4.57–4.93 valide les *titres*. Mais **aucun banc ne valide les synthèses elles-mêmes** (fidélité du résumé aux claims, hallucinations, proportions annoncées). La synthèse globale live du Grand Débat est plausible et structurée, mais sa fidélité n'est pas mesurée.

**Limite structurelle** : les synthèses ne sont **pas sourcées** — pas de citation inline ni de lien claim→phrase de synthèse. La traçabilité, excellente au niveau claim/thème, **s'arrête à la porte de la synthèse**. Face à Talk to the City (qui cite dans le rapport), c'est le maillon faible du récit « transparent ».

## A.4 Opinion (clivage + stance) — SERVIE, la partie la plus honnêtement fragile

**Servi** (vérifié live) : `/opinion` renvoie par feuille une **proposition polaire** (objet de clivage), les comptes fav/def/nuance, engagement, opposition, profil (consensuel/clivant), agrégation aux parents (`is_aggregate`, `child_propositions`) ; les **prompts système sont exposés dans le payload** (transparence rare). Stance par claim avec confiance et justification dans `/avis`. Filtre par stance dans `/avis_list` (vérifié : fonctionne).

**Validation — le solide** :
- **Stance vs gold externe x-stance** (3 000 items, DE/FR/IT) : accuracy **0.79 sur les décidés** (0.672 brut avec abstention=erreur), sans biais de classe, homogène par langue. x-stance est un benchmark *adverse* (stance implicite/ironique) → c'est un **plancher**, pas la perf opérationnelle.
- **La confiance auto-déclarée est bien calibrée** : « high » (73 % du volume) = 81 % d'accuracy et 2,4 % d'abstention ; « low » s'abstient à 97 %. C'est rare et exploitable (bande de fiabilité affichable).
- **Cible = objet de clivage dérivé** : seule cible cohérente testée (erreur 11,1 % sur gold manuel, toutes des sur-attributions ; titre et question rejetés avec mesures). L'agrégation sur le sujet du cluster tient (98 %/90 % nets), là où les cibles per-claim explosent (217–360 sujets, non agrégeables).

**Validation — le fragile, dit franchement** :
- **La cible de clivage v2 n'est PAS confirmée** : le « 12/15 » initial était un fit auto-déclaré (circulaire) ; re-jugé **à l'aveugle panel-3 : 6-6**, aucun gain. Le `cleavage_fit` est un proxy faible (concordance 0.58 avec le panel, à peine mieux que pile-ou-face). Le projet l'a écrit lui-même — c'est tout à son honneur — mais cela signifie que **la qualité des propositions polaires servies n'a pas de métrique de confiance fiable**.
- **Consensus par construction** : sur une consultation ouverte, les gens se regroupent autour de ce qu'ils proposent → les thèmes sont favorables à 80–96 % *par construction* (vérifié live : n0 = 96,6 % favorable). Le « X % pour / Y % contre » intra-thème n'est **pas** l'information ; le vrai signal est *entre* thèmes + les minorités sceptiques (4–20 %). Ce point est documenté en interne mais **pas expliqué dans l'UI** — un député lira « 96 % favorables » comme un sondage. Risque de mésusage élevé.
- Classe « défavorable » fragile (0/2 sur le gold manuel), erreurs = sur-attributions du favorable.
- La détection de polarité **par clustering** a échoué (NMI ≈ hasard) — c'est *pourquoi* il y a une passe LLM dédiée ; l'embedding capte le thème, pas la position.

## A.5 Front — SERVI, cohérent, sans exports

**Existe** (vérifié code + live) : carte des thèmes d3-pack avec drill-down, toggle Graphe/Densité 3D/Nuage 2D, explorateur d'avis avec **recherche plein texte** (debounce, `q` sur `/avis_list`), filtres macro/sous-thème/stance, pagination réelle, surlignage verbatim, affichage de la confiance stance (agrégée par le MIN — choix honnête), dashboard d'indices avec tooltips anti-mésusage (« PAS un accord d'opinion » — vérifié dans `strings.ts`), page de participation pour la consultation ouverte (`/submit`, dégradé proprement en public).

**N'existe pas** (confirmé par grep) : aucun export (0 occurrence de download/CSV/PDF), aucune comparaison de consultations, pas d'i18n de l'UI (FR en dur ; le *contenu* est traduit via `text_fr` + toggle original), pas de démarche RGAA (attributs aria présents ~53×, mais informels).

## A.6 Métriques & honnêteté d'affichage — APPLIQUÉE (vérifiée), un vrai différenciateur

Le contrat de métriques (mémoire projet, 2026-06) est appliqué : indices servis = effusion, concentration, **cohésion sémantique** (ex-« consensus », relabellisé côté front avec tooltip explicite), structuration, **couverture**, **fidélité verbatim** (affichée !). Les non-fournis (sentiment, évolution temporelle, représentativité) sont assumés comme absents plutôt que simulés. Le payload expose `sovereign:false` + note de données. **Bémol** : la clé JSON servie s'appelle toujours `consensus` (le relabel vit dans le front) — un consommateur d'API brute retombe dans le piège ; à renommer au serve-time.

## A.7 Reproductibilité, ops, coût — BONNE, avec un trou dans le coût affiché

- Install 1 commande vérifiable (`scripts/setup.sh` : uv + npm + caches ~250 Mo depuis release GitHub), `make dev`, CI pytest+build sur PR, déploiement auto sur push main, promotion de cache dev→prod explicite. Fonctionne **sans clé Mistral** (lecture des caches).
- **Coût** : `/cost` live = **0,064 $ (Grand Débat) / 0,077 $ (TikTok)** — mais **uniquement la phase « analysis »** (mistral-small). **Le coût d'extraction mistral-large (~9–15 $/build selon `research/audit_cost.md`) n'apparaît pas** dans le `/cost` servi. Le chiffre affiché sous-estime le coût réel d'un facteur ~100. À corriger avant toute démo où le coût est un argument.
- Pas de mesure du temps de calcul ni du coût CPU/embeddings.

## A.8 Sécurité, privacy, mode public — DURCI récemment (vérifié live), avec des dettes réelles

**Corrigé et vérifié live** : mode public fail-closed réel (`/recluster` → 403, `force:true` ignoré sur `/analysis`, `/docs` et `/openapi.json` → 404, aucun build déclenchable par une lecture publique — le scénario « facture LLM pilotée par polling anonyme » de `audit_cost.md` est fermé). Code actuel : comparaison de token à temps constant, en-têtes de sécurité, limite de corps 64 Ko, CORS restreint, `/submit` masque désormais la PII. Audit secrets : **0 fuite sur 518 commits**.

**Dettes ouvertes (réelles)** :
1. **Masquage PII regex-only** (email/tél/URL/@mention) : noms, adresses, âges, signatures **non masqués** — or les claims servis sont du verbatim citoyen.
2. **Texte brut (`text`) committé dans `ideas.jsonl`** à côté de `text_clean` : le masquage est annulé dans l'historique git d'un dépôt destiné à être public.
3. Rate-limit par IP inopérant derrière reverse-proxy (pas de X-Forwarded-For), en mémoire, par worker.
4. `author_hash` = pseudonymisation, pas anonymisation au sens RGPD (ré-identification par recoupement possible).

---

# Volet B — Ce qui MANQUE pour un outil COMPLET d'analyse de consultations

Référentiel : Pol.is (clustering d'opinion par votes, groupes d'opinion, temps réel), Talk to the City / AI Objectives Institute (rapports narratifs sourcés, exportables), Make.org (échelle, socio-démo, modération), Decidim (plateforme complète : processus, RGPD, accessibilité) — et les attentes concrètes d'une AN / d'un ministère / d'une collectivité.

**Positionnement honnête** : Agora est un excellent *moteur d'analyse* (extraction traçable + thèmes émergents validés — sur ce cœur, il est au niveau ou au-dessus de Talk to the City grâce au verbatim ancré, là où TttC paraphrase). Ce n'est **pas encore un outil** qu'un commanditaire public peut acheter : il manque la couche livrable, la couche conformité, et la couche analyste.

## B.1 BLOQUANTS (sans ça, pas de vente à un acteur public)

| Manque | Pourquoi bloquant | Effort | Suggestion concrète |
|---|---|---|---|
| **RGPD / conformité** : aucune mention légale, pas de DPO, pas de politique de rétention, pas de droit à l'effacement ; PII regex-only ; texte brut dans l'historique git | Un acteur public ne peut légalement pas déployer un outil qui sert du verbatim citoyen sans base RGPD. C'est éliminatoire en marché public, avant même la démo. | **M** | (a) Purger `text` brut de l'historique git (filter-repo) avant toute publication du dépôt ; (b) passer le masquage à un NER (Presidio/spaCy fr) pour noms/adresses ; (c) page mentions légales + registre de traitement + procédure d'effacement par `author_hash` ; (d) documenter le flux Mistral UE dans une AIPD type. |
| **Exports & rapports** : zéro export (PDF, CSV, DOCX) | Le livrable d'une consultation EST un rapport. Aujourd'hui l'outil ne produit rien qu'un cabinet ou un secrétariat général puisse déposer sur un bureau. TttC et les prestataires (Roland Berger sur le Grand Débat) livrent des rapports. | **S–M** | (a) CSV par thème (thème, n, %, stance, claims représentatifs) — 1 endpoint + 1 bouton, **S** ; (b) rapport PDF/HTML imprimable généré depuis `/insights` + `/opinion` + citations (weasyprint), avec la page « méthode & limites » auto-incluse (échantillon, couverture, fidélité verbatim) — **M**. |
| **Complétude / montée en charge** : 11 % du Grand Débat, 4 % de x-stance analysés ; tout en fichiers JSON, pas de BDD | « On analyse votre consultation » ≠ « on analyse 3 000 réponses tirées de votre consultation ». Face à une AN qui reçoit 30–200k contributions, la démo actuelle prouve la méthode, pas l'échelle. La sensibilité à l'échantillon n'est même pas mesurée. | **M–L** | (a) Court terme : **afficher l'échantillonnage en tête de chaque vue** (déjà dans `dataset_stats`, pas assez visible) + une expérience de stabilité (2 tirages de 3 000, mesurer le recouvrement des thèmes) — **S–M** ; (b) passer le Grand Débat à 28 384 via l'API batch Mistral (−50 % de coût, l'extraction v3 batchée est déjà validée non-dégradante) et publier le coût/temps réel — **M** ; c'est la preuve d'échelle la moins chère à obtenir. |
| **Human-in-the-loop** : aucune édition/validation des thèmes (renommer, fusionner, scinder, exclure un claim) ; flags passifs seulement ; pas d'audit trail | Le défi AN le demande textuellement (« validation humaine : auditer chaque cluster »). Aucun analyste public n'endossera une synthèse qu'il ne peut pas corriger. C'est aussi la réponse au risque LLM : l'humain signe. | **M** | Une « vue analyste » : renommage de titre (simple write-through sur `analysis.json` + journal), fusion de feuilles (recompute local déjà possible via la mécanique recluster), exclusion de claims du bruit (le filtre aval que `extract_b2` recommandait !), avec journal des modifications (qui/quoi/quand) exporté dans le rapport. Les flags existants sont l'embryon : les brancher. |
| **Représentativité & intégrité du corpus** : zéro socio-démo, zéro pondération, pas de détection de campagnes de copier-coller organisées (le near-dup cos>0.95 attrape les doublons, pas les campagnes reformulées) | Sans ça, « 96 % favorables » est indéfendable et même dangereux (le point est connu en interne — consensus par construction — mais pas affiché). Les consultations publiques FR sont régulièrement ciblées par des campagnes coordonnées ; un outil qui ne les voit pas peut être instrumentalisé. | **M–L** | (a) Détection de campagnes : clusters de similarité très haute + heuristiques temporelles sur `ts` (déjà stocké, inutilisé) + affichage « n contributions quasi identiques » — **M** ; (b) ingestion optionnelle de champs socio-démo (le schéma `extra` existe déjà dans `ideas.jsonl`) + tableau croisé thème × attribut — **M** ; (c) l'avertissement UI « ceci n'est pas un sondage » sur toute stat d'opinion — **S**, à faire vendredi. |

## B.2 IMPORTANTS (crédibilité face à l'état de l'art)

| Manque | Détail | Effort | Suggestion |
|---|---|---|---|
| **Synthèses sourcées** | Citations inline (claim-ids) dans les insights, cliquables vers le verbatim — la traçabilité s'arrête aujourd'hui à la synthèse | S–M | Prompt d'insights contraint à citer des ids + rendu front en notes |
| **Incertitude affichée** | Pas d'IC sur les % de stance ; n petits non signalés ; le `cleavage_fit` (proxy faible, 0.58) ne doit pas être montré comme confiance | S | IC de Wilson sur fav/def par thème + badge « n faible » ; masquer les % sous n minimal |
| **Coût complet** | `/cost` omet l'extraction mistral-large (~facteur 100) | S | Tracer la phase extraction dans `cost.json` au bake |
| **Comparaison / longitudinal** | `ts` stocké mais inexploité ; aucune vue multi-consultations ; Pol.is et Make.org vivent de ça | M | Vue « thème dans le temps » (arrivée des contributions) + comparaison de deux consultations par alignement de centroïdes |
| **API publique documentée** | OpenAPI coupé en public, aucune spec versionnée ; le contrat vit dans `contract.ts` | S | Publier openapi.json statique (lecture seule) + page docs |
| **Gold TikTok** | Le dataset vitrine du défi n'a aucune validation de bout en bout | M | Annoter 200 items (thème + stance) à la main, publier le score — honnêteté oblige |
| **Accessibilité RGAA** | aria informel, aucune démarche ; obligation légale pour un service public en ligne | M | Audit RGAA 4 critères bloquants (clavier, contrastes, alternatives) sur les 4 vues |
| **Argument mining structuré** | Les claims+stance donnent la matière, mais pas de vue « pour/contre » par proposition avec les meilleurs arguments de chaque camp | M | Regrouper les claims d'une feuille par stance et servir top-arguments par camp (les données existent déjà dans `claim_stance.json`) |

## B.3 NICE-TO-HAVE

- **Mode 100 % souverain servi** : ministral-3b (0.934) et le clf MLP (0.939) *battent* Mistral API sur les thèmes — un flag `AGORA_CLAIMS_BACKEND=mac` existe déjà ; en faire une option de déploiement documentée « on-premise intégral » serait un argument fort pour les ministères sensibles. (Effort M — le vrai coût est la re-validation du chemin complet.)
- Nuages de mots / heatmaps du cahier des charges du défi (la densité 3D couvre l'esprit).
- Rate-limit derrière proxy (X-Forwarded-For) + store partagé.
- Distribution des caches par DVC/LFS plutôt que release GitHub.
- i18n de l'UI (le contenu est déjà traduit).

## B.4 Ce qu'Agora a que l'état de l'art n'a PAS (à assumer dans le pitch)

Pour être complet, l'audit note aussi les différenciateurs réels — rares dans le domaine :
1. **Verbatim ancré au caractère près** avec gate dur à 100 % (TttC paraphrase ; Pol.is n'analyse pas le texte libre).
2. **Culture de validation** : gold externes, juges aveugles en panel, résultats négatifs publiés, auto-invalidation d'un gain circulaire (clivage v2). Aucun concurrent ne montre ça.
3. **Anti-mésusage intégré** : « cohésion sémantique ≠ accord », non-fournis affichés, prompts exposés dans l'API.
4. **Coût marginal quasi nul** et install reproductible — contre des prestations de synthèse à 6 chiffres (Grand Débat 2019).

---

# Recommandations priorisées (si vendredi = démo/jury)

1. **Boucler l'histoire « honnête » jusqu'au bout de l'UI (S, 1 jour)** : bandeau échantillonnage visible (« 3 000 avis analysés sur 28 384 »), avertissement « ceci n'est pas un sondage » sur les % d'opinion, coût d'extraction dans `/cost`, renommer la clé `consensus` au serve-time. Ce sont les 4 trous entre l'honnêteté interne (réelle) et ce que voit le jury.
2. **Un export rapport (M, 2–3 jours)** : PDF/HTML auto-généré (synthèse + thèmes + citations verbatim + page méthode & limites). C'est LE manque qui transforme un moteur en outil, et la page « méthode & limites » convertit la culture de validation — l'atout du projet — en argument visible.
3. **Purge PII de l'historique git + NER avant toute publication du dépôt (M, à ne pas différer)** : `text` brut committé + masquage regex-only = le risque juridique et réputationnel n°1 au moment où le projet devient public.

Ensuite (post-vendredi) : preuve d'échelle sur le Grand Débat complet via batch API, vue analyste (human-in-the-loop branché sur les flags), gold TikTok.

---

*Sources : tests live du 2026-06-30 (payloads dans le scratchpad de session), `research/*.md` (verdicts cités dans le texte), `backend/server.py` + `backend/auth.py` (état courant, postérieur aux audits `audit_*.md` dont plusieurs findings sont corrigés), mémoire projet. Les absences sont confirmées par grep, pas présumées.*

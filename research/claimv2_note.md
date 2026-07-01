# CLAIM v2 — multi-spans + cible (target) verbatim

> Branche `work/claim-v2`. PRIORITÉ N°1 du groom (cf. `queue/iteration-feedback-distill.md`).
> Évolue le modèle de claim de bout en bout : un claim peut prendre **plusieurs portions
> non-contiguës** d'un avis et porte une **cible** (l'aspect dont il parle), elle aussi
> **verbatim**. Re-extraction nécessaire ; gate verbatim **dur** (fidélité non négociable).

## Modèle (rétro-compatible)
- `Span = (start, end)` — offsets de caractères dans l'avis (mi-ouvert, `end` exclu).
- `Claim = {text, spans:[Span, …], target: Span | None}` (`pipeline/claims/span.py`).
  - `text` = JOINT des portions (`SPAN_JOIN = " … "`) → **embedding du texte joint**.
  - `spans` = 1..N portions VERBATIM. **Mono-span = liste de 1** → l'ancien modèle est un
    cas particulier (rétro-compatible). `start`/`end` restent exposés en **propriétés**
    (1er span / dernier span) pour les vieux lecteurs.
  - `target` = la cible/aspect, **portion VERBATIM** de l'avis (ex. « temps passé sur
    l'écran »), PAS une étiquette normalisée. La normalisation en aspect propre est un
    traitement **EN AVAL** (clustering des cibles), pas une invention à l'extraction.
- Cache `claims.json` : `{text, spans:[[s,e],…], target:[s,e]|null}` ; `as_claim` lit
  aussi le **legacy** `{text, start, end}` (un span) et la chaîne nue (ré-ancrée).

## Extraction (`pipeline/claims/extract.py`)
- Prompt : `{"claims":[{"parts":["verbatim A","verbatim B"], "target":"cible verbatim"}]}`.
  Garde les acquis du témoin : **sélectivité** (laisse le narratif/cadrage), **regroupement**
  (ne fragmente pas une idée : contraste/justification/condition/énumération), **sujet+
  position** (chaque claim dit SUR QUOI + la POSITION), **verbatim strict**, **few-shot**.
  Few-shots étendus pour illustrer multi-parts et target.
- `align_spans(avis_text, specs)` : pour chaque spec, ancre **CHAQUE part** ET **la target**
  comme sous-chaîne exacte (exact `str.find` puis tolérant aux espaces) → offsets dérivés
  par nous (le LLM ne donne pas de positions). **Part non ancrée = rejetée** ; claim sans
  AUCUNE part ancrée = écarté ; target non ancrée → `target=None` (le claim reste). Repli :
  un avis dont rien ne s'ancre devient 1 claim = son texte entier (`whole_avis_claim`).
- `parse_claims` tolérant : `parts` chaîne unique, claim donné comme chaîne (legacy → 1
  part), clés alternatives (`text`/`claim`/`verbatim`), clé `claims` renommée.

## Provenance `/avis/{id}` — FORMAT CONTRAT (cross-lane, `/tmp/contract-claimv2.md`)
`backend/avis.py` rend désormais :
```json
{ "id": "...", "text": "...",
  "claims": [ { "id": "avisid#ci", "cluster_id": "nX", "color": "#rrggbb",
                "spans": [ {"start":int,"end":int}, … ],
                "target": {"start":int,"end":int} | null,
                "theme_title": "..." } ] }
```
- Un claim par entrée (plus de liste de spans à plat) ; `spans` = ses portions ; `target`
  = sa cible (ou null) ; couleur/titre = ceux de son **macro-thème**. Trié par position.
- `theme_title` = titre court LLM du macro (repli label). (avant : `theme_label`.)

### ⚠️ HANDOFF FRONT (lane `frontend/`, hors de ce worker)
Le front consomme encore l'ANCIEN format (`body.spans` à plat, `seg.span.theme_label` —
cf. `frontend/src/redesign/{analysisApi.ts,AvisDetail.tsx,contract.ts}`). À adapter :
itérer `body.claims`, surligner les `spans` de chaque claim (couleur cluster) et
**souligner** le `target` span à l'intérieur ; renommer `theme_label`→`theme_title`.

## Deux modèles séparés dans le BUILD (`backend/build_analysis.py`)
- **EXTRACTION** (lente, ~1 appel/avis, CACHÉE) = `mistral-large-latest`
  (`AGORA_EXTRACT_MODEL`). Qualité non négociable (claims fidèles, multi-spans + target).
- **ENRICHISSEMENT** (titres/accroches/descriptions/insights, ~3-4 appels/thème) =
  `mistral-small-latest` (`AGORA_ENRICH_MODEL`). C'est le gros du coût d'un **rebuild**
  (extraction cachée) → cheap = rebuild nettement plus rapide. `render_insight` prend
  désormais un `model=`. CLI : `--model` (extraction), `--enrich-model` (enrichissement).
- Progression de l'extraction LLM remontée dans `status.json` (done/total).

## Validation
- **Unitaire** (`backend/selftest_extractive.py`) : exact / espaces / rejet / répétitions
  distinctes / **multi-spans** (2 portions → 1 claim, texte joint, verbatim) / **target** /
  round-trip cache (mono + multi). `uv run python -m backend.scripts.selftest_extractive` → OK.
- **Échantillon réel** (`backend/sample_claimv2.py`, n'écrit aucun cache) :
  `uv run python -m backend.scripts.sample_claimv2 --dataset tiktok --n 8` →
  **16 claims, 100% parts verbatim, 15/15 cibles verbatim**, 0 erreur. Sélectivité OK
  (avis « Comparaison à autrui » → 0 claim = pur narratif). Cibles pertinentes
  (« boucle d'addiction », « contenus de haine… », « perte de temps »).
- **Re-extraction tiktok + rebuild** (`build_analysis --dataset tiktok --force`,
  extraction mistral-large fraîche + enrichissement mistral-small) :
  - **1604 avis → 3024 claims** ; model `mistral-large-latest`.
  - **100% verbatim** (3024/3024 sous-chaînes exactes, selftest provenance).
  - **57 multi-spans** ; **2576 claims avec cible** (85%).
  - Claims couvrant l'avis entier : 561 — dont **470 sélections légitimes** d'un avis
    court (avec cible) ; vrais replis (extraction vide, fragments vagues type
    « Comparaison à autrui ») ≤ 91. La sélectivité fait son travail, le repli garantit
    qu'aucun avis n'est perdu.
  - **244 thèmes, 13 macros.** `/avis` rend bien le format contrat (vérifié sur
    multi-span tiktok:58 + multi-claims tiktok:20 : spans, cible, couleur+titre du macro).
  - **Temps** : extraction mistral-large ~1 h (1604 appels, 429 absorbés par le backoff)
    puis **CACHÉE** → un rebuild la saute. Rebuild (embed + clustering + enrichissement
    cheap) = **~19 min** ; un rebuild ultérieur réutilise aussi l'enrichissement caché
    par contenu → quasi instantané.

> ⚠️ Setup : l'embedder par défaut `nomic-v2` charge du code custom dépendant de
> `einops` (extra `embed-contender`). Lancer le build avec
> `uv run --extra embed-contender python -m backend.build_analysis …`.

## Acceptance
- [x] claims multi-spans + target, 100% sous-chaînes exactes (unitaire + échantillon + corpus).
- [x] `/avis` au format contrat (claims[] avec spans[] + target + theme_title) — vérifié.
- [x] build paramètre deux modèles (extraction large / enrichissement cheap) → rebuild
      rapide (extraction cachée ~1 h sautée, enrichissement cheap ~19 min).
- [x] tiktok ré-extrait (mistral-large) + rebuild → READY (244 thèmes, 100% verbatim).

"""Endpoint `/avis/{id}` — texte d'un avis + ses CLAIMS verbatim surlignables.

Provenance citoyenne : un avis affiché EN ENTIER, avec ses CLAIMS (idées extraites)
surlignés à la couleur de LEUR cluster (macro-thème) — la même couleur que les bulles
de la carte. Un claim peut prendre PLUSIEURS portions non-contiguës (`spans`) et porte
une **cible** verbatim (`target`, une sous-portion ou null). Comme tout est extractif
(sous-chaînes exactes, cf. `pipeline.claims.span`), chaque span pointe une plage réelle
du texte : highlight fidèle, zéro dérive.

    GET /avis/{id} {dataset} -> {
      id, text, text_fr|null, lang,
      claims: [ {id, cluster_id, color, spans:[{start,end}], target:{start,end}|null,
                 theme_title} ]
    }

`text` = texte CANONIQUE de l'avis = `text_clean` (`pipeline.ingest.normalize`) :
normalisé ET **PII évidentes masquées** (emails/tél./URL/@mentions → placeholders).
C'est ce MÊME texte qui a servi à l'extraction et sur lequel les spans/cibles des claims
sont ancrés (cf. `claims_endpoint._avis_from_ideas`, qui privilégie `text_clean`) : le
highlight reste donc verbatim ET on ne sert JAMAIS la PII brute (SEC3). `text_fr` =
traduction française (précalculée au BUILD sur ce même `text_clean`, cf. `backend.translate`)
ou `null` si l'avis est déjà français / non traduit. `lang` = code langue de l'avis. Le front
affiche `text_fr` par défaut quand `lang != fr`, avec un toggle « voir l'original »
(surlignages sur `text`). ⚠️ Ne JAMAIS remplacer `a.text` par le texte source brut ici :
cela réintroduirait la PII ET décalerait les offsets des spans.

Construit depuis l'arbre variance-adaptatif (`backend.analysis`) — aucun recalcul.
Précalculé au BUILD et persisté (`analysis_store`), servi tel quel (instantané).
"""

from __future__ import annotations

import json
import unicodedata

from backend.analysis import ThemeTree, macro_of


def _claim_macro(tree: ThemeTree) -> dict[int, str]:
    """Map claim (index global) → id du macro-thème qui le porte.

    Les `members` d'un macro contiennent TOUS les claims de son sous-arbre, donc
    ce balayage des macros couvre chaque claim une fois.
    """
    out: dict[int, str] = {}
    for mid in tree.macros:
        for ci in tree.nodes[mid].members:
            out[ci] = mid
    return out


def _claim_leaf(tree: ThemeTree) -> dict[int, str]:
    """Map claim (index global) → id de la FEUILLE (nœud sans enfant) qui le porte.

    Les feuilles partitionnent les claims → chaque claim reçoit exactement une feuille.
    Le `cluster_id` d'un claim = son MACRO (pour la couleur des surlignages) ; ce `leaf_id`
    permet de FILTRER l'explorateur par n'importe quel niveau de thème (feuille/sous-thème),
    via le sous-arbre — sinon un filtre non-macro ne matche jamais (claims tous au macro).
    """
    out: dict[int, str] = {}
    for nid, node in tree.nodes.items():
        if not getattr(node, "children", None):          # feuille (robuste aux mocks de test)
            for ci in getattr(node, "members", []) or []:
                out[ci] = nid
    return out


def avis_claims(tree: ThemeTree, avis_index: int, claim_macro: dict[int, str],
                claim_leaf: dict[int, str] | None = None) -> list[dict]:
    """Claims verbatim d'un avis — spans + target, colorés par macro-thème, triés par position.

    Chaque claim : `spans` (1..N portions verbatim non-contiguës), `target` (cible
    verbatim, sous-portion, ou null), couleur/titre du macro qui le porte. Un claim
    sans aucune portion ancrée (repli) est ignoré (rien à surligner).
    """
    prep = tree.prepared
    avis_id = prep.avis[avis_index].id
    text_len = len(prep.avis[avis_index].text)
    claims: list[dict] = []
    for ci, owner in enumerate(prep.claim_owner):
        if owner != avis_index:
            continue
        spans = [{"start": s, "end": e}
                 for s, e in prep.claim_spans[ci] if 0 <= s < e <= text_len]
        if not spans:                          # claim non ancré (repli) → pas de surlignage
            continue
        mid = claim_macro.get(ci)
        node = tree.nodes.get(mid) if mid else None
        tgt = prep.claim_target[ci]
        target = ({"start": tgt[0], "end": tgt[1]}
                  if tgt is not None and 0 <= tgt[0] < tgt[1] <= text_len else None)
        claims.append({
            "id": f"{avis_id}#{ci}",
            "cluster_id": mid,
            # Feuille réelle du claim (pour le filtre par sous-thème de l'explorateur) ;
            # `cluster_id`=macro reste la couleur. Repli sur le macro si absent.
            "leaf_id": (claim_leaf.get(ci) if claim_leaf else None) or mid,
            "color": node.color if node else "",
            "spans": spans,
            "target": target,
            "theme_title": (node.title or node.label) if node else "",
        })
    claims.sort(key=lambda c: (c["spans"][0]["start"], c["spans"][0]["end"]))
    return claims


def avis_payload_for(tree: ThemeTree, avis_index: int,
                     claim_macro: dict[int, str] | None = None,
                     translations: dict[str, dict] | None = None,
                     claim_leaf: dict[int, str] | None = None) -> dict:
    """`{id, text, text_fr, lang, claims}` d'un avis (par son index dans `prepared.avis`).

    `translations` (optionnel) : map `avis_id -> {lang, text_fr}` précalculée par
    `backend.translate`. Absente → `lang="fr"`, `text_fr=None` (rétro-compat, datasets FR).
    """
    if claim_macro is None:
        claim_macro = _claim_macro(tree)
    if claim_leaf is None:
        claim_leaf = _claim_leaf(tree)
    a = tree.prepared.avis[avis_index]
    tr = (translations or {}).get(str(a.id)) or {}
    # `a.text` est le texte CANONIQUE masqué (`text_clean`) sur lequel les spans sont
    # ancrés : on le sert TEL QUEL (cohérence offsets + zéro PII brute, cf. docstring).
    return {"id": a.id, "text": a.text,
            "text_fr": tr.get("text_fr"), "lang": tr.get("lang", "fr"),
            "claims": avis_claims(tree, avis_index, claim_macro, claim_leaf)}


def join_claim_stance(claims: list[dict], stance_map: dict | None) -> list[dict]:
    """Enrichit chaque claim d'une `stance` (+`proposition`/`stance_justif`) si connue.

    Transparence par claim, à l'image du surlignage verbatim : la stance bakée par
    `backend.build_opinion` (`{claim_id: {stance, justif, proposition, theme_id}}`) est
    JOINTE par l'id de claim servi (`f"{avis_id}#{index}"`). Gracieux : `stance_map`
    absente, ou claim sans entrée (thème impur / non classé), → claim inchangé. Ne touche
    NI les spans NI la cible (l'ancrage verbatim reste intact).
    """
    if not stance_map:
        return claims
    out: list[dict] = []
    for c in claims:
        rec = stance_map.get(c.get("id"))
        if rec:
            c = {**c,
                 "stance": rec.get("stance"),
                 "stance_confidence": rec.get("stance_confidence"),
                 "proposition": rec.get("proposition"),
                 "stance_justif": rec.get("justif")}
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Liste / recherche d'avis (endpoint `/avis_list`, SERVE depuis `avis.json`)
# --------------------------------------------------------------------------- #
def _fold(s: str) -> str:
    """Normalise pour une comparaison insensible à la casse ET aux accents.

    NFD + suppression des diacritiques (catégorie `Mn`) + casefold : « Réglementé »
    matche « reglemente ». Générique (aucune table de corpus), suffisant pour une
    recherche sous-chaîne plein-texte.
    """
    decomposed = unicodedata.normalize("NFD", s)
    no_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return no_marks.casefold()


def _descendants_of(themes: list[dict], theme_id: str) -> set[str]:
    """Sous-arbre fermé `{theme_id} ∪ descendants` d'après les `parent_id` des thèmes.

    Filtrer un macro doit ramener TOUS ses sous-thèmes : on dérive l'ensemble fermé du
    sous-arbre depuis la hiérarchie de `/analysis` (générique, aucune profondeur en dur).
    """
    children: dict[str, list[str]] = {}
    for t in themes:
        children.setdefault(t.get("parent_id"), []).append(t["id"])
    keep: set[str] = set()
    stack = [theme_id]
    while stack:
        cur = stack.pop()
        if cur in keep:
            continue
        keep.add(cur)
        stack.extend(children.get(cur, []))
    return keep


def _excerpt(text: str, n: int = 220) -> str:
    """Aperçu ~`n` caractères, coupé sur un mot, avec ellipse si tronqué."""
    text = " ".join(text.split())        # aplatit les blancs pour un aperçu compact
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0].rstrip()
    return (cut or text[:n].rstrip()) + "…"


def _avis_themes(claims: list[dict]) -> list[dict]:
    """Thèmes UNIQUES (id/title/color) portés par les claims d'un avis, dans l'ordre vu."""
    seen: dict[str, dict] = {}
    for c in claims:
        cid = c.get("cluster_id")
        if cid and cid not in seen:
            seen[cid] = {"id": cid,
                         "title": c.get("theme_title", ""),
                         "color": c.get("color", "")}
    return list(seen.values())


def _build_item(key: str, entry: dict) -> dict:
    """Item servi par `/avis_list` : avis ENTIER + aperçu + thèmes uniques.

    Reconstruit à L'IDENTIQUE depuis une entrée `avis.json` — partagé par le chemin Python
    (`avis_list`) ET le chemin DuckDB (`avis_list_duckdb`), garantissant une parité de shape
    entre les deux (l'un lit le dict, l'autre le `payload` verbatim de `analysis.duckdb`).
    """
    claims = entry.get("claims") or []
    text = entry.get("text") or ""
    # Avis ENTIER (text/text_fr/lang/claims) servi tel quel depuis `avis.json` —
    # spans des claims ancrés sur `text` (text_clean masqué, cf. docstring module).
    return {
        "avis_id": entry.get("id", key),
        "excerpt": _excerpt(text),
        "themes": _avis_themes(claims),
        "text": text,
        "text_fr": entry.get("text_fr"),
        "lang": entry.get("lang", "fr"),
        "claims": claims,
    }


def avis_list(avis_data: dict, themes: list[dict], *,
              theme_id: str | None = None, q: str | None = None,
              stance: str | None = None, claim_stance: dict | None = None,
              limit: int = 15, offset: int = 0) -> dict:
    """Liste paginée/filtrée des avis depuis `avis.json` → `{total, items}`.

    Chaque item porte l'avis ENTIER (`text`, `text_fr`, `lang`, `claims` — déjà dans
    `avis.json`, aucun recalcul) EN PLUS de l'aperçu (`excerpt`) et des thèmes uniques
    (`themes`), pour un rendu INLINE complet (texte + surlignages verbatim) côté front,
    sans appel `/avis/{id}` par carte. Items lourds → `limit` par défaut bas (~15).

    `theme_id` : ne garde que les avis ayant ≥1 claim dont le `cluster_id` est dans le
    sous-arbre de `theme_id` (un macro filtre tous ses sous-thèmes). `q` : sous-chaîne
    insensible casse/accents sur le `text`. `limit`/`offset` paginent le résultat filtré
    (l'ordre suit `avis.json`, stable d'une requête à l'autre).

    Chemin de repli O(N) (scan + fold Unicode par requête) : `avis_list_duckdb` sert la
    MÊME sémantique en SQL indexé quand `analysis.duckdb` est présent (cf. `server`).
    """
    keep_ids = _descendants_of(themes, theme_id) if theme_id else None
    needle = _fold(q.strip()) if q and q.strip() else None

    matched: list[dict] = []
    for key, entry in avis_data.items():
        if not isinstance(entry, dict):
            continue
        claims = entry.get("claims") or []
        # Filtre par sous-arbre : on teste la FEUILLE du claim (`leaf_id`), pas son macro
        # (`cluster_id`), sinon un filtre feuille/sous-thème ne matcherait jamais. Repli
        # sur `cluster_id` pour les avis.json anciens (sans `leaf_id`).
        if keep_ids is not None and not any(
            (c.get("leaf_id") or c.get("cluster_id")) in keep_ids for c in claims
        ):
            continue
        # Filtre par SENTIMENT : ne garder que les avis ayant ≥1 claim (DANS le thème filtré)
        # dont la stance bakée == `stance`. La stance est jointe par id de claim (`avis_id#idx`).
        if stance and claim_stance:
            def _in_theme(c: dict) -> bool:
                return keep_ids is None or (c.get("leaf_id") or c.get("cluster_id")) in keep_ids
            if not any(
                _in_theme(c) and (claim_stance.get(c.get("id")) or {}).get("stance") == stance
                for c in claims
            ):
                continue
        text = entry.get("text") or ""
        if needle is not None and needle not in _fold(text):
            continue
        matched.append(_build_item(key, entry))

    total = len(matched)
    start = max(0, offset)
    page = matched[start:start + limit] if limit >= 0 else matched[start:]
    return {"total": total, "items": page}


def avis_list_duckdb(con, themes: list[dict], *,
                     theme_id: str | None = None, q: str | None = None,
                     stance: str | None = None, claim_stance: dict | None = None,
                     limit: int = 15, offset: int = 0) -> dict:
    """Variante SQL de `avis_list` : filtrage/pagination délégués à `analysis.duckdb`.

    MÊME contrat et MÊME sémantique que `avis_list` — l'item est reconstruit par
    `_build_item` depuis le `payload` verbatim stocké au bake, donc byte-identique au
    fallback. Les prédicats miroir du fallback :
      * `theme_id` → EXISTS un claim dont `filter_theme ∈ descendants(theme_id)` (calculés
        en Python comme le fallback, générique, aucune profondeur en dur) ;
      * `stance`   → (avec `claim_stance` présent) EXISTS un claim (DANS le thème) dont la
        `stance` bakée == `stance` — même claim, comme le fallback ;
      * `q`        → `contains(text_fold, needle)` où `text_fold`/`needle` sont foldés par
        le MÊME `_fold` : sous-chaîne insensible casse/accents, parité exacte avec `in`.
    `con` est un curseur DuckDB en lecture seule (thread-local, cf. `analysis_store`).
    """
    keep_ids = _descendants_of(themes, theme_id) if theme_id else None
    needle = _fold(q.strip()) if q and q.strip() else None

    conds: list[str] = []
    params: list = []

    # Filtre thème/stance : semi-jointure sur `claims` (`rank IN (SELECT …)`), planifiée en
    # HASH JOIN par DuckDB, avec la sémantique « l'avis a ≥1 claim (dans le thème) portant la
    # stance ». Le sous-arbre est passé comme UN SEUL paramètre liste (`list_contains`) et non
    # comme N placeholders `IN (?,?,…)` : la requête reste de taille constante (un macro a des
    # centaines de descendants) → pas de re-parse/plan proportionnel au sous-arbre.
    use_stance = bool(stance and claim_stance)
    if keep_ids is not None or use_stance:
        sub = []
        if keep_ids is not None:
            sub.append("list_contains(?, filter_theme)")
            params.append(sorted(keep_ids))     # une liste = un paramètre (ordre déterministe)
        if use_stance:
            sub.append("stance = ?")
            params.append(stance)
        conds.append("a.rank IN (SELECT avis_rank FROM claims WHERE "
                     + " AND ".join(sub) + ")")

    if needle is not None:
        conds.append("contains(a.text_fold, ?)")
        params.append(needle)

    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    total = con.execute(f"SELECT count(*) FROM avis a {where}", params).fetchone()[0]

    page_sql = f"SELECT a.key, a.payload FROM avis a {where} ORDER BY a.rank"
    page_params = list(params)
    if limit >= 0:
        page_sql += " LIMIT ? OFFSET ?"
        page_params += [limit, max(0, offset)]
    else:
        page_sql += " OFFSET ?"
        page_params.append(max(0, offset))
    rows = con.execute(page_sql, page_params).fetchall()
    items = [_build_item(key, json.loads(payload)) for key, payload in rows]
    return {"total": total, "items": items}


def build_avis_provenance(tree: ThemeTree,
                          translations: dict[str, dict] | None = None) -> dict[str, dict]:
    """Provenance de TOUS les avis → `{avis_id: {id, text, text_fr, lang, claims}}` (BUILD).

    `translations` : traductions FR précalculées (`backend.translate.build_translations`),
    injectées par avis ; `None` pour un dataset entièrement français.
    """
    claim_macro = _claim_macro(tree)
    claim_leaf = _claim_leaf(tree)
    return {a.id: avis_payload_for(tree, i, claim_macro, translations, claim_leaf)
            for i, a in enumerate(tree.prepared.avis)}

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


def avis_claims(tree: ThemeTree, avis_index: int, claim_macro: dict[int, str]) -> list[dict]:
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
            "color": node.color if node else "",
            "spans": spans,
            "target": target,
            "theme_title": (node.title or node.label) if node else "",
        })
    claims.sort(key=lambda c: (c["spans"][0]["start"], c["spans"][0]["end"]))
    return claims


def avis_payload_for(tree: ThemeTree, avis_index: int,
                     claim_macro: dict[int, str] | None = None,
                     translations: dict[str, dict] | None = None) -> dict:
    """`{id, text, text_fr, lang, claims}` d'un avis (par son index dans `prepared.avis`).

    `translations` (optionnel) : map `avis_id -> {lang, text_fr}` précalculée par
    `backend.translate`. Absente → `lang="fr"`, `text_fr=None` (rétro-compat, datasets FR).
    """
    if claim_macro is None:
        claim_macro = _claim_macro(tree)
    a = tree.prepared.avis[avis_index]
    tr = (translations or {}).get(str(a.id)) or {}
    # `a.text` est le texte CANONIQUE masqué (`text_clean`) sur lequel les spans sont
    # ancrés : on le sert TEL QUEL (cohérence offsets + zéro PII brute, cf. docstring).
    return {"id": a.id, "text": a.text,
            "text_fr": tr.get("text_fr"), "lang": tr.get("lang", "fr"),
            "claims": avis_claims(tree, avis_index, claim_macro)}


def build_avis_provenance(tree: ThemeTree,
                          translations: dict[str, dict] | None = None) -> dict[str, dict]:
    """Provenance de TOUS les avis → `{avis_id: {id, text, text_fr, lang, claims}}` (BUILD).

    `translations` : traductions FR précalculées (`backend.translate.build_translations`),
    injectées par avis ; `None` pour un dataset entièrement français.
    """
    claim_macro = _claim_macro(tree)
    return {a.id: avis_payload_for(tree, i, claim_macro, translations)
            for i, a in enumerate(tree.prepared.avis)}

"""Endpoint `/avis/{id}` — texte d'un avis + ses CLAIMS verbatim surlignables.

Provenance citoyenne : un avis affiché EN ENTIER, avec ses CLAIMS (idées extraites)
surlignés à la couleur de LEUR cluster (macro-thème) — la même couleur que les bulles
de la carte. Un claim peut prendre PLUSIEURS portions non-contiguës (`spans`) et porte
une **cible** verbatim (`target`, une sous-portion ou null). Comme tout est extractif
(sous-chaînes exactes, cf. `pipeline.claims.span`), chaque span pointe une plage réelle
du texte : highlight fidèle, zéro dérive.

    GET /avis/{id} {dataset} -> {
      id, text,
      claims: [ {id, cluster_id, color, spans:[{start,end}], target:{start,end}|null,
                 theme_title} ]
    }

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
                     claim_macro: dict[int, str] | None = None) -> dict:
    """`{id, text, claims}` d'un avis donné (par son index dans `prepared.avis`)."""
    if claim_macro is None:
        claim_macro = _claim_macro(tree)
    a = tree.prepared.avis[avis_index]
    return {"id": a.id, "text": a.text,
            "claims": avis_claims(tree, avis_index, claim_macro)}


def build_avis_provenance(tree: ThemeTree) -> dict[str, dict]:
    """Provenance de TOUS les avis → `{avis_id: {id, text, claims}}` (précalcul BUILD)."""
    claim_macro = _claim_macro(tree)
    return {a.id: avis_payload_for(tree, i, claim_macro)
            for i, a in enumerate(tree.prepared.avis)}

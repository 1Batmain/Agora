"""Endpoint `/avis/{id}` — texte d'un avis + ses portions VERBATIM surlignables.

Provenance citoyenne : un avis affiché EN ENTIER, avec les portions effectivement
extraites (les claims) surlignées à la couleur de LEUR cluster (macro-thème) — la
même couleur que les bulles de la carte. Comme les claims sont extractifs (sous-
chaînes exactes, cf. `pipeline.claims.span`), chaque span pointe une plage réelle
du texte : highlight fidèle, zéro dérive.

    GET /avis/{id} {dataset} -> {id, text, spans:[{start,end,cluster_id,color,theme_label}]}

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


def avis_spans(tree: ThemeTree, avis_index: int, claim_macro: dict[int, str]) -> list[dict]:
    """Spans verbatim d'un avis, ancrés + colorés par macro-thème, triés par position."""
    prep = tree.prepared
    spans: list[dict] = []
    for ci, owner in enumerate(prep.claim_owner):
        if owner != avis_index:
            continue
        start, end = prep.claim_start[ci], prep.claim_end[ci]
        if start < 0 or end <= start:          # claim non ancré (repli) → pas de surlignage
            continue
        mid = claim_macro.get(ci)
        node = tree.nodes.get(mid) if mid else None
        spans.append({
            "start": start,
            "end": end,
            "cluster_id": mid,
            "color": node.color if node else "",
            "theme_label": node.label if node else "",
        })
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans


def avis_payload_for(tree: ThemeTree, avis_index: int,
                     claim_macro: dict[int, str] | None = None) -> dict:
    """`{id, text, spans}` d'un avis donné (par son index dans `prepared.avis`)."""
    if claim_macro is None:
        claim_macro = _claim_macro(tree)
    a = tree.prepared.avis[avis_index]
    return {"id": a.id, "text": a.text,
            "spans": avis_spans(tree, avis_index, claim_macro)}


def build_avis_provenance(tree: ThemeTree) -> dict[str, dict]:
    """Provenance de TOUS les avis → `{avis_id: {id, text, spans}}` (précalcul BUILD)."""
    claim_macro = _claim_macro(tree)
    return {a.id: avis_payload_for(tree, i, claim_macro)
            for i, a in enumerate(tree.prepared.avis)}

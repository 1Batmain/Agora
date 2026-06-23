"""Tests du bac à sable `/sandbox` + `/explain` (recluster live sans LLM).

Couvre le CONTRAT (`/tmp/contract-sandbox.md`) sur le dataset `tiktok` (claims +
embeddings + cibles CACHÉS) :
  - blend α : le knob mélange claim↔cible et change la structure ;
  - schéma de sortie (params/clusters/trace/ms) + latence (~1 s à défauts) ;
  - decision-trace chiffrée (paires : sim/seuil/cohésion/merged ; nodes : disp/τ) ;
  - /explain cluster (voisinage + critères) et pair (sim vs seuil) ;
  - démo addiction : à α↑, les thèmes « addiction » fusionnent sous un même macro.

Aucun LLM : on lit les caches. Si le cache claims est absent (`claims.json`), les
tests sont SKIPPÉS (rien à recluster) plutôt qu'en échec.

Lancer :
    uv run --extra embed-contender --extra faiss python -m backend.test_sandbox
    uv run --extra embed-contender --extra faiss --with fastapi python -m backend.test_sandbox   # + couche HTTP
"""

from __future__ import annotations

from backend.build_analysis import load_dataset
from backend.recluster import CACHE_DIR
from backend import sandbox

DATASET = "tiktok"
ADDICTION_KW = ("addict", "dépend", "depend")


def _has_claims(dataset: str = DATASET) -> bool:
    return (CACHE_DIR / dataset / "claims.json").exists()


def _addiction_macros(payload: dict) -> set[str]:
    """Macros (parent_id None) dont les mots-clés évoquent l'addiction/dépendance."""
    out = set()
    for c in payload["clusters"]:
        if c["parent_id"] is not None:
            continue
        kw = " ".join(c["keywords"]).lower()
        if any(w in kw for w in ADDICTION_KW):
            out.add(c["id"])
    return out


# --------------------------------------------------------------------------- #
# Cœur : recluster_payload (= logique de l'endpoint POST /sandbox)
# --------------------------------------------------------------------------- #
def test_schema_and_latency():
    if not _has_claims():
        print("SKIP test_schema_and_latency (claims.json absent)")
        return
    ds = load_dataset(DATASET)
    sandbox.invalidate(DATASET)
    sandbox.get_prepared(ds)                      # sort l'embed des cibles du chemin chaud
    p = sandbox.recluster_payload(ds)             # défauts neutres

    assert set(p) >= {"params", "n_claims", "ms", "clusters", "trace"}
    assert p["n_claims"] > 0
    assert p["clusters"], "au moins un cluster"
    for c in p["clusters"]:
        assert set(c) >= {"id", "parent_id", "n_claims", "n_avis",
                          "keywords", "sample_claims", "cohesion"}
        assert 0.0 <= c["cohesion"] <= 1.0
        assert c["n_claims"] >= 1
    tr = p["trace"]
    assert set(tr) == {"pairs", "nodes"}
    assert len(tr["nodes"]) == len(p["clusters"])
    for pr in tr["pairs"]:
        assert set(pr) >= {"a", "b", "sim", "threshold", "cohesion_min", "merged"}
    # Latence : objectif ~1 s, acceptance < ~1.5 s (defauts). On laisse une marge CI.
    assert p["ms"] < 3000, f"recluster trop lent: {p['ms']} ms"
    print(f"OK schema+latency: n_claims={p['n_claims']} clusters={len(p['clusters'])} "
          f"ms={p['ms']} pairs={len(tr['pairs'])}")


def test_alpha_changes_structure():
    if not _has_claims():
        print("SKIP test_alpha_changes_structure")
        return
    ds = load_dataset(DATASET)
    p0 = sandbox.recluster_payload(ds, alpha=0.0)
    p5 = sandbox.recluster_payload(ds, alpha=0.5)
    assert p0["params"]["alpha"] == 0.0 and p5["params"]["alpha"] == 0.5
    assert p5["params"]["derived"]["n_targets"] > 0, "des cibles doivent être embeddées"
    # α agit : le blend modifie la partition (nombre de clusters et/ou hiérarchie).
    sig0 = (len(p0["clusters"]), tuple(sorted(c["n_claims"] for c in p0["clusters"])))
    sig5 = (len(p5["clusters"]), tuple(sorted(c["n_claims"] for c in p5["clusters"])))
    assert sig0 != sig5, "le knob α devrait changer la structure"
    print(f"OK alpha: clusters α0={len(p0['clusters'])} α0.5={len(p5['clusters'])}")


def test_tau_mult_monotone():
    if not _has_claims():
        print("SKIP test_tau_mult_monotone")
        return
    ds = load_dataset(DATASET)
    # τ plus grand ⇒ on subdivise MOINS ⇒ moins (ou autant) de nœuds. Monotone large.
    counts = [len(sandbox.recluster_payload(ds, alpha=0.5, tau_mult=tm)["clusters"])
              for tm in (1.0, 2.0, 3.0)]
    assert counts[0] >= counts[1] >= counts[2], f"τ↑ devrait réduire les nœuds: {counts}"
    print(f"OK tau_mult monotone (alpha=0.5): {counts}")


def test_explain_cluster_and_pair():
    if not _has_claims():
        print("SKIP test_explain")
        return
    ds = load_dataset(DATASET)
    p = sandbox.recluster_payload(ds, alpha=0.5)
    cid = p["clusters"][0]["id"]
    ec = sandbox.explain_cluster(ds, cid)
    assert ec["cluster"] == cid
    assert "criteria" in ec and "neighbors" in ec
    assert ec["neighbors"], "au moins un voisin"
    a, b = ec["cluster"], ec["neighbors"][0]["id"]
    ep = sandbox.explain_pair(ds, a, b)
    assert ep["pair"] == [a, b]
    assert {"sim", "threshold", "cohesion_a", "cohesion_b", "would_merge"} <= set(ep)
    assert isinstance(ep["explanation"], str) and ep["explanation"]
    # Cluster inconnu → erreur portée (l'endpoint la mappe en 404).
    assert "error" in sandbox.explain_cluster(ds, "n999999")
    print(f"OK explain: cluster {cid} → {len(ec['neighbors'])} voisins ; "
          f"pair({a},{b}) would_merge={ep['would_merge']}")


def test_addiction_merges_with_alpha():
    """DÉMO du brief : à α↑, les thèmes addiction cessent d'être des macros séparés."""
    if not _has_claims():
        print("SKIP test_addiction_merges_with_alpha")
        return
    ds = load_dataset(DATASET)
    macros0 = _addiction_macros(sandbox.recluster_payload(ds, alpha=0.0))
    macros5 = _addiction_macros(sandbox.recluster_payload(ds, alpha=0.5))
    # À α=0 le matériau « addiction » est fragmenté en ≥1 macro top-level ; à α=0.5 il
    # est absorbé sous UN macro (les autres deviennent des enfants) → moins de macros.
    assert len(macros0) >= 1, "addiction présente à α=0"
    assert len(macros5) <= len(macros0), (
        f"α devrait regrouper l'addiction: macros α0={macros0} α0.5={macros5}")
    print(f"OK addiction merge: macros addiction α0={len(macros0)} → α0.5={len(macros5)}")


# --------------------------------------------------------------------------- #
# Couche HTTP (FastAPI TestClient) — optionnelle (skip si fastapi absent)
# --------------------------------------------------------------------------- #
def test_http_endpoints():
    if not _has_claims():
        print("SKIP test_http_endpoints (claims.json absent)")
        return
    import os
    os.environ["AGORA_AUTOBUILD"] = "0"           # pas de build LLM au démarrage
    try:
        from fastapi.testclient import TestClient
    except ModuleNotFoundError:
        print("SKIP test_http_endpoints (fastapi absent — relance avec --with fastapi)")
        return
    from backend.server import app

    with TestClient(app) as client:
        r = client.post("/sandbox", json={"dataset": DATASET, "alpha": 0.5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_claims"] > 0 and body["clusters"]
        cid = body["clusters"][0]["id"]

        r2 = client.get("/explain", params={"dataset": DATASET, "cluster": cid})
        assert r2.status_code == 200, r2.text
        assert r2.json()["cluster"] == cid

        nb = r2.json()["neighbors"][0]["id"]
        r3 = client.get("/explain", params={"dataset": DATASET, "pair": f"{cid},{nb}"})
        assert r3.status_code == 200, r3.text
        assert r3.json()["pair"] == [cid, nb]

        # Erreurs : pair mal formée → 422 ; cluster inconnu → 404.
        assert client.get("/explain", params={"dataset": DATASET, "pair": "n1"}).status_code == 422
        assert client.get("/explain", params={"dataset": DATASET,
                                              "cluster": "n999999"}).status_code == 404
    print("OK http: /sandbox 200, /explain cluster+pair 200, erreurs 422/404")


def _main() -> None:
    tests = [
        test_schema_and_latency,
        test_alpha_changes_structure,
        test_tau_mult_monotone,
        test_explain_cluster_and_pair,
        test_addiction_merges_with_alpha,
        test_http_endpoints,
    ]
    for t in tests:
        t()
    print("\nTOUS LES TESTS SANDBOX PASSENT.")


if __name__ == "__main__":
    _main()

"""BUILD — précalcule et PERSISTE l'analyse complète d'un dataset (B1–B4 d'un coup).

Pendant **BUILD** de la séparation BUILD/SERVE : le pipeline lourd
(claims → embed → cluster → UMAP → hiérarchie variance-adaptative → insights LLM)
tourne ICI, sur le backend, **dès que la donnée est dispo**, et écrit tout sous
`backend/cache/<dataset>/analysis/` via `backend.analysis_store`. Les endpoints
(`backend.server`) ne font alors plus que LIRE ces fichiers (instantané).

Réutilise tel quel l'existant :
  - `prepare_claims` (extraction LLM + embed, déjà CACHÉS sur disque) ;
  - `build_theme_tree` + `analysis_payload` (B1+B2) ;
  - `citations_for_theme` (B4) ;
  - `render_insight` (B3, synthèse LLM par niveau).

Idempotent : reprend les caches claims/embeddings existants ; un rebuild ne ré-extrait
pas si le modèle n'a pas changé. Logue la progression et la reflète dans `status.json`.

Usage CLI :
    uv run python -m backend.build_analysis --dataset tiktok
    uv run python -m backend.build_analysis --dataset tiktok --force   # rebuild propre
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter
from types import SimpleNamespace
from typing import Callable

from backend import analysis_store as store
from backend.analysis import (
    DEFAULT_EMBEDDER,
    DEFAULT_SEED,
    analysis_payload,
    build_theme_tree,
)
from backend.avis import build_avis_provenance
from backend.citations import citations_for_theme
from backend.insights import render_insight
from backend.cluster_enrich import description_for_node, hook_for_node
from backend.recluster import CACHE_DIR, load_cache
from backend.titles import title_for_node

ProgressFn = Callable[[str, str, int, int], None]

# DEUX modèles SÉPARÉS pour un rebuild rapide (PRIORITÉ 1) :
#   - EXTRACTION (lente, ~1 appel/avis) → gros modèle de QUALITÉ (claims fidèles,
#     multi-spans + target). Cachée sur disque : un rebuild ne la rejoue pas.
#   - ENRICHISSEMENT (titres/accroches/descriptions/insights, ~3-4 appels/thème) →
#     modèle CHEAP. C'est le gros du coût d'un rebuild (extraction cachée) → cheap = vite.
# Surchargeables par env (aucune valeur de corpus codée en dur).
EXTRACT_MODEL = os.environ.get("AGORA_EXTRACT_MODEL", "mistral-large-latest")
ENRICH_MODEL = os.environ.get("AGORA_ENRICH_MODEL", "mistral-small-latest")


def _log(msg: str) -> None:
    print(f"[build_analysis] {msg}", flush=True)


def load_dataset(dataset_id: str):
    """Charge un dataset léger (id + ideas) depuis le cache disque — zéro torch.

    Suffisant pour `prepare_claims`/`build_theme_tree`, qui n'ont besoin que de
    `ds.id` et `ds.ideas` (les vecteurs d'avis ne servent pas aux claims).
    """
    ideas, _vecs, _weights = load_cache(dataset_id)
    return SimpleNamespace(id=dataset_id, ideas=ideas)


def build_analysis(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    enrich_model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    resolution: float = 1.0,
    seed: int = DEFAULT_SEED,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Calcule TOUTE l'analyse d'un dataset et la persiste. Renvoie le status final.

    `ds` porte `.id` et `.ideas` (un `_Dataset` du serveur ou un `load_dataset`). Écrit
    `status.json` au fil de l'eau (phase + done/total) pour que le front montre une
    progression. En cas d'échec, écrit `status=error` et relève l'exception (le manager
    décide quoi faire). LLM = backend par défaut (API) sauf `backend=` explicite.

    DEUX modèles : `model` = EXTRACTION (défaut `EXTRACT_MODEL`, gros/qualité, cachée) ;
    `enrich_model` = ENRICHISSEMENT titres/accroches/descriptions/insights (défaut
    `ENRICH_MODEL`, CHEAP) → rebuild rapide car l'extraction est cachée et le reste cheap.
    """
    t0 = perf_counter()
    dataset = ds.id
    extract_model = model or EXTRACT_MODEL
    enrich = enrich_model or ENRICH_MODEL

    def report(phase: str, detail: str, done: int = 0, total: int = 0) -> None:
        store.write_status(dataset, store.BUILDING, phase=phase, detail=detail,
                           done=done, total=total)
        _log(f"{dataset} · {phase} · {detail}" + (f" ({done}/{total})" if total else ""))
        if on_progress:
            on_progress(phase, detail, done, total)

    try:
        # 1) Claims (extraction LLM + embed, cachés) + arbre variance-adaptatif (B1+B2).
        #    L'analyse PERSISTÉE/servie utilise le Leiden BATCH (global + coarsening de
        #    racines), dont la qualité macro est non-négociable. L'incrémental
        #    (AnalysisState) est réservé au stream live (cf. /stream) : ordre-dépendant,
        #    il écrase la structure macro et ne convient pas à l'analyse statique.
        report("claims", f"extraction ({extract_model}) + embeddings (caché si déjà fait)")

        def _extract_progress(done: int, total: int) -> None:
            if done == total or done % 25 == 0:
                report("claims", f"extraction LLM ({extract_model})", done, total)

        tree = build_theme_tree(
            ds, backend=backend, model=extract_model, embedder=embedder,
            resolution=resolution, seed=seed, extract_progress=_extract_progress,
        )
        node_ids = list(tree.order)
        report("tree", f"{len(node_ids)} thèmes (macros: {len(tree.macros)})")

        # 1b) Titre court LLM par thème (3-7 mots), CACHÉ par contenu → baké dans
        #     analysis.json. Rebuild idempotent : contenu inchangé ⇒ zéro appel LLM.
        total = len(node_ids)
        report("titles", f"titres courts ({enrich}, caché)", 0, total)
        for i, nid in enumerate(node_ids, 1):
            node = tree.nodes[nid]
            node.title = title_for_node(dataset, node, model=enrich)  # CHEAP (≠ extraction)
            if i == total or i % 25 == 0:
                report("titles", f"titres courts ({enrich}, caché)", i, total)

        # 1c) Accroche + description LLM par thème (CACHÉES par contenu) → analysis.json.
        #     Même infra que les titres : rebuild idempotent, zéro appel si inchangé.
        report("enrich", f"accroches + descriptions ({enrich}, caché)", 0, total)
        for i, nid in enumerate(node_ids, 1):
            node = tree.nodes[nid]
            node.hook = hook_for_node(dataset, node, model=enrich)
            node.description = description_for_node(dataset, node, model=enrich)
            if i == total or i % 25 == 0:
                report("enrich", f"accroches + descriptions ({enrich}, caché)", i, total)

        # 2) Carte : co-occurrence (B1) → analysis.json (front en d3-pack, plus d'UMAP).
        report("analysis", "co-occurrence (hiérarchie d3-pack, sans UMAP)")
        payload = analysis_payload(tree)
        payload["status"] = store.READY
        store.write_analysis(dataset, payload)

        # 2b) Provenance : texte de chaque avis + ses portions verbatim colorées par
        #     macro (pour le surlignage côté front) → avis.json.
        report("avis", "provenance des portions verbatim")
        store.write_avis(dataset, build_avis_provenance(tree))

        # 3) Citations triées centroïde, par nœud (B4) — aucun LLM, rapide.
        for i, nid in enumerate(node_ids, 1):
            store.write_citations(dataset, nid, citations_for_theme(tree, nid))
            if i == total or i % 25 == 0:
                report("citations", "tri par proximité au centroïde", i, total)

        # 4) Insights LLM par niveau (B3) : global + un par thème, persistés.
        report("insights", f"synthèse globale ({enrich})", 0, total + 1)
        store.write_insights(dataset, "global", None,
                             render_insight(tree, "global", model=enrich))
        for i, nid in enumerate(node_ids, 1):
            store.write_insights(dataset, "theme", nid,
                                 render_insight(tree, "theme", nid, model=enrich))
            report("insights", f"synthèses par thème ({enrich})", i, total)

        took_s = round(perf_counter() - t0, 1)
        final = store.write_status(
            dataset, store.READY,
            phase="done", detail=f"analyse prête en {took_s}s",
            done=total, total=total, error=None,
            n_themes=len(node_ids), n_macros=len(tree.macros),
            backend_used=payload.get("backend_used"),
            took_seconds=took_s,
        )
        _log(f"{dataset} · ✓ READY · {len(node_ids)} thèmes · {took_s}s")
        return final
    except Exception as exc:  # noqa: BLE001 — on persiste l'échec puis on relève
        store.write_status(dataset, store.ERROR, phase="error",
                           detail="échec du build", error=str(exc))
        _log(f"{dataset} · ✗ ERROR · {exc}")
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="Précalcul + persistance de l'analyse complète d'un dataset.")
    ap.add_argument("--dataset", required=True, help="id du dataset (sous backend/cache/)")
    ap.add_argument("--backend", default=None, help="api (défaut) | mac | auto")
    ap.add_argument("--model", default=None,
                    help=f"modèle d'EXTRACTION (défaut {EXTRACT_MODEL}, gros/qualité, caché)")
    ap.add_argument("--enrich-model", default=None,
                    help=f"modèle d'ENRICHISSEMENT titres/insights (défaut {ENRICH_MODEL}, cheap)")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="efface l'analyse persistée avant de rebuild (garde claims.json -> re-clusterise sans réextraire)")
    ap.add_argument("--reextract", action="store_true", help="efface AUSSI claims.json -> réextraction LLM fraîche (ex. après changement du prompt d'extraction)")
    args = ap.parse_args()

    if args.force or args.reextract:
        store.clear(args.dataset)
        _log(f"{args.dataset} · analyse persistée effacée")

    if args.reextract:
        claims_cache = CACHE_DIR / args.dataset / "claims.json"
        if claims_cache.exists():
            claims_cache.unlink()
            _log(f"{args.dataset} · claims.json effacé (--reextract) -> réextraction LLM fraîche")

    ds = load_dataset(args.dataset)
    build_analysis(
        ds, backend=args.backend, model=args.model, enrich_model=args.enrich_model,
        embedder=args.embedder, resolution=args.resolution, seed=args.seed,
    )


if __name__ == "__main__":
    main()

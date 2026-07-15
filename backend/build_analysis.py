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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from types import SimpleNamespace
from typing import Callable

from backend import analysis_store as store
from backend.analysis import (
    DEFAULT_EMBEDDER,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    analysis_payload,
    build_theme_tree,
)
from backend.avis import build_avis_provenance
from backend.translate import build_translations
from backend.keywords_fr import translate_tree_keywords
from backend.citations import citations_for_theme
from backend.insights import render_insight
from backend.cluster_enrich import description_for_node, hook_for_node
from backend.recluster import CACHE_DIR, load_cache
from backend.titles import title_for_node
from backend import cost
from pipeline.cluster import mistral_client

ProgressFn = Callable[[str, str, int, int], None]

# DEUX modèles SÉPARÉS pour un rebuild rapide (PRIORITÉ 1) :
#   - EXTRACTION (lente, ~1 appel/avis) → gros modèle de QUALITÉ (claims fidèles,
#     multi-spans + target). Cachée sur disque : un rebuild ne la rejoue pas.
#   - ENRICHISSEMENT (titres/accroches/descriptions/insights, ~3-4 appels/thème) →
#     modèle CHEAP. C'est le gros du coût d'un rebuild (extraction cachée) → cheap = vite.
# Surchargeables par env (aucune valeur de corpus codée en dur).
EXTRACT_MODEL = os.environ.get("AGORA_EXTRACT_MODEL", "mistral-large-latest")
# Laisse passer un arbre entièrement plat (aucun macro subdivisé). Fail-closed par défaut :
# la hiérarchie est le produit, pas un bonus (cf. `_assert_tree_is_structured`).
ALLOW_FLAT_TREE = os.environ.get("AGORA_ALLOW_FLAT_TREE", "").strip() == "1"
# Part MAXIMALE de titres/synthèses en repli avant de refuser de servir le build.
MAX_FALLBACK_SHARE = float(os.environ.get("AGORA_MAX_FALLBACK_SHARE", "0.05"))
ALLOW_DEGRADED = os.environ.get("AGORA_ALLOW_DEGRADED", "").strip() == "1"
ENRICH_MODEL = os.environ.get("AGORA_ENRICH_MODEL", "mistral-large-latest")

# Concurrence BORNÉE des appels LLM d'enrichissement (titres/accroches/descriptions/
# insights). Chaque thème est indépendant (effet de bord sur SON nœud, cache idempotent)
# → on lance jusqu'à N appels en parallèle. La borne respecte le RPM (pas de tempête 429)
# et garde le backoff existant. Résultats IDENTIQUES au chemin sériel (ordre indifférent).
LLM_MAX_WORKERS = max(1, int(os.environ.get("AGORA_LLM_MAX_WORKERS", "4")))


def _log(msg: str) -> None:
    print(f"[build_analysis] {msg}", flush=True)


def _parallel_for(
    items: list,
    work: Callable[[object], None],
    *,
    on_done: Callable[[int], None] | None = None,
    workers: int = LLM_MAX_WORKERS,
) -> None:
    """Exécute `work(item)` pour chaque item, jusqu'à `workers` en parallèle (borné).

    `work` produit son résultat par EFFET DE BORD sur l'item (p.ex. `node.title = …`),
    donc l'ordre d'exécution est indifférent → résultats identiques au sériel. `on_done(k)`
    est appelé après chaque tâche terminée avec le nombre cumulé d'items finis (progression).
    Séquentiel si `workers<=1` ou un seul item. Les exceptions sont propagées (fail-fast).
    """
    total = len(items)
    if total == 0:
        return
    if workers <= 1 or total == 1:
        for i, it in enumerate(items, 1):
            work(it)
            if on_done is not None:
                on_done(i)
        return
    done = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agora-llm") as ex:
        futures = [ex.submit(work, it) for it in items]
        for fut in as_completed(futures):
            fut.result()  # propage toute exception du worker
            if on_done is not None:
                with lock:
                    done += 1
                    k = done
                on_done(k)


class FlatTreeError(RuntimeError):
    """L'arbre servi n'a AUCUN sous-thème alors que le corpus est multi-macro."""


def _assert_tree_is_structured(tree) -> None:
    """Refuse un arbre entièrement plat — la hiérarchie EST le produit.

    Le produit promet « thèmes → sous-thèmes → verbatim » : un arbre à profondeur 0 sur
    un corpus multi-macro est un échec de build, pas un résultat. On lève AVANT
    l'enrichissement LLM (donc avant la dépense). Un corpus réellement mono-facette a
    <3 macros et passe.

    Historique : le rebuild tiktok du 2026-07-08 a servi un arbre plat avec un
    `status: ready` serein, parce que le seuil de dispersion `tau` s'était calé au-dessus
    de toutes les dispersions. `tau` a depuis été supprimé (`.agent/notes/HIERARCHY_TAU.md`)
    — ce garde-fou reste : il protège contre la PROCHAINE cause d'aplatissement.
    """
    macros = list(tree.macros)
    if len(macros) < 3:
        return                                   # trop peu de macros : platitude légitime
    structured = sum(1 for m in macros if tree.nodes[m].children)
    if structured:
        return
    tailles = sorted(tree.nodes[m].n_claims for m in macros)
    if ALLOW_FLAT_TREE:
        print(f"⚠️  arbre PLAT toléré (AGORA_ALLOW_FLAT_TREE=1) : {len(macros)} macros, "
              f"aucun sous-thème")
        return
    raise FlatTreeError(
        f"{len(macros)} macros, AUCUN avec sous-thèmes → arbre plat (macros ≡ thèmes fins).\n"
        f"  tailles des macros, en claims = {tailles}\n"
        f"  → la chaîne d'emboîtement n'a pas dégagé de couche macro au-dessus des thèmes fins.\n"
        f"Vérifier la chaîne (pipeline/cluster/layers.py) et `resolution` avant de servir.\n"
        f"AGORA_ALLOW_FLAT_TREE=1 pour passer outre en connaissance de cause."
    )


class DegradedEnrichmentError(RuntimeError):
    """Trop de titres/synthèses sont des REPLIS : le build est dégradé, pas prêt."""


def _assert_enrichment_is_complete(dataset: str, tree, node_ids: list[str]) -> None:
    """Refuse de servir un build dont l'enrichissement LLM s'est effondré en silence.

    Un 429 (quota Mistral atteint) ne lève pas : chaque appelant retombe sur son repli —
    titre en mots-clés, synthèse jamais écrite — et le build se déclarait `ready`. Vécu
    deux fois : le « bug des 257 titres-labels », puis un rebuild tiktok à 27 % de titres
    en mots-clés et 37 % de synthèses absentes. `mistral_client` réessaie désormais ; ce
    garde-fou constate ce qui a malgré tout été perdu.

    Le repli reste LÉGITIME à petite dose (un thème minuscule, un texte pauvre) : on ne
    lève qu'au-delà du seuil, et `AGORA_ALLOW_DEGRADED=1` passe outre en connaissance de
    cause (utile pour un build hors-ligne, sans clé).
    """
    exhausted = mistral_client.get_exhausted()
    n = len(node_ids) or 1
    kw_titles = [i for i in node_ids if " · " in (tree.nodes[i].title or "")]
    missing_insights = [i for i in node_ids if not store.read_insights(dataset, "theme", i)]

    part_titles = len(kw_titles) / n
    part_insights = len(missing_insights) / n
    if exhausted["count"]:
        _log(f"{dataset} · ⚠️  {exhausted['count']} appels LLM perdus après réessais "
             f"{exhausted['by_status']} (429 = quota Mistral)")

    if part_titles <= MAX_FALLBACK_SHARE and part_insights <= MAX_FALLBACK_SHARE:
        return
    msg = (
        f"enrichissement DÉGRADÉ sur {dataset} :\n"
        f"  titres en repli mots-clés : {len(kw_titles)}/{n} ({part_titles:.0%})\n"
        f"  synthèses absentes        : {len(missing_insights)}/{n} ({part_insights:.0%})\n"
        f"  appels LLM perdus         : {exhausted['count']} {exhausted['by_status'] or ''}\n"
        f"  seuil toléré              : {MAX_FALLBACK_SHARE:.0%}\n"
        f"Cause la plus fréquente : quota Mistral atteint (429) en cours de build.\n"
        f"Recharger le crédit puis RELANCER — les replis ne sont pas cachés, ils se\n"
        f"régénèrent. AGORA_ALLOW_DEGRADED=1 pour servir quand même."
    )
    if ALLOW_DEGRADED:
        _log(f"{dataset} · ⚠️  {msg}")
        return
    raise DegradedEnrichmentError(msg)


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
    resolution: float = DEFAULT_RESOLUTION,
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
        mistral_client.reset_usage()  # suivi tokens/coût Mistral de CE build (tout passe par chat())
        # 1) Claims (extraction LLM + embed, cachés) + arbre variance-adaptatif (B1+B2).
        #    L'analyse PERSISTÉE/servie utilise le Leiden BATCH (global + coarsening de
        #    racines), dont la qualité macro est non-négociable.
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
        _assert_tree_is_structured(tree)

        # 1a-bis) Mots-clés en FRANÇAIS (datasets multilingues) : on traduit les TERMES
        #     c-TF-IDF non-FR AU BUILD (caché, batché), AVANT que les titres/accroches/
        #     descriptions LLM et le payload persisté ne les lisent. Mono-FR → no-op.
        lang_of = {str(getattr(it, "id", "")): getattr(it, "lang", None)
                   for it in getattr(ds, "ideas", []) or []}
        report("keywords_fr", "traduction FR des mots-clés non-français (caché)")
        kw_map = translate_tree_keywords(
            dataset, tree, lang_of,
            on_progress=lambda d, t: report("keywords_fr", "traduction FR des mots-clés (caché)", d, t),
        )
        if kw_map:
            report("keywords_fr", f"{len(kw_map)} termes traduits en français")

        # 1b) Titre court LLM par thème (3-7 mots), CACHÉ par contenu → baké dans
        #     analysis.json. Rebuild idempotent : contenu inchangé ⇒ zéro appel LLM.
        total = len(node_ids)
        report("titles", f"titres courts ({enrich}, caché)", 0, total)

        def _title_work(nid: str) -> None:
            node = tree.nodes[nid]
            node.title = title_for_node(dataset, node, model=enrich)  # CHEAP (≠ extraction)

        def _title_done(k: int) -> None:
            if k == total or k % 25 == 0:
                report("titles", f"titres courts ({enrich}, caché)", k, total)

        _parallel_for(node_ids, _title_work, on_done=_title_done)

        # 1c) Accroche + description LLM par thème (CACHÉES par contenu) → analysis.json.
        #     Même infra que les titres : rebuild idempotent, zéro appel si inchangé.
        report("enrich", f"accroches + descriptions ({enrich}, caché)", 0, total)

        def _enrich_work(nid: str) -> None:
            node = tree.nodes[nid]
            node.hook = hook_for_node(dataset, node, model=enrich)
            node.description = description_for_node(dataset, node, model=enrich)

        def _enrich_done(k: int) -> None:
            if k == total or k % 25 == 0:
                report("enrich", f"accroches + descriptions ({enrich}, caché)", k, total)

        _parallel_for(node_ids, _enrich_work, on_done=_enrich_done)

        # 2) Carte : co-occurrence (B1) → analysis.json (front en d3-pack, plus d'UMAP).
        report("analysis", "co-occurrence (hiérarchie d3-pack, sans UMAP)")
        payload = analysis_payload(tree)
        payload["status"] = store.READY
        store.write_analysis(dataset, payload)

        # 2a-bis) Traduction des avis non-FR en français (CHEAP, batché, CACHÉE &
        #     idempotente) → translations.json. Le front affiche le FR par défaut, avec
        #     « voir l'original » (surlignages sur l'original). Datasets FR : rien à faire.
        report("translate", "traduction FR des avis non-français (caché)")
        translations = build_translations(  # `lang_of` calculé en 1a-bis (réutilisé)
            dataset, tree.prepared.avis, lang_of,
            on_progress=lambda d, t: report("translate", "traduction FR (caché)", d, t),
        )

        # 2b) Provenance : texte de chaque avis (+ traduction FR/langue) + ses portions
        #     verbatim colorées par macro (pour le surlignage côté front) → avis.json.
        report("avis", "provenance des portions verbatim")
        store.write_avis(dataset, build_avis_provenance(tree, translations))

        # 3) Citations triées centroïde, par nœud (B4) — aucun LLM, rapide.
        for i, nid in enumerate(node_ids, 1):
            store.write_citations(dataset, nid, citations_for_theme(tree, nid))
            if i == total or i % 25 == 0:
                report("citations", "tri par proximité au centroïde", i, total)

        # 4) Insights LLM par niveau (B3) : global + un par thème, persistés.
        # BOTTOM-UP : les thèmes sont générés par PROFONDEUR DÉCROISSANTE (feuilles →
        # racines). La synthèse d'un thème PARENT AGRÈGE les synthèses déjà rédigées de
        # ses sous-thèmes (`insights_md`, cf. backend.insights) ; une feuille part de ses
        # propres claims. Le global reste bâti depuis les macro-thèmes (inchangé).
        # Garde-fou : on N'ÉCRIT PAS un insight de REPLI (Mistral indispo/erreur) → le `.json`
        # reste absent (`/insights` → 404 gracieux) et un re-bake ultérieur le régénère, au lieu
        # de FIGER un message d'erreur en cache et de le servir. Un enfant en repli n'entre
        # pas dans `insights_md` → le parent retombe gracieusement sur ses claims.
        insights_md: dict[str, str] = {}
        md_lock = threading.Lock()

        # OPINION (si déjà bakée) → nourrit la section « À relever » (tensions/consensus)
        # du harness d'insights. Graceful si absente (repli sur les claims). L'ordre
        # opinion→insights est assuré par l'orchestration du rebuild complet : bake
        # l'opinion AVANT ce build (ou re-lance ce build après build_opinion — idempotent).
        op = store.read_opinion(dataset) or {}
        opinions_by_id: dict[str, dict] = {o["theme_id"]: o for o in op.get("themes", [])}
        if opinions_by_id:
            _log(f"{dataset} · opinion chargée ({len(opinions_by_id)} thèmes) → « À relever »")

        def _write_insight(level: str, nid: str | None) -> None:
            payload = render_insight(
                tree, level, nid, model=enrich,
                child_insights=insights_md if level == "theme" else None,
                opinion=opinions_by_id.get(nid) if level == "theme" else None,
            )
            if (payload.get("meta") or {}).get("fallback"):
                return
            store.write_insights(dataset, level, nid, payload)
            if level == "theme" and nid is not None:
                with md_lock:  # dispo pour le PARENT (bande de profondeur suivante)
                    insights_md[nid] = payload.get("markdown", "")

        report("insights", f"synthèse globale ({enrich})", 0, total + 1)
        _write_insight("global", None)

        # Bandes de profondeur, des feuilles (profondeur max) vers les racines. Chaque
        # bande est un BARRAGE : quand un parent est synthétisé, TOUS ses enfants (plus
        # profonds) sont déjà dans `insights_md`. Parallélisme intra-bande conservé (les
        # nœuds d'une même profondeur sont indépendants — jamais parent↔enfant entre eux).
        by_depth: dict[int, list[str]] = {}
        for nid in node_ids:
            by_depth.setdefault(tree.nodes[nid].depth, []).append(nid)
        done = 0
        for depth in sorted(by_depth, reverse=True):
            band = by_depth[depth]

            def _insight_done(k: int, _base: int = done) -> None:
                seen = _base + k
                if seen == total or seen % 25 == 0:
                    report("insights", f"synthèses par thème ({enrich})", seen, total)

            _parallel_for(band, lambda nid: _write_insight("theme", nid),
                          on_done=_insight_done)
            done += len(band)

        # Coût LLM de ce build (extraction + nommage + enrichissement + insights).
        try:
            cost.record_phase(dataset, "analysis", mistral_client.get_usage(),
                              duration_seconds=perf_counter() - t0)
        except Exception as _e:  # le coût est un bonus, jamais bloquant
            _log(f"{dataset} · (coût non enregistré: {_e})")

        _assert_enrichment_is_complete(dataset, tree, node_ids)

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
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
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

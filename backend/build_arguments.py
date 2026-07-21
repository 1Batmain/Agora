"""BUILD ARGUMENTS — mine les ARGUMENTS principaux par thème (pour / contre / neutre),
chacun SOURCÉ sur des contributions réelles, et persiste `analysis/arguments.json`.

Approche « V-SELECT » (sélection verbatim, fail-closed) — VALIDÉE research/argmine_verbatim_note.md
(bat la paraphrase servie au panel aveugle, 71% tête-à-tête) :
  1. par (feuille × stance), le LLM SÉLECTIONNE ≤ k contributions représentatives d'idées
     DISTINCTES — il ne RÉDIGE rien, il renvoie des NUMÉROS (`json_mode` caché par contenu).
     Le texte d'argument servi EST le claim choisi → VERBATIM par construction (join de
     sous-chaînes exactes de l'avis), invariant Agora respecté (fini la paraphrase inventée) ;
  2. le claim sélectionné (son vecteur RAW, MÊME espace nomic-v2) re-source par cosinus les
     claims du groupe (argmax EXCLUSIF → comptes disjoints) — l'ablation a écarté tout
     enrichissement cible-référencé (research/argmine_extract_note.md) ;
  3. un argument qui ne rassemble pas `MIN_SUPPORT` claims au-dessus du seuil est SUPPRIMÉ :
     aucun argument servi sans exemples verbatim réels derrière.

La partition pour/contre vient de `claim_stance.json` (baké par `build_opinion`) ; les
feuilles sans clivage tombent en mode « neutre » (tous les claims). Les parents sont
agrégés par fusion d'embeddings des arguments de leurs feuilles (zéro LLM en plus).

Artefact À PART et OPTIONNEL : ce build LIT les caches existants (zéro ré-extraction),
n'écrit QUE `arguments.json` (+ son cache LLM `arguments_llm/`) — les datasets déjà
analysés n'en ont pas et rien ne casse sans lui (contrat de rétro-compat).

Usage CLI :
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python -m backend.build_arguments --dataset granddebat
    # ou sur GPU local (vLLM offline) :
    PYTHONPATH=. python -m pipeline.cluster.local_llm_offline --model <hf-id> \
        build_arguments --dataset <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from backend import analysis_store as store
from backend.analysis import (
    DEFAULT_EMBEDDER,
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    ThemeNode,
    ThemeTree,
    build_theme_tree,
)
from backend.build_analysis import EXTRACT_MODEL, load_dataset
from backend.build_opinion import _leaf_claims
from backend.llm_cache import cached_llm
from pipeline.cluster import mistral_client

# Modèle CHEAP (1 appel par feuille×stance) — surchargeable, même chaîne de repli
# que l'enrichissement.
MODEL = os.environ.get(
    "AGORA_ARGMINE_MODEL", os.environ.get("AGORA_ENRICH_MODEL", "mistral-small-latest")
)
MAX_K = int(os.environ.get("AGORA_ARG_MAX_K", "5"))                # arguments max / groupe
INPUT_CAP = int(os.environ.get("AGORA_ARG_INPUT_CAP", "60"))       # claims montrés au LLM
SIM_THRESHOLD = float(os.environ.get("AGORA_ARG_SIM_THRESHOLD", "0.60"))
MIN_SUPPORT = int(os.environ.get("AGORA_ARG_MIN_SUPPORT", "3"))    # fail-closed
DEDUP_THRESHOLD = float(os.environ.get("AGORA_ARG_DEDUP", "0.85"))
PARENT_MAX = int(os.environ.get("AGORA_ARG_PARENT_MAX", "7"))
TOP_SOURCES = 5
DEBUG = os.environ.get("AGORA_ARG_DEBUG", "0") == "1"              # stats de calibration
LLM_MAX_WORKERS = max(1, int(os.environ.get("AGORA_LLM_MAX_WORKERS", "4")))

# Libellés externes des groupes (le vocabulaire interne de claim_stance est
# favorable/defavorable ; on sert pour/contre, cf. slot OpinionBar).
_STANCE_LABELS = {"favorable": "pour", "defavorable": "contre"}

ProgressFn = Callable[[str, int, int], None]


def _log(msg: str) -> None:
    print(f"[build_arguments] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Prompts (json_mode) — fidélité stricte, rien d'inventé, langue des contributions.
# --------------------------------------------------------------------------- #
def _select_system(stance: str, max_k: int) -> str:
    """Prompt de SÉLECTION (V-SELECT) — le LLM CHOISIT des contributions, ne RÉDIGE rien.

    VALIDÉ research/argmine_verbatim_note.md : servir un span VERBATIM (une contribution
    réelle) bat la paraphrase servie au panel aveugle (71% tête-à-tête). L'invariant verbatim
    est garanti par CONSTRUCTION — le texte servi EST un claim (join de sous-chaînes d'avis).
    """
    camp = {"pour": " (toutes FAVORABLES à la proposition)",
            "contre": " (toutes DÉFAVORABLES à la proposition)"}.get(stance, "")
    return (
        "Tu es analyste de consultations citoyennes. On te donne des CONTRIBUTIONS citoyennes "
        f"NUMÉROTÉES{camp}. SÉLECTIONNE les contributions qui expriment le MIEUX les arguments "
        "PRINCIPAUX et DISTINCTS du groupe — une par idée distincte, la plus claire et "
        f"représentative. Au plus {max_k}, MOINS s'il y a moins d'idées distinctes. Tu ne "
        "RÉDIGES rien, tu ne reformules rien : tu renvoies UNIQUEMENT les NUMÉROS choisis, de "
        "la plus représentative à la moins. Ignore les redites. "
        'Réponds en JSON strict : {"selected":[<int>, ...]} — rien d\'autre.'
    )


def _select_user(stance: str, proposition: str | None, title: str,
                 texts: list[str]) -> str:
    head = (f"PROPOSITION : {proposition}" if proposition
            else f"THÈME : {title}")
    label = {"pour": "favorables", "contre": "défavorables"}.get(stance, "")
    return (f"{head}\n\nCONTRIBUTIONS{f' ({label})' if label else ''} :\n"
            + "\n".join(f"[{i}] {t[:200]}" for i, t in enumerate(texts)))


def _parse_selected(content: str, n: int, max_k: int) -> list[int]:
    """Sortie LLM → indices de contributions choisis. Durci : bornés, dédupliqués, ≤ max_k.

    Robuste au phrasé de `large` qui renvoie parfois des objets {\"i\":3} au lieu d'entiers nus.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    items = data.get("selected") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[int] = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("i", item.get("index", item.get("numero")))
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n and idx not in out:
            out.append(idx)
        if len(out) >= max_k:
            break
    return out


# --------------------------------------------------------------------------- #
# Cœur numérique PUR (numpy, déterministe — recalibrer ne re-paye jamais le LLM)
# --------------------------------------------------------------------------- #
@dataclass
class Matched:
    """Un argument gardé : indices (lignes du groupe) de ses claims, similarités alignées."""
    arg_index: int
    assigned: list[int]   # triés par similarité décroissante
    sims: list[float]


def dedup_candidates(vecs: np.ndarray, threshold: float) -> list[int]:
    """Indices des candidats gardés — glouton dans l'ordre du LLM (le 1er énoncé prime).

    Évite qu'un même argument reformulé deux fois SPLITTE son support entre doublons.
    """
    kept: list[int] = []
    for i in range(vecs.shape[0]):
        if all(float(np.dot(vecs[i], vecs[j])) < threshold for j in kept):
            kept.append(i)
    return kept


def back_match(arg_vecs: np.ndarray, group_vecs: np.ndarray, *,
               sim_threshold: float, min_support: int) -> list[Matched]:
    """Re-source chaque argument sur les claims du groupe (cosinus, vecteurs L2-normalisés).

    Assignation ARGMAX EXCLUSIVE : chaque claim soutient AU PLUS UN argument (Σ des
    supports ≤ taille du groupe — pas de double comptage). Fail-closed : un argument
    avec moins de `min_support` claims assignés est supprimé.
    """
    if arg_vecs.size == 0 or group_vecs.size == 0:
        return []
    sims = group_vecs @ arg_vecs.T                     # (n_claims, k_args)
    best = np.argmax(sims, axis=1)                     # argmax exclusif par claim
    best_sim = sims[np.arange(sims.shape[0]), best]
    kept: list[Matched] = []
    for j in range(arg_vecs.shape[0]):
        rows = np.where((best == j) & (best_sim >= sim_threshold))[0]
        if len(rows) < min_support:
            continue
        order = rows[np.argsort(-best_sim[rows])]
        kept.append(Matched(arg_index=j,
                            assigned=[int(r) for r in order],
                            sims=[float(best_sim[r]) for r in order]))
    return kept


def merge_arguments(entries: list[dict], *, dedup_threshold: float,
                    parent_max: int, top_sources: int) -> list[dict]:
    """Fusionne les arguments de feuilles descendantes (rollup parent, zéro LLM).

    Tri par support décroissant puis absorption gloutonne des quasi-doublons
    (cos ≥ seuil sur `_vec`). Les ensembles de claims des feuilles étant DISJOINTS,
    sommer `n_support`/`weight` est honnête. `merged_from` trace la provenance.
    """
    ordered = sorted(entries, key=lambda e: -e["n_support"])
    merged: list[dict] = []
    for entry in ordered:
        home = next((m for m in merged
                     if float(np.dot(m["_vec"], entry["_vec"])) >= dedup_threshold), None)
        if home is None:
            merged.append({**entry, "sources": list(entry["sources"]),
                           "merged_from": [entry["theme_id"]], "share": None})
            continue
        home["n_support"] += entry["n_support"]
        home["weight"] = round(home["weight"] + entry["weight"], 3)
        home["merged_from"].append(entry["theme_id"])
        home["sources"] = sorted(home["sources"] + list(entry["sources"]),
                                 key=lambda s: -s["similarity"])[:top_sources]
    merged.sort(key=lambda e: -e["n_support"])
    out = []
    for entry in merged[:parent_max]:
        entry = dict(entry)
        entry["merged_from"] = sorted(set(entry["merged_from"]))
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Synthèse LLM par groupe (cachée par contenu — rebuilds idempotents)
# --------------------------------------------------------------------------- #
_ARG_MEM: dict = {}


def _content_key(dataset: str, theme_id: str, stance: str, model: str,
                 proposition: str | None, texts: list[str]) -> str:
    payload = "\x00".join([dataset, theme_id, stance, model,
                           proposition or "", str(MAX_K), *texts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def select_group(dataset: str, theme_id: str, stance: str,
                 proposition: str | None, title: str, texts: list[str],
                 *, model: str) -> list[int]:
    """Indices des contributions SÉLECTIONNÉES pour un groupe (caché ; repli = [] non caché).

    Le LLM CHOISIT (V-SELECT), il ne rédige pas — la sortie est une liste d'entiers (indices
    dans `texts`). Un cache d'ancien format (`{"arguments":[...]}` de l'ex-synthèse) décode en
    None → on régénère proprement (pas de collision).
    """
    n = len(texts)
    key = _content_key(dataset, theme_id, stance, model, proposition, texts)
    value, _source = cached_llm(
        mem_cache=_ARG_MEM,
        key=key,
        disk_path=store.analysis_dir(dataset) / "arguments_llm" / f"{key}.json",
        build_messages=lambda: [
            {"role": "system", "content": _select_system(stance, MAX_K)},
            {"role": "user", "content": _select_user(stance, proposition, title, texts)},
        ],
        fallback_fn=lambda _reason, _exc=None: [],
        model=model,
        max_tokens=120,
        temperature=0.0,
        json_mode=True,
        decode=lambda data: ([int(i) for i in data["selected"]
                              if isinstance(i, int) and 0 <= i < n]
                             if isinstance(data, dict)
                             and isinstance(data.get("selected"), list)
                             and all(isinstance(i, int) for i in data["selected"]) else None),
        encode=lambda v: {"theme_id": theme_id, "stance": stance,
                          "model": model, "selected": v},
        postprocess=lambda raw: _parse_selected(raw, n, MAX_K),
        accept=lambda v: isinstance(v, list) and len(v) > 0,
        cache_fallback=False,  # sélection vide → on réessaie quand l'API/LLM revient
    )
    return value if isinstance(value, list) else []


# --------------------------------------------------------------------------- #
# Groupes par feuille — partition stance existante, repli neutre
# --------------------------------------------------------------------------- #
def _leaf_groups(node: ThemeNode, prepared, stance_map: dict,
                 prop_by_leaf: dict) -> tuple[str, list[tuple[str, str | None, list]]]:
    """(mode, [(stance, proposition, [(gi, avis_id, text)])]) pour une feuille.

    Mode `pour_contre` si la feuille a un clivage baké (les groupes = partition
    favorable/défavorable de `claim_stance`, nuance écartée) ; sinon `neutre` (tous
    les claims). Un groupe plus petit que MIN_SUPPORT est écarté d'office : il ne
    pourra jamais produire d'argument suffisamment sourcé (zéro appel LLM).
    """
    claims = _leaf_claims(node, prepared)
    proposition = prop_by_leaf.get(node.id)
    if proposition:
        groups = []
        for internal, label in _STANCE_LABELS.items():
            sub = [c for c in claims
                   if stance_map.get(f"{c[1]}#{c[0]}", {}).get("stance") == internal]
            if len(sub) >= MIN_SUPPORT:
                groups.append((label, proposition, sub))
        return "pour_contre", groups
    if len(claims) >= MIN_SUPPORT:
        return "neutre", [("neutre", None, claims)]
    return "neutre", []


def _cap_positions(n: int, vecs: np.ndarray) -> list[int]:
    """POSITIONS (indices dans le groupe) montrées au LLM. Si le groupe dépasse INPUT_CAP :
    garde les plus proches du centroïde (déterministe, représentatif). On renvoie les POSITIONS
    (pas les items) pour pouvoir re-mapper les indices SÉLECTIONNÉS par le LLM vers le groupe."""
    if n <= INPUT_CAP:
        return list(range(n))
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    sims = vecs @ centroid
    return sorted(int(i) for i in np.argsort(-sims)[:INPUT_CAP])


# --------------------------------------------------------------------------- #
def build_arguments(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    extract_model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = DEFAULT_SEED,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Mine les arguments par feuille×stance, les re-source, agrège aux parents,
    persiste `arguments.json`, renvoie le payload.

    `extract_model` doit matcher `build_analysis` (clé du cache claims), comme pour
    l'opinion. L'embedding des candidats se fait EN UN SEUL batch hors threads (torch).
    """
    t0 = perf_counter()
    dataset = ds.id
    model = model or MODEL
    extract_model = extract_model or EXTRACT_MODEL
    mistral_client.reset_usage()

    _log(f"{dataset} · construction de l'arbre (caché si déjà extrait)…")
    # `abstract=True` : même arbre (macros comprises) que `build_analysis` → theme_id identiques
    # (charge le cache d'abstraction), quel que soit l'ordre des builds. Cf. build_opinion.
    tree = build_theme_tree(ds, backend=backend, model=extract_model, embedder=embedder,
                            resolution=resolution, seed=seed, abstract=True)
    prepared = tree.prepared

    opinion = store.read_opinion(dataset) or {}
    prop_by_leaf = {t["theme_id"]: t.get("proposition")
                    for t in opinion.get("themes", [])
                    if not t.get("is_aggregate") and t.get("profil") != "impur"
                    and t.get("proposition")}
    stance_map = store.read_claim_stance(dataset) or {}

    # ── 1. Groupes (feuille × stance), puis SÉLECTION LLM parallèle par groupe (V-SELECT).
    leaves = [tree.nodes[nid] for nid in tree.order if not tree.nodes[nid].children]
    jobs: list[dict] = []
    leaf_mode: dict[str, str] = {}
    leaf_group_sizes: dict[str, dict[str, int]] = {}
    for node in leaves:
        mode, groups = _leaf_groups(node, prepared, stance_map, prop_by_leaf)
        leaf_mode[node.id] = mode
        leaf_group_sizes[node.id] = {stance: len(sub) for stance, _p, sub in groups}
        title = node.title or node.label
        for stance, proposition, sub in groups:
            gvecs = prepared.claim_vecs[[gi for gi, _aid, _t in sub]].astype(np.float32)
            shown_pos = _cap_positions(len(sub), gvecs)   # positions montrées au LLM
            jobs.append({"node": node, "stance": stance, "proposition": proposition,
                         "title": title, "group": sub, "gvecs": gvecs,
                         "shown_pos": shown_pos,
                         "texts": [sub[p][2] for p in shown_pos]})

    total = len(jobs)
    _log(f"{dataset} · {total} groupe(s) feuille×stance à sélectionner (V-SELECT, modèle {model})")
    done = 0
    lock = threading.Lock()

    def _select(job: dict) -> dict:
        job["selected_shown"] = select_group(
            dataset, job["node"].id, job["stance"], job["proposition"],
            job["title"], job["texts"], model=model)
        return job

    def _record(k: int) -> None:
        if on_progress:
            on_progress("arguments", k, total)
        if k == total or k % 5 == 0:
            _log(f"{dataset} · sélection {k}/{total}")

    if LLM_MAX_WORKERS <= 1 or total <= 1:
        for job in jobs:
            _select(job)
            done += 1
            _record(done)
    else:
        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS,
                                thread_name_prefix="agora-arguments") as ex:
            futures = [ex.submit(_select, job) for job in jobs]
            for fut in as_completed(futures):
                fut.result()
                with lock:
                    done += 1
                    k = done
                _record(k)

    # ── 2. Par groupe : les claims SÉLECTIONNÉS sont les candidats (leurs vecteurs RAW servent
    #    de vecteurs d'argument — pas de ré-embedding : l'ablation a validé le span BRUT, cf.
    #    research/argmine_extract_note.md). Dédup → back-match fail-closed → entrées gardées.
    #    Le texte d'argument servi EST le claim sélectionné → VERBATIM par construction.
    theme_entries: dict[str, dict] = {}
    leaf_args_kept: dict[str, list[dict]] = {}   # avec _vec, pour le rollup
    n_kept = n_dropped = n_candidates = 0
    for job in jobs:
        node, stance = job["node"], job["stance"]
        group, gvecs = job["group"], job["gvecs"]
        sel_pos = [job["shown_pos"][j] for j in job["selected_shown"]]  # positions dans le groupe
        n_candidates += len(sel_pos)
        sel_vecs = gvecs[sel_pos] if sel_pos else np.zeros((0, gvecs.shape[1]), np.float32)
        kept_idx = dedup_candidates(sel_vecs, DEDUP_THRESHOLD) if sel_pos else []
        matches = back_match(sel_vecs[kept_idx], gvecs,
                             sim_threshold=SIM_THRESHOLD, min_support=MIN_SUPPORT) \
            if kept_idx else []

        entry = theme_entries.setdefault(node.id, {
            "theme_id": node.id, "title": job["title"], "mode": leaf_mode[node.id],
            "proposition": job["proposition"], "n_claims": leaf_group_sizes[node.id],
            "arguments": [],
        })

        for k, m in enumerate(matches):
            seed_pos = sel_pos[kept_idx[m.arg_index]]   # position (dans le groupe) du claim-seed
            seed_gi, seed_aid, seed_text = group[seed_pos]
            gi_rows = [group[r] for r in m.assigned]
            sources = [{"avis_id": aid, "claim_id": f"{aid}#{gi}",
                        "text": prepared.claim_texts[gi], "similarity": round(sim, 4)}
                       for (gi, aid, _t), sim in zip(gi_rows[:TOP_SOURCES], m.sims)]
            arg_entry = {
                "id": f"{node.id}:{stance}:{k}",
                "theme_id": node.id,
                "stance": stance,
                "argument": seed_text,          # VERBATIM (claim sélectionné, jamais reformulé)
                "verbatim": True,
                "claim_id": f"{seed_aid}#{seed_gi}",
                "n_support": len(m.assigned),
                "weight": round(float(sum(prepared.claim_weight[gi]
                                          for gi, _aid, _t in gi_rows)), 3),
                "share": round(len(m.assigned) / len(group), 3),
                "sources": sources,
            }
            entry["arguments"].append(arg_entry)
            leaf_args_kept.setdefault(node.id, []).append(
                {**arg_entry, "_vec": sel_vecs[kept_idx[m.arg_index]]})
            n_kept += 1
        n_dropped += len(kept_idx) - len(matches)

    # ── 4. Rollup parents : fusion par embeddings des arguments des feuilles (zéro LLM).
    def _leaf_descendants(nid: str) -> list[str]:
        node = tree.nodes[nid]
        if not node.children:
            return [nid]
        return [l for c in node.children for l in _leaf_descendants(c)]

    for nid in tree.order:
        node = tree.nodes[nid]
        if not node.children:
            continue
        pooled = [a for l in _leaf_descendants(nid) for a in leaf_args_kept.get(l, [])]
        if not pooled:
            continue
        by_stance: dict[str, list[dict]] = {}
        for a in pooled:
            by_stance.setdefault(a["stance"], []).append(a)
        merged_all: list[dict] = []
        for stance, group_args in by_stance.items():
            merged = merge_arguments(group_args, dedup_threshold=DEDUP_THRESHOLD,
                                     parent_max=PARENT_MAX, top_sources=TOP_SOURCES)
            for k, entry in enumerate(merged):
                entry.pop("_vec", None)
                entry.update(id=f"{nid}:{stance}:{k}", theme_id=nid)
                merged_all.append(entry)
        theme_entries[nid] = {
            "theme_id": nid, "title": node.title or node.label,
            "mode": "pour_contre" if any(a["stance"] != "neutre" for a in merged_all)
                    else "neutre",
            "proposition": None, "is_aggregate": True,
            "n_children": len(node.children),
            "arguments": merged_all,
        }

    # ── 5. Payload trié dans l'ordre de l'arbre + persistance + coût.
    rank = {nid: i for i, nid in enumerate(tree.order)}
    themes = sorted((t for t in theme_entries.values() if t["arguments"] or DEBUG),
                    key=lambda t: rank.get(t["theme_id"], 1 << 30))
    took_s = round(perf_counter() - t0, 1)
    payload = {
        "dataset": dataset,
        "model": model,
        "embedder": embedder,
        "seed": seed,
        "params": {"sim_threshold": SIM_THRESHOLD, "min_support": MIN_SUPPORT,
                   "max_k": MAX_K, "dedup_threshold": DEDUP_THRESHOLD,
                   "input_cap": INPUT_CAP, "parent_max": PARENT_MAX,
                   "top_sources": TOP_SOURCES},
        "prompt_system": _select_system("<stance>", MAX_K),
        "counts": {"themes": len(themes), "arguments": n_kept,
                   "candidates": n_candidates, "dropped": n_dropped},
        "n_leaves": len(leaves),
        "took_seconds": took_s,
        "themes": themes,
    }
    store.write_arguments(dataset, payload)
    try:
        from backend import cost as _cost
        _cost.record_phase(dataset, "arguments", mistral_client.get_usage(),
                           duration_seconds=took_s)
    except Exception as _e:
        _log(f"{dataset} · (coût arguments non enregistré: {_e})")
    _log(f"{dataset} · ✓ arguments.json écrit · {n_kept} arguments gardés / "
         f"{n_candidates} candidats ({n_dropped} droppés faute de support) · {took_s}s")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mine les arguments par thème (synthèse LLM sourcée sur contributions).")
    ap.add_argument("--dataset", required=True, help="id du dataset (sous backend/cache/)")
    ap.add_argument("--backend", default=None, help="api (défaut) | mac | auto")
    ap.add_argument("--model", default=None, help=f"modèle de synthèse (défaut {MODEL})")
    ap.add_argument("--extract-model", default=None,
                    help=f"modèle d'extraction de l'arbre (défaut {EXTRACT_MODEL} — doit "
                         f"matcher build_analysis pour réutiliser le cache claims)")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")

    ds = load_dataset(args.dataset)
    build_arguments(ds, backend=args.backend, model=args.model,
                    extract_model=args.extract_model, embedder=args.embedder,
                    resolution=args.resolution, seed=args.seed)


if __name__ == "__main__":
    main()

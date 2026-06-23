"""`AnalysisState` — construction INCRÉMENTALE de l'arbre de thèmes (cœur live).

Au lieu de clusteriser tout le corpus d'un coup (Leiden global + coarsening), on
ajoute les claims **un par un**, ce qui permet de rejouer un build EN STREAMING :

  1. **rattachement** au cluster FEUILLE le plus proche (cos centroïde max) — toujours ;
  2. **maj O(1)** des stats du nœud et de ses ancêtres : on garde la somme `S = Σv`
     par nœud → centroïde = S/‖S‖, dispersion = 1 − ‖S‖/n, consensus via les identités
     exactes de `_node_stats` (aucune matrice n×n, aucun recompute global) ;
  3. **SPLIT LOCAL sur divergence** : si la dispersion de la feuille touchée dépasse le
     seuil τ DÉRIVÉ des dispersions courantes (`_derive_tau`) ET que `_subdivide` dégage
     ≥2 sous-thèmes viables, ses claims deviennent ses enfants (récursif, borné).

Tout est RÉUTILISÉ depuis `analysis.py` (ThemeNode/ThemeTree, `_node_stats`,
`_derive_tau`, `_subdivide`, `_representatives`, `theme_dict`, `analysis_payload`) et
`pipeline.cluster` (defaults dérivés, naming c-TF-IDF, palette) → généricité préservée,
zéro magic-number corpus-spécifique.

Deux usages :
  - **BUILD** (`backend.build_analysis`) : `AnalysisState.from_prepared(...)` puis
    `add_all()` + `snapshot()` → persiste (UMAP retiré, front en d3-pack) ;
  - **STREAM** (`/stream`) : rejoue les claims CACHÉS d'un dataset (AUCUN LLM) via
    `stream_events()`, qui émet les events du contrat au fil de l'eau.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from backend.analysis import (
    MAX_DEPTH,
    ThemeNode,
    ThemeTree,
    _assign_convergence,
    _derive_tau,
    _name_nodes,
    _node_stats,
    _representatives,
    _shrinkage_k,
    _subdivide,
    analysis_payload,
    theme_dict,
)
from backend.claims_endpoint import PreparedClaims, prepare_claims
from backend.develop import corpus_idf
from pipeline.claims.pipeline import DEFAULT_EMBEDDER, DEFAULT_SEED
from pipeline.cluster.adaptive import MIN_SUB_FLOOR, derive_defaults
from pipeline.cluster.knn import build_knn_graph
from pipeline.cluster.leiden_cluster import run_leiden
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters
from pipeline.cluster.palette import color_for

# Mode de dérivation du niveau MACRO de l'incrémental :
#   - "root"      : macros = enfants du root (split-racine unique, défaut historique) ;
#   - "recompute" : RECOMPUTE PARTIEL (option B' — Lane E0) = Leiden sur les centroïdes
#                   des FEUILLES (n_feuilles ≪ n_claims → cheap), matérialisé en
#                   root→macro→feuilles. Corrige la STALENESS du split-racine figé tôt.
# Verdict d'éval (research/inc_macro_report.md) : "recompute" est la seule dérivation
# GÉNÉRIQUE (ne s'effondre jamais — option A coarsen : granddebat V=0.03) ; gain V
# modeste, le plafond étant fixé par la structure des feuilles (oracle ≈0.52/0.75).
MACRO_MODE_ROOT = "root"
MACRO_MODE_RECOMPUTE = "recompute"


class AnalysisState:
    """État mutable = l'arbre de thèmes courant + sommes incrémentales par nœud.

    Construit à partir d'un `PreparedClaims` (claims + embeddings DÉJÀ cachés). Les
    membres d'un nœud sont des indices GLOBAUX dans `prepared.claim_vecs` ; on les
    « ajoute » dans l'ordre du cache, donc l'indice == position dans `prepared`.
    """

    def __init__(self, prepared: PreparedClaims, *, dataset: str = "",
                 resolution: float = 1.0, seed: int = DEFAULT_SEED,
                 macro_mode: str = MACRO_MODE_ROOT) -> None:
        self.prepared = prepared
        self.dataset = dataset
        self.macro_mode = macro_mode
        self.mat = np.asarray(prepared.claim_vecs, dtype=np.float64)   # support des vecs
        self.owner = prepared.claim_owner
        self.weights = prepared.claim_weight
        self.texts = prepared.claim_texts
        self.base_resolution = resolution
        self.seed = seed

        self.nodes: dict[str, ThemeNode] = {}
        self.order: list[str] = []
        self._counter = 0
        self.root_id: str | None = None

        # Sommes incrémentales par nœud (maj O(1)).
        self._sum: dict[str, np.ndarray] = {}              # S = Σv
        self._wsum: dict[str, float] = {}                  # Σ poids social
        self._owners: dict[str, dict[int, int]] = {}       # owner -> nb de claims (n_avis = clés)

        self._n = len(self.texts)
        self.derived_global = derive_defaults(self.mat.astype(np.float32)) if self._n else None
        self.floor = self.derived_global.min_sub_size if self.derived_global else MIN_SUB_FLOOR
        self._tau = float("inf")
        self._corpus_stop: set[str] | None = None
        # Anti-réessai : taille à laquelle un `_subdivide` a échoué (None = jamais).
        self._split_blocked: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Fabrique : prépare (claims + embeddings cachés) puis construit l'état
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dataset(cls, ds, *, backend: str | None = None, model: str | None = None,
                     embedder: str = DEFAULT_EMBEDDER, min_chars: int | None = None,
                     resolution: float = 1.0, seed: int = DEFAULT_SEED,
                     macro_mode: str = MACRO_MODE_ROOT) -> "AnalysisState":
        kw = {} if min_chars is None else {"min_chars": min_chars}
        prepared = prepare_claims(ds, backend=backend, model=model, embedder=embedder, **kw)
        return cls(prepared, dataset=ds.id, resolution=resolution, seed=seed,
                   macro_mode=macro_mode)

    # ------------------------------------------------------------------ #
    # Helpers d'arbre
    # ------------------------------------------------------------------ #
    def _new_id(self) -> str:
        nid = f"n{self._counter}"
        self._counter += 1
        return nid

    def _leaves(self) -> list[ThemeNode]:
        return [n for n in self.nodes.values() if not n.children]

    def _ancestors(self, node: ThemeNode) -> Iterator[ThemeNode]:
        cur = node
        while cur is not None:
            yield cur
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None

    def _refresh_stats(self, node: ThemeNode) -> None:
        """Recalcule centroïde/dispersion/consensus/poids/n_avis depuis la somme S (O(d))."""
        s = self._sum[node.id]
        n = node.n_claims
        norm = float(np.linalg.norm(s))
        node.centroid = s / norm if norm > 0 else s
        node.dispersion = round(max(0.0, 1.0 - (norm / n if n else 0.0)), 4)
        node.consensus = round((norm * norm - n) / (n * (n - 1)), 4) if n > 1 else 1.0
        node.weight = round(self._wsum[node.id], 1)
        node.n_avis = len(self._owners[node.id])

    def _init_node(self, members: list[int], parent_id: str | None, depth: int) -> ThemeNode:
        """Crée un nœud à partir d'une liste de membres (somme initialisée d'un bloc)."""
        nid = self._new_id()
        s = self.mat[members].sum(axis=0) if members else np.zeros(self.mat.shape[1])
        owners: dict[int, int] = {}
        for i in members:
            owners[self.owner[i]] = owners.get(self.owner[i], 0) + 1
        node = ThemeNode(
            id=nid, parent_id=parent_id, depth=depth, members=list(members),
            centroid=s, dispersion=0.0, consensus=1.0, weight=0.0,
            n_claims=len(members), n_avis=len(owners),
        )
        self.nodes[nid] = node
        self.order.append(nid)
        self._sum[nid] = s.astype(np.float64)
        self._wsum[nid] = float(self.weights[members].sum()) if members else 0.0
        self._owners[nid] = owners
        self._refresh_stats(node)
        return node

    # ------------------------------------------------------------------ #
    # Seuil τ DÉRIVÉ des dispersions des feuilles courantes (qualifiantes)
    # ------------------------------------------------------------------ #
    def _tau_now(self) -> float:
        disps = [n.dispersion for n in self._leaves() if n.n_claims >= self.floor]
        self._tau = _derive_tau(disps)
        return self._tau

    # ------------------------------------------------------------------ #
    # Naming / couleur / convergence LOCAUX (sans recompute global)
    # ------------------------------------------------------------------ #
    def _stopwords(self) -> set[str]:
        if self._corpus_stop is None:
            self._corpus_stop, _ = derive_corpus_stopwords(self.texts)
        return self._corpus_stop

    def _name(self, node_ids: list[str]) -> None:
        """Nomme `node_ids` via c-TF-IDF (mots-vides corpus-dérivés, partagés)."""
        ids = [nid for nid in node_ids if nid in self.nodes]
        if not ids:
            return
        docs = {i: [self.texts[m] for m in self.nodes[nid].members] for i, nid in enumerate(ids)}
        names = name_clusters(docs, corpus_stopwords=self._stopwords())
        for i, nid in enumerate(ids):
            info = names.get(i, {})
            self.nodes[nid].label = info.get("label", f"thème {nid}")
            self.nodes[nid].keywords = info.get("keywords", [])

    def effective_macro_ids(self) -> list[str]:
        """Niveau « macro » d'affichage : les enfants de la racine si elle a splité,
        sinon la racine seule. (L'incrémental est mono-racine : le 1er nœud est racine.)"""
        if self.root_id is None:
            return []
        root = self.nodes[self.root_id]
        return list(root.children) if root.children else [self.root_id]

    def _recolor(self) -> None:
        """Couleur par MACRO effectif (palette.py), héritée par les descendants."""
        macro_ids = self.effective_macro_ids()
        rank = {mid: i for i, mid in enumerate(macro_ids)}
        n = max(1, len(macro_ids))
        for node in self.nodes.values():
            cur = node
            while cur.id not in rank and cur.parent_id is not None:
                cur = self.nodes[cur.parent_id]
            node.color = color_for(rank.get(cur.id, 0), n)

    def _shrink_k(self) -> float:
        return _shrinkage_k([max(0, self.nodes[m].n_avis) for m in self.effective_macro_ids()])

    def _set_convergence(self, node: ThemeNode, kk: float) -> None:
        n = max(0, node.n_avis)
        node.convergence = round((n / (n + kk)) * node.consensus, 4) if n else 0.0

    # ------------------------------------------------------------------ #
    # Ajout d'un claim (rattache + maj O(1) + split local éventuel)
    # ------------------------------------------------------------------ #
    def add_claim(self, idx: int) -> list[dict]:
        """Ajoute le claim d'indice GLOBAL `idx` ; renvoie les events du contrat émis."""
        vec = self.mat[idx]
        events: list[dict] = []

        if self.root_id is None:                       # 1er claim → racine
            root = self._init_node([idx], None, 0)
            self.root_id = root.id
            self._recolor()
            events.append(self._claim_event(root))
            return events

        # 1) rattachement à la feuille la plus proche (cos centroïde max).
        leaves = self._leaves()
        cents = np.asarray([n.centroid for n in leaves])
        leaf = leaves[int(np.argmax(cents @ vec))]

        # 2) maj O(1) de la feuille ET de ses ancêtres (membres imbriqués).
        for node in self._ancestors(leaf):
            node.members.append(idx)
            node.n_claims += 1
            self._sum[node.id] = self._sum[node.id] + vec
            self._wsum[node.id] += float(self.weights[idx])
            self._owners[node.id][self.owner[idx]] = self._owners[node.id].get(self.owner[idx], 0) + 1
            self._refresh_stats(node)
        events.append(self._claim_event(leaf))

        # 3) split local si la feuille a divergé (récursif, borné, throttlé).
        events.extend(self._maybe_split(leaf))
        return events

    def add_all(self) -> None:
        """Ajoute tous les claims du cache dans l'ordre (build batch via l'incrémental).

        En mode `recompute`, re-dérive le niveau MACRO à la fin (option B' — Lane E0).
        """
        for idx in range(self._n):
            self.add_claim(idx)
        if self.macro_mode == MACRO_MODE_RECOMPUTE:
            self.rebuild_macro_layer()

    # ------------------------------------------------------------------ #
    # RECOMPUTE PARTIEL du niveau MACRO (option B' — Lane E0, derrière flag)
    # ------------------------------------------------------------------ #
    def rebuild_macro_layer(self) -> bool:
        """Re-dérive la partition MACRO par Leiden sur les CENTROÏDES des feuilles.

        L'incrémental pur fige sa partition macro au 1er split du root (sur un petit
        échantillon) → trop peu de macros, partition STALE. On la recalcule sur un
        sous-problème MINUSCULE (n_feuilles ≪ n_claims, borné par n_feuilles) : kNN+Leiden
        DÉRIVÉS sur les centroïdes des feuilles courantes → communautés de feuilles, puis
        on MATÉRIALISE un niveau propre `root → macro_i → feuilles` (les subdivisions
        internes intermédiaires sont aplaties sous les macros). Générique (k/seuil/
        résolution dérivés), ZÉRO LLM (réutilise les embeddings cachés).

        Verdict d'éval (research/inc_macro_report.md) : option B' (Leiden SANS coarsening)
        est la seule dérivation à NE JAMAIS s'effondrer — le coarsening final (option B
        littérale) détruit les corpus multi-sujets (granddebat V 0.38→0.14). Le gain de
        V-mesure sur la baseline reste modeste, le PLAFOND étant fixé par la structure des
        feuilles (oracle ≈0.52 granddebat / 0.75 tiktok), pas par la dérivation macro.

        Renvoie True si la partition a été matérialisée, False si abstention (on garde
        la structure courante : <2 feuilles, ou Leiden ne dégage pas ≥2 communautés).
        """
        if self.root_id is None:
            return False
        leaves = self._leaves()
        if len(leaves) < 2:
            return False

        cents = np.ascontiguousarray(
            np.asarray([lf.centroid for lf in leaves]), dtype=np.float64)
        dd = derive_defaults(cents.astype(np.float32))
        graph = build_knn_graph(cents, k=dd.k, threshold=dd.threshold)
        membership = run_leiden(graph, resolution=self.base_resolution,
                                seed=self.seed).membership
        by_comm: dict[int, list[int]] = {}
        for li, c in enumerate(membership):
            by_comm.setdefault(c, []).append(li)
        groups = list(by_comm.values())
        if len(groups) < 2:
            return False                  # abstention : on garde la structure courante

        # Matérialise root → macros → feuilles. On conserve racine + feuilles, on crée un
        # macro par communauté, on jette les nœuds internes intermédiaires (stale).
        root = self.nodes[self.root_id]
        nodes: dict[str, ThemeNode] = {self.root_id: root}
        order: list[str] = [self.root_id]
        keep_sum = {self.root_id: self._sum[self.root_id]}
        keep_wsum = {self.root_id: self._wsum[self.root_id]}
        keep_owners = {self.root_id: self._owners[self.root_id]}
        root.children, root.depth = [], 0

        for grp in groups:
            grp_leaves = [leaves[li] for li in grp]
            members = [m for lf in grp_leaves for m in lf.members]
            mid = self._new_id()
            s = self.mat[members].sum(axis=0)
            owners: dict[int, int] = {}
            for i in members:
                owners[self.owner[i]] = owners.get(self.owner[i], 0) + 1
            macro = ThemeNode(
                id=mid, parent_id=self.root_id, depth=1, members=list(members),
                centroid=s, dispersion=0.0, consensus=1.0, weight=0.0,
                n_claims=len(members), n_avis=len(owners),
            )
            nodes[mid] = macro
            order.append(mid)
            keep_sum[mid] = s.astype(np.float64)
            keep_wsum[mid] = float(self.weights[members].sum())
            keep_owners[mid] = owners
            root.children.append(mid)
            for lf in grp_leaves:                  # ré-parente les feuilles sous le macro
                lf.parent_id, lf.depth, lf.children = mid, 2, []
                macro.children.append(lf.id)
                nodes[lf.id] = lf
                order.append(lf.id)
                keep_sum[lf.id] = self._sum[lf.id]
                keep_wsum[lf.id] = self._wsum[lf.id]
                keep_owners[lf.id] = self._owners[lf.id]

        self.nodes, self.order = nodes, order
        self._sum, self._wsum, self._owners = keep_sum, keep_wsum, keep_owners
        for node in self.nodes.values():
            self._refresh_stats(node)
        # Nommage / couleur / convergence LOCAUX (tout l'arbre re-matérialisé).
        self._name(list(self.nodes.keys()))
        self._recolor()
        kk = self._shrink_k()
        for node in self.nodes.values():
            self._set_convergence(node, kk)
        return True

    # ------------------------------------------------------------------ #
    # Split LOCAL d'une feuille hétérogène
    # ------------------------------------------------------------------ #
    def _maybe_split(self, leaf: ThemeNode) -> list[dict]:
        if leaf.children or leaf.depth >= MAX_DEPTH:
            return []
        # Précondition de taille : de quoi former ≥2 sous-thèmes RÉELLEMENT viables. On
        # exige ≥ 2·floor (floor = `min_sub_size` DÉRIVÉ du corpus) — même garde-fou que
        # le build batch, qui n'évalue la subdivision que des groupes ≥ floor (les petites
        # feuilles ne peuvent pas être coupées et sur-fragmenteraient l'arbre).
        if leaf.n_claims < 2 * self.floor:
            return []
        # Anti-réessai : pas de nouvel essai tant que la feuille n'a pas regrossi de 50 %.
        blocked = self._split_blocked.get(leaf.id)
        if blocked is not None and leaf.n_claims < blocked * 1.5:
            return []

        tau = self._tau_now()
        # Gate τ : on subdivise les feuilles plus dispersées que le seuil DÉRIVÉ. En
        # amorçage (τ = +inf faute de ≥2 feuilles qualifiantes), la précondition de taille
        # ci-dessus suffit → la 1re partition (les macros) émerge.
        if tau != float("inf") and leaf.dispersion <= tau:
            return []

        groups = _subdivide(leaf.members, self.mat, self.base_resolution, self.seed)
        if not groups or len(groups) < 2:
            self._split_blocked[leaf.id] = leaf.n_claims      # échec → throttle
            return []

        # Réalise le split : les claims de la feuille deviennent ses enfants.
        child_ids: list[str] = []
        for grp in groups:
            child = self._init_node(grp, leaf.id, leaf.depth + 1)
            leaf.children.append(child.id)
            child_ids.append(child.id)
        self._split_blocked.pop(leaf.id, None)

        # Nommage / couleur / convergence LOCAUX (parent + enfants seulement).
        self._name([leaf.id, *child_ids])
        self._recolor()
        kk = self._shrink_k()
        for nid in (leaf.id, *child_ids):
            self._set_convergence(self.nodes[nid], kk)

        events = [self._split_event(leaf, child_ids)]
        # Récursion bornée : un enfant encore trop dispersé peut se subdiviser à son tour.
        for cid in child_ids:
            events.extend(self._maybe_split(self.nodes[cid]))
        return events

    # ------------------------------------------------------------------ #
    # Sérialisation des events (format du contrat)
    # ------------------------------------------------------------------ #
    def _claim_event(self, node: ThemeNode) -> dict:
        self._set_convergence(node, self._shrink_k())
        return {
            "type": "claim_added",
            "theme_id": node.id,
            "n_avis": node.n_avis,
            "n_claims": node.n_claims,
            "weight": node.weight,
            "dispersion": node.dispersion,
            "consensus": node.consensus,
            "convergence": node.convergence,
        }

    def _split_event(self, parent: ThemeNode, child_ids: list[str]) -> dict:
        children = []
        for cid in child_ids:
            c = self.nodes[cid]
            children.append({
                "id": c.id,
                "title": c.title or c.label,
                "label": c.label,
                "n_avis": c.n_avis,
                "n_claims": c.n_claims,
                "weight": c.weight,
                "color": c.color,
                "parent_id": c.parent_id,
                "has_children": c.has_children,
            })
        return {"type": "theme_split", "parent_id": parent.id, "children": children}

    # ------------------------------------------------------------------ #
    # Finalisation + arbre + snapshot (payload /analysis SANS x,y)
    # ------------------------------------------------------------------ #
    def finalize(self) -> None:
        """Nommage GLOBAL distinctif + représentants + couleurs + convergence (build)."""
        if not self.nodes:
            return
        _name_nodes(self.nodes, self.texts)
        self._recolor()
        _assign_convergence(self.nodes, self.effective_macro_ids())
        # idf corpus calculé une fois, partagé par tous les nœuds (D1) et /citations.
        self._claim_idf = corpus_idf(self.texts)
        for node in self.nodes.values():
            node.representative_claims = _representatives(
                node, self.mat, self.texts, idf=self._claim_idf)

    def tree(self) -> ThemeTree:
        """Vue `ThemeTree` de l'état courant (macros = niveau d'affichage effectif)."""
        return ThemeTree(
            nodes=self.nodes, order=list(self.order), macros=self.effective_macro_ids(),
            dataset=self.dataset, prepared=self.prepared, tau=self._tau,
            base_resolution=self.base_resolution, seed=self.seed,
            derived_global=self.derived_global,
            root_coarsen={"mode": "incrémental", "note": "arbre construit claim par claim "
                          "(rattachement plus-proche + split local sur divergence) — "
                          "pas de coarsening de racines"},
            claim_idf=getattr(self, "_claim_idf", None),
        )

    def snapshot(self, *, took_ms: int | None = None, finalize: bool = True) -> dict:
        """Payload au format `/analysis` (themes/edges/params/stats) SANS x,y."""
        if finalize:
            self.finalize()
        return analysis_payload(self.tree(), took_ms=took_ms)

    # ------------------------------------------------------------------ #
    # STREAM : rejoue les claims cachés en émettant les events du contrat
    # ------------------------------------------------------------------ #
    def stream_events(self, *, dataset_context: str = "") -> Iterator[dict]:
        """Génère snapshot (état initial vide) → claim_added/theme_split… → done.

        Aucun LLM : on rejoue `prepared` (claims + embeddings cachés). Les labels des
        events viennent du c-TF-IDF (local, non-LLM) ; les titres LLM restent au build.
        """
        from backend.analysis import _dataset_stats

        yield {
            "type": "snapshot",
            "themes": [theme_dict(self.nodes[i]) for i in self.order],
            "dataset_stats": _dataset_stats(self.tree()),
            "dataset_context": dataset_context,
        }
        for idx in range(self._n):
            for ev in self.add_claim(idx):
                yield ev
        root = self.nodes.get(self.root_id) if self.root_id else None
        yield {
            "type": "done",
            "n_avis": root.n_avis if root else 0,        # la racine contient tous les claims
            "n_themes": len(self.nodes),
        }

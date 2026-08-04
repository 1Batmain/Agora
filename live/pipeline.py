"""Pipeline LIVE — amorçage puis traitement tour par tour, sans re-clustering.

Deux étages, correspondant au découpage validé dans `.agent/notes/LIVE_SCENARIO.md` :

**Amorçage** (une fois, coûteux) : extraire les claims d'un lot de tours, les embedder,
les clusteriser une bonne fois, dériver un objet de clivage par thème. Les thèmes sont
ensuite GELÉS.

**Tour** (à chaque prise de parole, quelques secondes) :
`extraire → embedder → assigner au plus proche centroïde → juger la position`.
Aucun re-clustering : la carte ne bouge pas sous les yeux du spectateur.

## Ce qui est réutilisé tel quel

Extraction verbatim ancrée (`pipeline.claims`), embeddings cachés par contenu
(`pipeline.embed.vector_store`), partition à résolution γ (`pipeline.cluster.layers`),
nommage c-TF-IDF sans LLM (`pipeline.cluster.naming`), stance et clivage
(`pipeline.stance`). Le pipeline live n'est pas une réimplémentation : c'est une
COMPOSITION différente des mêmes couches — ce que la classe `PipelineProfile` rend explicite.

## Budget

Mesuré : ~1,45 claim par tour, stance à ~4 claims/s, un tour de parole toutes les ~50 s en
séance réelle. Le coût par tour est de quelques secondes d'API — environ 10× de marge sur le
temps réel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline import profile as _profile
from pipeline import stance as stance_layer
from pipeline.claims.backend import resolve_backend
from pipeline.claims.extract import extract_claims
from pipeline.claims.ollama import OllamaStats
from pipeline.claims.pipeline import Avis, embed_claim_texts
from pipeline.cluster import layers
from pipeline.cluster.naming import derive_corpus_stopwords, name_clusters
from pipeline.embed import vector_store
from live.state import Assignment, LiveState, LiveTheme, derive_coverage_threshold
from live.transcript import Turn

# Nombre de claims représentatifs montrés au LLM pour dériver l'objet de clivage d'un thème.
CLEAVAGE_SAMPLE = 12

# Taille maximale d'un FRAGMENT soumis à l'extraction.
#
# ⚠️ Mesuré, et non anticipé : un tour de parole n'est PAS court. La médiane de la séance
# retraite est de 1 121 caractères, mais le maximum atteint 20 655 (une intervention
# ministérielle interrompue 40 fois, recollée en un seul tour — cf. `live.transcript`).
# `extract_claims` groupe 8 avis par appel : sans découpage, un lot d'amorçage dépassait
# 100 000 caractères d'entrée pour un plafond de sortie de 3 200 tokens. Symptômes attendus :
# lenteur extrême, troncature du JSON, claims perdus en silence.
#
# On découpe donc les tours longs en fragments AUX FRONTIÈRES DE PHRASE. C'est sans risque
# pour l'invariant verbatim : l'ancrage se fait PAR AVIS, donc un fragment est un contexte
# d'ancrage valide — un claim reste une sous-chaîne exacte de son fragment.
MAX_CHUNK_CHARS = 1800


@dataclass
class LiveConfig:
    """Réglages d'une session live. Les MODÈLES viennent du profil, jamais d'ici."""

    profile_name: str = "live"
    cache_dir: Path = Path("var/live-cache")
    gamma: float | None = None            # None → la résolution du profil
    seed: int = 42
    # Le discours rapporté INVERSE la position d'un orateur qui cite l'adversaire pour le
    # réfuter (observé en séance). Le garde-fou est activable ; il n'est PAS validé par un
    # banc, d'où le drapeau explicite plutôt qu'un comportement implicite.
    reported_speech_guard: bool = True
    stance_enabled: bool = True

    @property
    def profile(self):
        return _profile.get_profile(self.profile_name)

    @property
    def emb_path(self) -> Path:
        return Path(self.cache_dir) / "claims_emb.npz"


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_for_extraction(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Découpe un tour long en fragments ≤ `max_chars`, aux frontières de PHRASE.

    Une phrase plus longue que la borne n'est PAS coupée : mieux vaut un fragment un peu
    trop grand qu'un claim tronqué au milieu d'une proposition — l'ancrage verbatim
    survivrait, mais le sens non.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append(current)
    return chunks


def _extract(turns: list[Turn], cfg: LiveConfig, *, question: str | None,
             progress=None) -> dict[str, list]:
    """Claims VERBATIM des tours donnés (ancrés, jamais paraphrasés).

    Les tours longs sont découpés en fragments (`split_for_extraction`) : chaque fragment
    devient un « avis » pour l'extracteur, puis les claims sont RECOLLÉS sous l'id du tour.
    """
    avis: list[Avis] = []
    chunk_owner: dict[str, str] = {}
    for t in turns:
        for k, chunk in enumerate(split_for_extraction(t.text)):
            cid = f"{t.id}~{k}"
            avis.append(Avis(id=cid, text=chunk, weight=1.0))
            chunk_owner[cid] = t.id
    if not avis:
        return {}

    backend = resolve_backend("api", model=cfg.profile.model_for("extract"))
    by_chunk = extract_claims(avis, backend=backend, stats=OllamaStats(),
                              question=question, progress=progress)

    out: dict[str, list] = {}
    for cid, claims in by_chunk.items():
        out.setdefault(chunk_owner.get(cid, cid), []).extend(claims)
    return out


def _flatten_claims(turns: list[Turn], claims_by_id: dict[str, list]
                    ) -> tuple[list[str], list[Turn], list[str]]:
    """Aplati `{turn_id: [claims]}` en listes ALIGNÉES (texte, tour d'origine, id de claim)."""
    texts: list[str] = []
    owners: list[Turn] = []
    ids: list[str] = []
    for t in turns:
        for j, c in enumerate(claims_by_id.get(t.id, [])):
            texts.append(c.text)
            owners.append(t)
            ids.append(f"{t.id}#{j}")
    return texts, owners, ids


def _embed(texts: list[str], cfg: LiveConfig) -> np.ndarray:
    """Embeddings CACHÉS PAR CONTENU — un texte nouveau coûte un seul calcul."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    embedder = cfg.profile.embedder
    vecs, _ = vector_store.embed_cached(
        texts, embedder=embedder, path=cfg.emb_path,
        embed_fn=lambda missing: embed_claim_texts(missing, embedder=embedder),
    )
    return vecs


def bootstrap(turns: list[Turn], cfg: LiveConfig, *, question: str | None = None,
              session: str = "", topic: str = "", log=print) -> LiveState:
    """Constitue les thèmes GELÉS depuis un lot de tours d'amorçage.

    Le seuil de couverture est DÉRIVÉ de la distribution des similarités intra-thème
    observée ici (cf. `derive_coverage_threshold`) : il s'adapte à l'embedder au lieu de
    figer un littéral qui se périmerait à la première bascule de modèle.
    """
    log(f"[live] amorçage · {len(turns)} tours · extraction ({cfg.profile.model_for('extract')})")
    claims_by_id = _extract(turns, cfg, question=question,
                            progress=lambda d, n: log(f"[live]   extraction {d}/{n}")
                            if d % 25 == 0 or d == n else None)
    texts, owners, _ids = _flatten_claims(turns, claims_by_id)
    if not texts:
        raise RuntimeError("amorçage vide : aucun claim extrait des tours fournis.")
    log(f"[live] amorçage · {len(texts)} claims ({len(texts)/len(turns):.2f}/tour)")

    vecs = _embed(texts, cfg)
    # Recentrage : corrige l'anisotropie de l'espace d'embedding (+19 % d'ARI mesuré).
    centred = layers.centre(vecs)
    gamma = cfg.gamma if cfg.gamma is not None else layers.FINE_GAMMA
    membership, meta = layers.flat_partition(centred, gamma=gamma, seed=cfg.seed)
    n_themes = len(set(membership.tolist()))
    log(f"[live] amorçage · {n_themes} thèmes (γ={gamma}, modularité {meta['modularity']})")

    # Centroïdes dans l'espace NON recentré : c'est là que vivront les claims à venir.
    # (Recentrer un vecteur isolé demanderait le centroïde du corpus futur — inconnu.)
    themes: list[LiveTheme] = []
    cluster_docs: dict[int, list[str]] = {}
    for cid in sorted(set(membership.tolist())):
        idx = np.where(membership == cid)[0]
        cluster_docs[cid] = [texts[i] for i in idx]

    stopwords, _diag = derive_corpus_stopwords(texts)
    named = name_clusters(cluster_docs, corpus_stopwords=stopwords)

    centroids = []
    for cid in sorted(cluster_docs):
        idx = np.where(membership == cid)[0]
        c = vecs[idx].mean(axis=0)
        n = np.linalg.norm(c)
        c = c / n if n > 0 else c
        centroids.append(c)
        info = named.get(cid, {})
        themes.append(LiveTheme(
            id=f"t{cid}",
            label=info.get("label") or f"thème {cid}",
            keywords=list(info.get("keywords") or []),
            centroid=c.astype(np.float32),
            n_bootstrap=len(idx),
        ))

    centroid_mat = np.stack(centroids).astype(np.float32)
    threshold = derive_coverage_threshold(vecs, centroid_mat, membership)
    log(f"[live] amorçage · seuil de couverture dérivé = {threshold:.4f}")

    # Objet de clivage par thème : 1 appel LLM chacun, UNE fois. C'est la cible contre
    # laquelle toutes les positions seront jugées ensuite.
    model = cfg.profile.model_for("opinion")
    for theme, cid in zip(themes, sorted(cluster_docs)):
        sample = cluster_docs[cid][:CLEAVAGE_SAMPLE]
        got = stance_layer.derive_cleavage_from(theme.label, theme.keywords, sample,
                                                model=model)
        theme.cleavage = got["objet"]
        theme.cleavage_justif = got["justif"]
        log(f"[live]   {theme.id} « {theme.label} » → clivage : {theme.cleavage}")

    state = LiveState(themes, coverage_threshold=threshold, session=session, topic=topic)
    state.turns_seen = len(turns)
    return state


def process_turn(state: LiveState, turn: Turn, cfg: LiveConfig, *,
                 question: str | None = None, log=print) -> list[Assignment]:
    """Traite UN tour : extraction → embedding → assignation → position.

    Renvoie les assignations produites (déjà ajoutées à l'état). Aucun re-clustering.
    """
    claims_by_id = _extract([turn], cfg, question=question)
    texts, owners, ids = _flatten_claims([turn], claims_by_id)
    state.turns_seen += 1
    if not texts:
        return []

    vecs = _embed(texts, cfg)

    # 1) Assignation géométrique — instantanée, aucun LLM.
    made: list[Assignment] = []
    for text, owner, cid, vec in zip(texts, owners, ids, vecs):
        theme_id, sim = state.nearest(vec)
        made.append(Assignment(
            claim_id=cid, turn_id=turn.id, seq=turn.seq, stime=turn.stime,
            speaker=turn.speaker, speaker_id=turn.speaker_id, text=text,
            theme_id=theme_id, similarity=sim,
        ))

    # 2) Position — UNIQUEMENT par jugement LLM, et seulement pour les claims COUVERTS
    #    (juger la position d'un claim contre la cible d'un thème auquel il n'appartient
    #    pas produirait une position sur une question qui ne lui est pas posée).
    if cfg.stance_enabled:
        by_theme: dict[str, list[int]] = {}
        for i, a in enumerate(made):
            if a.covered:
                by_theme.setdefault(a.theme_id, []).append(i)
        model = cfg.profile.model_for("opinion")
        for theme_id, positions in by_theme.items():
            cible = state.themes[theme_id].cleavage or state.themes[theme_id].label
            items = [(i, made[i].text) for i in positions]
            got = stance_layer.run_stance(
                cible, items, model=model,
                reported_speech_guard=cfg.reported_speech_guard)
            for i in positions:
                rec = got.get(i)
                if rec:
                    made[i].stance = rec["stance"]
                    made[i].confidence = rec["confidence"]
                    made[i].justif = rec["justif"]

    for a in made:
        state.add(a)

    covered = sum(1 for a in made if a.covered)
    log(f"[live] tour {turn.seq} · {turn.speaker[:28]} · {len(made)} claims "
        f"({covered} couverts)")
    return made


__all__ = ["LiveConfig", "bootstrap", "process_turn", "CLEAVAGE_SAMPLE"]

"""Endpoint `/claims` — thèmes ÉMERGENTS d'un dataset (pipeline avis→claims→cluster).

Câble `pipeline.claims.run_claims` sur les caches du backend, avec DEUX niveaux
de cache disque par dataset :

  - `backend/cache/<dataset>/claims.json`     extraction LLM (LENTE ~2 s/avis) ;
  - `backend/cache/<dataset>/claims_emb.npz`  embeddings des claims (nomic, CPU).

Conséquence (acceptance) : le 1er run extrait + embed ; les suivants — y compris
un changement de RÉSOLUTION — rejouent le clustering SANS ré-extraire ni ré-embed.
L'extraction n'appelle le Mac (`AGORA_OLLAMA_URL`) QUE pour les avis manquants ;
si le Mac est injoignable on lève une erreur claire (l'API renvoie 503).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from pipeline.claims.backend import BackendUnavailable, ClaimBackend, resolve_backend
from pipeline.claims.extract import extract_claims
from pipeline.claims.ollama import OllamaStats
from pipeline.claims.span import Claim, as_claim
from pipeline.claims.pipeline import (
    DEFAULT_EMBEDDER,
    DEFAULT_SEED,
    Avis,
    _flatten,
    cluster_claims,
    embed_claim_texts,
)
from backend.recluster import dataset_dir

CLAIMS_NAME = "claims.json"
CLAIMS_EMB_NAME = "claims_emb.npz"
DEFAULT_MIN_CHARS = 12


class OllamaUnavailable(RuntimeError):
    """Backend d'extraction inutilisable (Mac injoignable, clé absente…) — 503 clair.

    Conservé pour compat ; `BackendUnavailable` (générique) en est un alias logique.
    """


def _avis_from_ideas(ideas: list, min_chars: int) -> list[Avis]:
    out: list[Avis] = []
    for idea in ideas:
        text = (getattr(idea, "text_clean", "") or getattr(idea, "text", "") or "").strip()
        if len(text) < min_chars:
            continue
        out.append(Avis(id=str(idea.id), text=text,
                        weight=float(getattr(idea, "weight", 1.0) or 1.0)))
    return out


def _load_claims_cache(path: Path, model: str) -> dict[str, list[Claim]]:
    """Charge l'extraction cachée (claims ancrés) si elle matche le modèle, sinon {}.

    Le cache sérialise des `Claim` en dicts `{text,start,end}` ; on les renormalise.
    """
    if not path.exists():
        return {}
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if rec.get("model") != model:
        return {}  # modèle différent → ré-extraire (claims dépendent du LLM)
    claims = rec.get("claims")
    if not isinstance(claims, dict):
        return {}
    return {aid: [as_claim(c) for c in (lst or [])] for aid, lst in claims.items()}


def _save_claims_cache(path: Path, model: str, claims: dict[str, list[Claim]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {aid: [as_claim(c).to_dict() for c in lst] for aid, lst in claims.items()}
    path.write_text(json.dumps({"model": model, "claims": serializable}, ensure_ascii=False),
                    encoding="utf-8")


def _emb_fingerprint(embedder: str, claim_texts: list[str]) -> str:
    blob = embedder + "\x00" + "\x00".join(claim_texts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_emb_cache(path: Path, fingerprint: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=False)
        if str(d["fingerprint"]) == fingerprint:
            return d["vecs"].astype(np.float64)
    except (OSError, KeyError, ValueError):
        return None
    return None


def _save_emb_cache(path: Path, fingerprint: str, vecs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vecs=vecs.astype(np.float32), fingerprint=np.str_(fingerprint))


@dataclass
class PreparedClaims:
    """Sortie de `prepare_claims` : claims extraits + embeddings, prêts à clusteriser.

    Mutualise les DEUX étapes lentes et CACHÉES (extraction LLM + embed nomic) entre
    tous les endpoints qui partent des thèmes émergents (`/claims`, `/analysis`,
    `/insights`, `/citations`). Les listes `claim_*` sont aplaties et ALIGNÉES (même
    index) sur `claim_vecs`, dans l'ordre des avis.
    """
    avis: list[Avis]
    claims_by_id: dict[str, list[Claim]]
    claim_texts: list[str]
    claim_owner: list[int]          # claim idx -> idx d'avis dans `avis`
    claim_weight: np.ndarray        # poids social par claim (hérité de l'avis)
    claim_vecs: np.ndarray          # embeddings L2-normalisés, alignés aux claims
    claim_spans: list[list[tuple[int, int]]]   # spans verbatim par claim (1..N portions)
    claim_target: list[tuple[int, int] | None]  # cible verbatim par claim (ou None)
    backend: ClaimBackend
    model: str
    embedder: str
    min_chars: int
    extracted: int                  # nb d'avis ré-extraits (0 = tout en cache)
    embedded: bool                  # embeddings recalculés ? (False = cache)
    cold_seconds: float             # temps d'extraction LLM (0 si cache complet)

    def meta(self) -> dict:
        """Bloc `meta` commun (backend/cache/coût) — sans `took_ms` ni `dataset`."""
        return {
            "backend": self.backend.name,      # `api` (Mistral UE) | `mac` (souverain local)
            "sovereign": self.backend.sovereign,
            "data_note": self.backend.note,
            "model": self.model,
            "embedder": self.embedder,
            "min_chars": self.min_chars,
            "n_avis": len(self.avis),
            "n_claims": len(self.claim_texts),
            "cache": {
                "claims_extracted": self.extracted,
                "claims_cached": len(self.avis) - self.extracted,
                "embeddings_recomputed": self.embedded,
            },
            "cost": {"cold_seconds": self.cold_seconds},
        }


def prepare_claims(
    ds,
    *,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    ollama_url: str | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> PreparedClaims:
    """Extrait (caché) puis embed (caché) les claims d'un dataset, sans clusteriser.

    Étapes 1 (extraction LLM) + 2 (embed) du pipeline, isolées pour être RÉUTILISÉES
    par `/analysis`, `/insights`, `/citations` (qui ont besoin des vecteurs internes,
    pas seulement du dict de sortie de `cluster_claims`). `ds` porte `.id` et `.ideas`.
    `backend` choisit le moteur (``api`` défaut, ``mac``, ``auto``). Le cache claims est
    clé par MODÈLE. Lève `OllamaUnavailable` si une extraction est nécessaire mais le
    backend est inutilisable (clé absente, Mac injoignable).
    """
    ollama_url = ollama_url or os.environ.get("AGORA_OLLAMA_URL")
    avis = _avis_from_ideas(ds.ideas, min_chars)
    if not avis:
        raise ValueError(f"Aucun avis ≥ {min_chars} caractères dans le dataset {ds.id!r}.")

    ddir = dataset_dir(ds.id)
    emb_path = ddir / CLAIMS_EMB_NAME

    # 1) Extraction (cachée). On résout le backend pour connaître le MODÈLE (clé de cache)
    #    et n'extraire que les avis manquants. La résolution est paresseuse côté réseau :
    #    `api` ne valide que la présence de la clé, `mac`/`auto` ne sondent qu'à l'usage.
    be = resolve_backend(backend, ollama_url=ollama_url, model=model)
    model = be.model
    claims_path = ddir / CLAIMS_NAME
    claims_by_id = _load_claims_cache(claims_path, model)
    missing = [a for a in avis if a.id not in claims_by_id]
    extracted = len(missing)
    cold_seconds = 0.0
    if missing:
        stats = OllamaStats()
        try:
            new = extract_claims(missing, backend=be, stats=stats)
        except BackendUnavailable as exc:
            raise OllamaUnavailable(str(exc)) from exc
        claims_by_id.update(new)
        cold_seconds = round(stats.cold_seconds, 2)
        _save_claims_cache(claims_path, model, claims_by_id)

    # 2) Embeddings des claims (cachés, alignés à l'ordre d'aplatissement).
    claim_texts, claim_owner, claim_weight, claim_spans, claim_target = _flatten(avis, claims_by_id)
    fingerprint = _emb_fingerprint(embedder, claim_texts)
    claim_vecs = _load_emb_cache(emb_path, fingerprint)
    embedded = claim_vecs is None
    if claim_vecs is None:
        claim_vecs = embed_claim_texts(claim_texts, embedder=embedder)
        _save_emb_cache(emb_path, fingerprint, claim_vecs)

    return PreparedClaims(
        avis=avis,
        claims_by_id=claims_by_id,
        claim_texts=claim_texts,
        claim_owner=claim_owner,
        claim_weight=np.asarray(claim_weight, dtype=np.float64),
        claim_vecs=np.asarray(claim_vecs, dtype=np.float64),
        claim_spans=claim_spans,
        claim_target=claim_target,
        backend=be,
        model=model,
        embedder=embedder,
        min_chars=min_chars,
        extracted=extracted,
        embedded=embedded,
        cold_seconds=cold_seconds,
    )


def claims_payload(
    ds,
    *,
    resolution: float = 1.0,
    backend: str | None = None,
    model: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    ollama_url: str | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Calcule (ou rejoue depuis le cache) la carte des thèmes émergents d'un dataset.

    `ds` est un `_Dataset` du serveur (porte `.id` et `.ideas`). `backend` choisit le
    moteur d'extraction (``api`` par défaut, ``mac``, ``auto`` ; sinon `AGORA_CLAIMS_BACKEND`).
    Le cache claims est clé par MODÈLE → API et Mac ne se mélangent pas. Lève
    `BackendUnavailable` si une extraction est nécessaire mais le backend est inutilisable
    (clé absente, Mac injoignable).
    """
    t0 = perf_counter()
    prepared = prepare_claims(
        ds, backend=backend, model=model, embedder=embedder,
        ollama_url=ollama_url, min_chars=min_chars,
    )

    # 3) Clustering émergent (rapide, rejouable à résolution variable).
    result = cluster_claims(
        prepared.avis, prepared.claims_by_id, resolution=resolution, seed=seed,
        embedder=prepared.embedder, claim_vecs=prepared.claim_vecs,
    )

    result["meta"] = {
        "dataset": ds.id,
        **prepared.meta(),
        "took_ms": round((perf_counter() - t0) * 1000),
    }
    return result

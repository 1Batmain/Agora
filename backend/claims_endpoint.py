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
from pathlib import Path
from time import perf_counter

import numpy as np

from pipeline.claims.extract import extract_claims
from pipeline.claims.ollama import OllamaClient
from pipeline.claims.pipeline import (
    DEFAULT_EMBEDDER,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    Avis,
    cluster_claims,
    embed_claim_texts,
)
from backend.recluster import dataset_dir

CLAIMS_NAME = "claims.json"
CLAIMS_EMB_NAME = "claims_emb.npz"
DEFAULT_MIN_CHARS = 12


class OllamaUnavailable(RuntimeError):
    """Le LLM local (Mac) est injoignable — l'API doit renvoyer un 503 clair."""


def _avis_from_ideas(ideas: list, min_chars: int) -> list[Avis]:
    out: list[Avis] = []
    for idea in ideas:
        text = (getattr(idea, "text_clean", "") or getattr(idea, "text", "") or "").strip()
        if len(text) < min_chars:
            continue
        out.append(Avis(id=str(idea.id), text=text,
                        weight=float(getattr(idea, "weight", 1.0) or 1.0)))
    return out


def _load_claims_cache(path: Path, model: str) -> dict[str, list[str]]:
    """Charge l'extraction cachée si elle correspond au modèle, sinon {}."""
    if not path.exists():
        return {}
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if rec.get("model") != model:
        return {}  # modèle différent → ré-extraire (claims dépendent du LLM)
    claims = rec.get("claims")
    return claims if isinstance(claims, dict) else {}


def _save_claims_cache(path: Path, model: str, claims: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "claims": claims}, ensure_ascii=False),
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


def claims_payload(
    ds,
    *,
    resolution: float = 1.0,
    model: str = DEFAULT_MODEL,
    embedder: str = DEFAULT_EMBEDDER,
    ollama_url: str | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Calcule (ou rejoue depuis le cache) la carte des thèmes émergents d'un dataset.

    `ds` est un `_Dataset` du serveur (porte `.id` et `.ideas`). Lève
    `OllamaUnavailable` si une extraction est nécessaire mais le Mac est injoignable.
    """
    t0 = perf_counter()
    ollama_url = ollama_url or os.environ.get("AGORA_OLLAMA_URL")
    avis = _avis_from_ideas(ds.ideas, min_chars)
    if not avis:
        raise ValueError(f"Aucun avis ≥ {min_chars} caractères dans le dataset {ds.id!r}.")

    ddir = dataset_dir(ds.id)
    claims_path = ddir / CLAIMS_NAME
    emb_path = ddir / CLAIMS_EMB_NAME

    # 1) Extraction (cachée). On n'appelle le Mac que pour les avis manquants.
    claims_by_id = _load_claims_cache(claims_path, model)
    missing = [a for a in avis if a.id not in claims_by_id]
    extracted = len(missing)
    cold_seconds = 0.0
    if missing:
        client = OllamaClient(ollama_url)
        ok, think = client.warmup(model)
        if not ok:
            raise OllamaUnavailable(
                f"LLM local {model!r} injoignable — exporter AGORA_OLLAMA_URL "
                "depuis var/MAC_LOCAL_OLLAMA et vérifier que le Mac est allumé."
            )
        from pipeline.claims.ollama import OllamaStats
        stats = OllamaStats()
        new = extract_claims(missing, model=model, client=client, think=think, stats=stats)
        claims_by_id.update(new)
        cold_seconds = round(stats.cold_seconds, 2)
        _save_claims_cache(claims_path, model, claims_by_id)

    # 2) Embeddings des claims (cachés, alignés à l'ordre d'aplatissement).
    claim_texts = [c for a in avis for c in claims_by_id.get(a.id, [])]
    fingerprint = _emb_fingerprint(embedder, claim_texts)
    claim_vecs = _load_emb_cache(emb_path, fingerprint)
    embedded = claim_vecs is None
    if claim_vecs is None:
        claim_vecs = embed_claim_texts(claim_texts, embedder=embedder)
        _save_emb_cache(emb_path, fingerprint, claim_vecs)

    # 3) Clustering émergent (rapide, rejouable à résolution variable).
    result = cluster_claims(avis, claims_by_id, resolution=resolution, seed=seed,
                            embedder=embedder, claim_vecs=claim_vecs)

    result["meta"] = {
        "dataset": ds.id,
        "model": model,
        "embedder": embedder,
        "min_chars": min_chars,
        "n_avis": len(avis),
        "cache": {
            "claims_extracted": extracted,
            "claims_cached": len(avis) - extracted,
            "embeddings_recomputed": embedded,
        },
        "cost": {"cold_seconds": cold_seconds},
        "took_ms": round((perf_counter() - t0) * 1000),
    }
    return result

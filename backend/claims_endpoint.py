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
from collections.abc import Callable
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
    DEFAULT_RESOLUTION,
    DEFAULT_SEED,
    Avis,
    _flatten,
    cluster_claims,
    embed_claim_texts,
)
from backend.recluster import dataset_dir

CLAIMS_NAME = "claims.json"
CLAIMS_EMB_NAME = "claims_emb.npz"
TARGET_EMB_NAME = "target_emb.npz"
DEFAULT_MIN_CHARS = 12
# Autorise une ré-extraction qui ÉCRASE un cache de claims produit par un autre modèle.
# Fail-closed par défaut : sans ce drapeau, la divergence de modèle lève.
ALLOW_REEXTRACT = os.environ.get("AGORA_ALLOW_REEXTRACT", "").strip() == "1"


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


class ClaimsCacheModelMismatch(RuntimeError):
    """Le cache de claims existe mais a été extrait par un AUTRE modèle.

    Le modèle est la clé du cache : un `model` différent invaliderait tout et
    déclencherait une ré-extraction complète (coûteuse) qui ÉCRASE `claims.json`
    et `claims_emb.npz`. Un appelant qui oublie `model=` récupère le défaut du
    backend (`ministral-3b-latest`) et détruit ainsi un cache mistral-large sans
    un mot — d'où l'échec explicite plutôt que la ré-extraction implicite.
    """


def _cached_claims_model(path: Path) -> str | None:
    """Modèle qui a produit le cache de claims (None si absent/illisible)."""
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    model = rec.get("model")
    return str(model) if isinstance(model, str) else None


def _load_claims_cache(path: Path, model: str) -> dict[str, list[Claim]]:
    """Charge l'extraction cachée (claims ancrés) si elle matche le modèle, sinon {}.

    Le cache sérialise des `Claim` en dicts `{text,start,end}` ; on les renormalise.
    Le CONTRÔLE de cohérence du modèle est fait en amont par `prepare_claims`
    (`ClaimsCacheModelMismatch`) ; ici un modèle divergent rend simplement {}.
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


# --------------------------------------------------------------------------- #
# Embeddings des CIBLES (target) — pour le knob α du bac à sable
# --------------------------------------------------------------------------- #
def _target_texts(avis: list[Avis], claim_owner: list[int],
                  claim_target: list[tuple[int, int] | None]) -> tuple[list[str], np.ndarray]:
    """Texte VERBATIM de la cible de chaque claim (ou "" si absente) + masque booléen.

    La cible est un span `(s, e)` dans le texte de l'avis propriétaire ; on relit le
    verbatim depuis `avis[owner].text`. Aligné à l'ordre d'aplatissement des claims.
    """
    texts: list[str] = []
    mask = np.zeros(len(claim_owner), dtype=bool)
    for i, tgt in enumerate(claim_target):
        if tgt is not None:
            s, e = tgt
            t = avis[claim_owner[i]].text[s:e].strip()
            if t:
                texts.append(t)
                mask[i] = True
                continue
        texts.append("")
    return texts, mask


def _load_target_cache(path: Path, fingerprint: str
                       ) -> tuple[np.ndarray, np.ndarray] | None:
    """Charge `target_emb.npz` (vecs + mask) s'il matche l'empreinte, sinon None."""
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=False)
        if str(d["fingerprint"]) == fingerprint:
            return d["vecs"].astype(np.float64), d["mask"].astype(bool)
    except (OSError, KeyError, ValueError):
        return None
    return None


def _save_target_cache(path: Path, fingerprint: str, vecs: np.ndarray,
                       mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vecs=vecs.astype(np.float32), mask=mask.astype(bool),
             fingerprint=np.str_(fingerprint))


def _embed_targets(avis, claim_owner, claim_target, claim_vecs, *, embedder, path):
    """Embeddings des CIBLES alignés aux claims (cachés). Cible absente → vecteur nul.

    Réutilise l'embedder claims (même espace nomic-v2). Cache `target_emb.npz` clé par
    une empreinte des textes de cibles. Renvoie `(target_vecs, target_mask)` : pour les
    claims sans cible, ligne nulle et `mask=False` (le blend les laisse intacts).
    """
    target_strings, mask = _target_texts(avis, claim_owner, claim_target)
    fingerprint = _emb_fingerprint(embedder, target_strings)
    cached = _load_target_cache(path, fingerprint)
    if cached is not None:
        return cached[0], cached[1], False

    dim = claim_vecs.shape[1] if claim_vecs.size else 1
    target_vecs = np.zeros((len(target_strings), dim), dtype=np.float64)
    idx = [i for i, m in enumerate(mask) if m]
    if idx:
        embedded = embed_claim_texts([target_strings[i] for i in idx], embedder=embedder)
        target_vecs[idx] = embedded
    _save_target_cache(path, fingerprint, target_vecs, mask)
    return target_vecs, mask, True


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
    target_vecs: np.ndarray         # embeddings L2-normalisés des CIBLES (nul si absente)
    target_mask: np.ndarray         # booléen : claim a-t-il une cible embeddée ? (knob α)
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
            "targets": {
                "n_with_target": int(self.target_mask.sum()),
                "coverage": (round(float(self.target_mask.mean()), 4)
                             if self.target_mask.size else 0.0),
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
    progress: "Callable[[int, int], None] | None" = None,
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

    # Question globale de la consultation (meta.json) → CADRE la granularité d'extraction
    # (v2 anti-sur-segmentation : des sous-points d'une même facette = 1 claim). Optionnelle :
    # un dataset sans `question` retombe sur le prompt nu (généricité préservée).
    try:
        _meta = json.loads((ddir / "meta.json").read_text(encoding="utf-8"))
        question = (_meta.get("question") or "").strip() or None
    except (OSError, ValueError):
        question = None

    # 1) Extraction (cachée). On résout le backend pour connaître le MODÈLE (clé de cache)
    #    et n'extraire que les avis manquants. La résolution est paresseuse côté réseau :
    #    `api` ne valide que la présence de la clé, `mac`/`auto` ne sondent qu'à l'usage.
    be = resolve_backend(backend, ollama_url=ollama_url, model=model)
    model = be.model
    claims_path = ddir / CLAIMS_NAME
    # GARDE-FOU : un cache existant extrait par un AUTRE modèle serait écrasé en silence
    # par la ré-extraction ci-dessous. On échoue plutôt, sauf autorisation explicite.
    cached_model = _cached_claims_model(claims_path)
    if cached_model is not None and cached_model != model and not ALLOW_REEXTRACT:
        raise ClaimsCacheModelMismatch(
            f"{claims_path} a été extrait par {cached_model!r}, or ce build demande "
            f"{model!r}. Poursuivre ré-extrairait TOUS les avis et écraserait le cache.\n"
            f"→ passe `model={cached_model!r}` (ou --model) pour réutiliser le cache,\n"
            f"→ ou pose AGORA_ALLOW_REEXTRACT=1 pour ré-extraire volontairement."
        )
    claims_by_id = _load_claims_cache(claims_path, model)
    missing = [a for a in avis if a.id not in claims_by_id]
    extracted = len(missing)
    cold_seconds = 0.0
    if missing:
        stats = OllamaStats()
        try:
            new = extract_claims(missing, backend=be, stats=stats, progress=progress, question=question)
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
    # float32 suffit pour le cosinus → ÷2 la RAM des vecteurs (le blend/clustering
    # qui a besoin de float64 upcaste localement, cf. sandbox._graph_ctx).
    claim_vecs = np.asarray(claim_vecs, dtype=np.float32)

    # 2b) Embeddings des CIBLES (cachés, alignés) — alimentent le knob α du bac à sable.
    #     Cible absente → vecteur nul + mask False (blend gracieux : claim inchangé).
    target_path = ddir / TARGET_EMB_NAME
    target_vecs, target_mask, target_embedded = _embed_targets(
        avis, claim_owner, claim_target, claim_vecs, embedder=embedder, path=target_path,
    )

    return PreparedClaims(
        avis=avis,
        claims_by_id=claims_by_id,
        claim_texts=claim_texts,
        claim_owner=claim_owner,
        claim_weight=np.asarray(claim_weight, dtype=np.float64),
        claim_vecs=claim_vecs,
        claim_spans=claim_spans,
        claim_target=claim_target,
        target_vecs=np.asarray(target_vecs, dtype=np.float32),  # cohérent claim_vecs, ÷2 RAM
        target_mask=np.asarray(target_mask, dtype=bool),
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
    resolution: float = DEFAULT_RESOLUTION,
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

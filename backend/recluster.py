"""Re-clustering LIVE sur les embeddings CACHÉS (jamais ré-embeddés).

Cœur du serveur :8010. À partir des vecteurs nomic-v2 en cache, applique la
chaîne du contrat — `min_chars` → `dedup` → k-NN(`k`,`threshold`) → Leiden
**hiérarchique** (macro/sub) → scoring → naming TF-IDF → **GraphPayload
hiérarchique** — en RÉUTILISANT `pipeline.cluster.*`. Aucun appel au modèle torch.

Le payload a la même shape que `data/graph.json` (`meta, nodes, links, themes`),
augmenté de `meta.stats { n_macros, n_subs, n_nodes, modularity, took_ms }`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from backend.consultation_schema import Consultation
from pipeline.cluster.io import Idea
from pipeline.cluster.naming_methods import DEFAULT_NAMING, NAMING_METHODS

# Méthodes de NOMMAGE switchables (orthogonales au clustering). "ctfidf" = défaut
# (rétro-compat) ; "centroid" = verbatim représentatif ; "llm" = titre via API Mistral.
NAMINGS = NAMING_METHODS
DEFAULT_NAMING_METHOD = DEFAULT_NAMING

# Cache MULTI-DATASET : `backend/cache/<dataset>/{embeddings.npy, ideas.jsonl,
# meta.json}`. Un dataset = un sous-dossier (aucun nom de corpus codé en dur ;
# les datasets sont DÉCOUVERTS en scannant le dossier). Défaut rétro-compat =
# "tiktok".
CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_DATASET = "tiktok"

# Descripteurs d'ingestion (un par consultation). Les consultations OUVERTES
# (status:"open") n'ont pas encore de cache d'analyse : on les découvre ici.
DESCRIPTORS_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "ingest" / "descriptors"

MODEL_ID = "nomic-ai/nomic-embed-text-v2-moe"

EMB_NAME = "embeddings.npy"
IDEAS_NAME = "ideas.jsonl"
META_NAME = "meta.json"


def dataset_dir(dataset: str) -> Path:
    return CACHE_DIR / dataset


def cache_paths(dataset: str) -> tuple[Path, Path, Path]:
    d = dataset_dir(dataset)
    return d / EMB_NAME, d / IDEAS_NAME, d / META_NAME


def list_datasets() -> list[str]:
    """Datasets disponibles = sous-dossiers de cache/ avec un cache complet.

    Découverte pure (zéro littéral de corpus) : on liste les dossiers qui
    contiennent à la fois `embeddings.npy` et `ideas.jsonl`. Triés avec le défaut
    (`tiktok`) en tête pour la rétro-compat de l'UI.
    """
    if not CACHE_DIR.exists():
        return []
    found = [
        p.name for p in CACHE_DIR.iterdir()
        if p.is_dir() and (p / EMB_NAME).exists() and (p / IDEAS_NAME).exists()
    ]
    found.sort(key=lambda n: (n != DEFAULT_DATASET, n))
    return found


def _read_descriptor_file(name: str) -> dict | None:
    """Lit un descripteur d'ingestion `pipeline/ingest/descriptors/<name>.json`."""
    p = DESCRIPTORS_DIR / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_open_consultations() -> list[str]:
    """Consultations OUVERTES = descripteurs `status:"open"` SANS cache d'analyse.

    Découverte pure (zéro littéral de corpus) : scanne les descripteurs et garde
    ceux marqués `status:"open"` qui ne sont PAS déjà des datasets cachés. Une
    consultation neuve n'a ni `embeddings.npy` ni `ideas.jsonl`, donc elle échappe
    à `list_datasets()` — on la liste ici pour que la landing affiche sa carte
    « Ouvert » et route vers la vue Participer.
    """
    if not DESCRIPTORS_DIR.exists():
        return []
    cached = set(list_datasets())
    found = [
        p.stem
        for p in sorted(DESCRIPTORS_DIR.glob("*.json"))
        if p.stem not in cached and (_read_descriptor_file(p.stem) or {}).get("status") == "open"
    ]
    return found


def open_consultation_descriptor(name: str) -> Consultation:
    """Descripteur `Consultation` d'une consultation OUVERTE (sans cache d'analyse).

    Même schéma que `dataset_descriptor` (pour peupler `/datasets`), enrichi de
    `question`/`context` (sujet affiché dans la vue Participer). Pour une ouverte,
    toutes les contributions reçues SONT l'échantillon : `n_sample == n_contributions`
    == nombre de contributions déjà reçues (seed + live), pour l'affichage de la carte.
    """
    from backend.submissions import count_submissions  # local : évite torch au boot

    d = _read_descriptor_file(name) or {}
    n = count_submissions(name)
    return {
        "id": name,
        "label": d.get("label", name),
        "status": "open",
        "n_sample": n,
        "n_contributions": n,
        "n_nodes": n,  # rétro-compat : == n_sample
        "languages": [],
        "lang_counts": {},
        "source": name,
        "question": d.get("question", ""),
        "context": d.get("context", ""),
    }


def load_cache(dataset: str = DEFAULT_DATASET) -> tuple[list[Idea], np.ndarray, np.ndarray]:
    """Charge le cache d'UN dataset (vecteurs + ideas alignés). Aucun torch."""
    emb_path, ideas_path, _ = cache_paths(dataset)
    if not emb_path.exists() or not ideas_path.exists():
        raise RuntimeError(
            f"Cache absent ({emb_path}). Construis-le d'abord :\n"
            f"  uv run --extra embed-contender python -m backend.build_cache --dataset {dataset}"
        )
    vecs = np.load(emb_path).astype(np.float32)
    ideas: list[Idea] = []
    with open(ideas_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                ideas.append(Idea.from_row(json.loads(line), i))
    if len(ideas) != vecs.shape[0]:
        raise RuntimeError(
            f"Cache désaligné ({dataset}) : {len(ideas)} ideas vs {vecs.shape[0]} vecteurs."
        )
    weights = np.array([idea.weight for idea in ideas], dtype=np.float32)
    return ideas, vecs, weights


def dataset_descriptor(dataset: str, ideas: list[Idea] | None = None) -> Consultation:
    """Constructeur UNIQUE d'une `Consultation` clôturée pour `GET /datasets`.

    Lit `meta.json` s'il existe (écrit par build_cache), sinon DÉRIVE tout des
    `ideas` cachés (langues, n, source). Générique : aucune valeur en dur.
    C'est LA source de vérité du schéma servi (mirroré côté front).
    """
    _, ideas_path, meta_path = cache_paths(dataset)
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}

    if ideas is None and (not meta or "languages" not in meta or "n_nodes" not in meta):
        ideas = []
        if ideas_path.exists():
            with open(ideas_path, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if line:
                        ideas.append(Idea.from_row(json.loads(line), i))

    if ideas is not None:
        lang_counts = Counter(idea.lang for idea in ideas if idea.lang)
        src_counts = Counter(idea.source for idea in ideas if idea.source)
        derived = {
            "n_nodes": len(ideas),
            "languages": [lg for lg, _ in lang_counts.most_common()],
            "lang_counts": dict(lang_counts.most_common()),
            "source": src_counts.most_common(1)[0][0] if src_counts else dataset,
            # Voix RÉELLES portées par les textes analysés (doublons exacts inclus via le
            # poids de dédup) — distinct de n_loaded (participants) et n_nodes (textes uniques).
            "n_responses": int(round(sum(getattr(i, "weight", 1.0) or 1.0 for i in ideas))),
        }
    else:
        derived = {}

    # Échantillon réellement analysé (ancien `n_nodes`).
    n_sample = meta.get("n_nodes", derived.get("n_nodes", 0))
    out: Consultation = {
        "id": dataset,
        "label": meta.get("label", dataset),
        # Statut de consultation : "open" | "closed". Défaut prudent "closed"
        # (les caches déjà construits sans ce champ restent en analyse seule).
        "status": meta.get("status", "closed"),
        "n_sample": n_sample,
        # Nombre RÉEL de contributions reçues (avant le cap d'échantillonnage à
        # n_sample pour le build). `meta["n_loaded"]` = total chargé à l'ingestion ;
        # repli sur n_sample (datasets non capés, ex. tiktok).
        "n_contributions": meta.get("n_loaded", n_sample),
        "n_nodes": n_sample,  # rétro-compat : == n_sample (lu par pytest /datasets)
        "languages": meta.get("languages", derived.get("languages", [])),
        "lang_counts": meta.get("lang_counts", derived.get("lang_counts", {})),
        "source": meta.get("source", derived.get("source", dataset)),
    }
    # Réponses réelles à la question (voix du CORPUS, pré-échantillonnage) — servi SEULEMENT
    # si connu (meta écrite par build_cache, ou dérivé des ideas d'un corpus non capé). Pas
    # de fallback n_sample : un dénominateur inventé fausserait la note d'échantillon du front.
    n_responses = meta.get("n_responses", derived.get("n_responses"))
    if n_responses and (meta.get("built_with", {}).get("cap") is None or "n_responses" in meta):
        out["n_responses"] = n_responses
    # Sujet affiché (consultations clôturées qui en exposent un dans meta.json).
    if meta.get("question"):
        out["question"] = meta["question"]
    if meta.get("context"):
        out["context"] = meta["context"]
    # Point de comparaison OFFICIEL (coût/durée du dispositif d'origine, sourcé) — porté par
    # le descripteur du dataset (générique, aucun corpus en dur) ; affiché par l'overview.
    if meta.get("official_baseline"):
        out["official_baseline"] = meta["official_baseline"]
    # Hiérarchie MÈRE→ENFANTS (cf. backend.build_children) : un enfant porte
    # `parent_id`, une mère porte `children`. Servis tels quels par /datasets.
    if meta.get("parent_id"):
        out["parent_id"] = meta["parent_id"]
    if meta.get("children"):
        out["children"] = meta["children"]
    return out

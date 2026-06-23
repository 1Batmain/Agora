"""Embeddings contextuels PAR TOKEN (last_hidden_state, AVANT pooling) + repli.

Cœur de faisabilité du banc : on veut un vecteur PAR token de l'avis (pas un seul
vecteur pour l'avis entier), pour mesurer où le sens bascule À L'INTÉRIEUR du texte.

- `nomic-ai/nomic-embed-text-v2-moe` (DÉFAUT, cohérent avec la prod) expose les
  token-embeddings via `model.encode(text, output_value='token_embeddings')`.
- Repli documenté : `intfloat/multilingual-e5-base` / `BAAI/bge-m3` (token-embeddings
  garantis par sentence-transformers). On résout préfixe + trust_remote_code via le
  registre de PROD (`pipeline.embed.registry`) — zéro convention dupliquée.

On garde la correspondance token → offset char (offset_mapping du tokenizer) pour
mapper les frontières détectées sur le texte et sur des UNITÉS-MOTS (granularité de
scoring Pk/WindowDiff). Le segmenteur travaille sur des vecteurs-MOTS = moyenne des
vecteurs-tokens du mot ; la fenêtre glissante = moyenne des vecteurs-mots.

Cache disque (`.cache/`, gitignoré) : (model_id, texte) → npz. Aucun ré-embed inutile.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from pipeline.embed.registry import get_spec, resolve_model_id

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

# Mot = suite de non-espaces. La granularité de SCORING (Pk/WindowDiff) et l'unité
# de la fenêtre glissante. Langue-agnostique (pas de règle FR/EN en dur).
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class WordUnits:
    """Avis découpé en unités-MOTS, chacune portée par un vecteur contextuel.

    - `words`       : liste des mots (texte).
    - `spans`       : (start, end) char de chaque mot dans le texte.
    - `vectors`     : [n_words, dim], L2-normalisé (moyenne des tokens du mot).
    - `text`        : texte source.
    """

    words: list[str]
    spans: list[tuple[int, int]]
    vectors: np.ndarray
    text: str

    def __len__(self) -> int:
        return len(self.words)

    def boundary_word_index(self, char_offset: int) -> int:
        """Indice d'unité-mot correspondant à une frontière à `char_offset`.

        = nombre de mots qui COMMENCENT avant l'offset (la frontière tombe à une
        jointure de segments, donc entre deux mots). Borne dans [0, n_words].
        """
        idx = 0
        for s, _ in self.spans:
            if s < char_offset:
                idx += 1
            else:
                break
        return idx


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    """Charge le SentenceTransformer (paresseux, CPU, 1 fois par model_id)."""
    from sentence_transformers import SentenceTransformer

    spec = get_spec(model_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SentenceTransformer(
            spec.model_id, trust_remote_code=spec.trust_remote_code, device="cpu"
        )
    return model


def _split_words(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words, spans = [], []
    for m in _WORD_RE.finditer(text):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans


def _cache_path(model_id: str, text: str) -> Path:
    h = hashlib.sha1(f"{model_id}\x00{text}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"wu_{h}.npz"


def embed_word_units(text: str, model_id: str = "nomic-v2", *, use_cache: bool = True) -> WordUnits:
    """Token-embeddings → vecteurs-MOTS alignés sur les offsets char du texte.

    1. tokenize `prefix + text` (préfixe doc du registre) en gardant offset_mapping ;
    2. récupère les token-embeddings (last_hidden_state, AVANT pooling) ;
    3. jette les tokens spéciaux et ceux du préfixe ;
    4. agrège les tokens par mot (moyenne) → vecteur-mot L2-normalisé.
    """
    resolved = resolve_model_id(model_id)
    words, spans = _split_words(text)
    if not words:
        return WordUnits([], [], np.zeros((0, 1), dtype=np.float32), text)

    cache_file = _cache_path(resolved, text)
    if use_cache and cache_file.exists():
        d = np.load(cache_file)
        return WordUnits(words, spans, d["vectors"].astype(np.float32), text)

    model = _load_model(resolved)
    spec = get_spec(resolved)
    prefix = spec.doc_prefix
    prefixed = prefix + text

    tok = model.tokenizer
    enc = tok(prefixed, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    special = tok.get_special_tokens_mask(enc["input_ids"], already_has_special_tokens=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok_emb = model.encode(prefixed, output_value="token_embeddings")
    tok_emb = np.asarray(tok_emb, dtype=np.float32)
    if tok_emb.shape[0] != len(offsets):
        raise RuntimeError(
            f"Désalignement tokens/offsets ({tok_emb.shape[0]} vs {len(offsets)}) "
            f"pour {resolved}. Vérifier la convention de tokenisation."
        )

    plen = len(prefix)
    dim = tok_emb.shape[1]
    # Accumulateur par mot : somme des vecteurs-tokens tombant dans le span du mot.
    sums = np.zeros((len(words), dim), dtype=np.float32)
    counts = np.zeros(len(words), dtype=np.int32)
    wi = 0  # curseur sur les mots (offsets croissants)
    for t, (s, e) in enumerate(offsets):
        if special[t] or (s == 0 and e == 0):
            continue  # token spécial (CLS/EOS/pad)
        if e <= plen:
            continue  # token du préfixe d'instruction
        cs = s - plen  # offset char dans le TEXTE (préfixe retiré)
        # Avance le curseur mot jusqu'à englober le début du token.
        while wi < len(words) - 1 and cs >= spans[wi][1]:
            wi += 1
        sums[wi] += tok_emb[t]
        counts[wi] += 1

    # Mots sans token (rare : ponctuation isolée fusionnée) → repli sur le mot voisin.
    for i in range(len(words)):
        if counts[i] == 0:
            j = i - 1 if i > 0 else i + 1
            j = min(max(j, 0), len(words) - 1)
            sums[i] = sums[j]
            counts[i] = max(counts[j], 1)
    vectors = sums / counts[:, None]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = (vectors / norms).astype(np.float32)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, vectors=vectors)
    return WordUnits(words, spans, vectors, text)


@lru_cache(maxsize=1)
def _doc_embedder(model_id: str):
    """Embedder de PROD (vecteur d'avis poolé) — même espace que le cache tiktok."""
    from pipeline.embed.embedder import Embedder

    return Embedder(model_id=resolve_model_id(model_id))


def embed_docs(texts: list[str], model_id: str = "nomic-v2") -> np.ndarray:
    """Vecteurs d'avis poolés (espace PROD), pour les centroïdes de taxonomie."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    return _doc_embedder(model_id).embed(texts).astype(np.float32)


def feasibility_probe(model_id: str = "nomic-v2") -> dict:
    """Sonde la disponibilité des token-embeddings d'un modèle (pour le report)."""
    resolved = resolve_model_id(model_id)
    spec = get_spec(resolved)
    info = {"model_id": resolved, "doc_prefix": spec.doc_prefix,
            "trust_remote_code": spec.trust_remote_code}
    try:
        wu = embed_word_units("Test de segmentation sémantique sur deux idées.",
                              model_id=resolved, use_cache=False)
        info["ok"] = True
        info["n_words"] = len(wu)
        info["dim"] = int(wu.vectors.shape[1])
    except Exception as exc:  # noqa: BLE001 — on RAPPORTE l'échec, on ne le masque pas
        info["ok"] = False
        info["error"] = repr(exc)[:300]
    return info

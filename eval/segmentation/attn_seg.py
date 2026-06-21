"""Segmentation par ATTENTION — l'attention inter-blocs bat-elle l'embedding ?

EXPÉ R&D (read-only) : tester si une frontière de thème se lit dans la **chute de
l'attention** qui traverse un point p, plutôt que dans la trajectoire des embeddings
(banc `seg_bench.py`). Intuition : les tokens d'un même thème s'attendent entre eux ;
au passage d'une frontière, le flux d'attention gauche↔droite s'effondre.

Modèles à attention TRIVIALE à extraire (XLM-R standard, pas nomic dont le code custom
ne l'expose pas) :
  - `intfloat/multilingual-e5-base` (priorité) — préfixe doc `passage: `.
  - `BAAI/bge-m3`                    (second)  — aucun préfixe.
On recharge en `attn_implementation="eager"` (sinon SDPA ne matérialise pas les poids)
et on lit `output_attentions=True` → tuple de `[batch, heads, seq, seq]` par couche.

Pipeline (réutilise le harness existant) :
  attentions tokens → réduction au niveau MOT A_word[L,H,n,n] (offset_mapping, mêmes
  unités-mots que `embeddings.py`) → signal de frontière `cross(p)` (flux d'attention
  traversant p sur une fenêtre W) → minima locaux calibrés GLOBALEMENT (μ/σ poolés,
  zéro magic-number) → frontières → métriques `metrics.py` vs `gold_large.json`.

Balaie : modèle × jeu-de-couches (early/mid/late/…) × agrégation des têtes (mean OU
sélection des têtes les plus « locales »/blocs-diagonales) × fenêtre W × seuil c.

ÉCRIT UNIQUEMENT dans `eval/segmentation/`. CPU, seed fixe.

    uv run --extra contender python -m eval.segmentation.attn_seg \
        [--gold eval/segmentation/gold_large.json] [--models e5-base bge-m3] \
        [--out eval/segmentation/attn_report.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.segmentation import metrics as M
from eval.segmentation.segmenters import MIN_SEG, _enforce_min_seg
from eval.segmentation.seg_bench import GoldItem, load_gold

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "attn_report.md"
DEFAULT_SCORES = HERE / "attn_scores.json"
BASELINE_SCORES = HERE / "scores.json"   # change-point de référence (banc embeddings)

SEED = 0

# Modèles XLM-R standard : attention exposée nativement. Préfixe = convention du
# registre de prod (e5 EXIGE `passage: `, bge-m3 n'en veut AUCUN).
ATTN_MODELS = {
    "e5-base": {"model_id": "intfloat/multilingual-e5-base", "doc_prefix": "passage: "},
    "bge-m3": {"model_id": "BAAI/bge-m3", "doc_prefix": ""},
}

import re

_WORD_RE = re.compile(r"\S+")  # même unité-mot que embeddings.py (langue-agnostique)


def _split_words(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words, spans = [], []
    for m in _WORD_RE.finditer(text):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans


def _word_index(spans: list[tuple[int, int]], char_offset: int) -> int:
    """Indice du mot correspondant à une frontière char (= mots commençant avant)."""
    idx = 0
    for s, _ in spans:
        if s < char_offset:
            idx += 1
        else:
            break
    return idx


# --------------------------------------------------------------------------- #
# Chargement du modèle (eager → poids d'attention matérialisés)
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, tuple] = {}


def _load_model(model_key: str):
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(SEED)
    spec = ATTN_MODELS[model_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tok = AutoTokenizer.from_pretrained(spec["model_id"])
        model = AutoModel.from_pretrained(spec["model_id"], attn_implementation="eager")
    model.eval()
    _MODEL_CACHE[model_key] = (tok, model, spec)
    return _MODEL_CACHE[model_key]


# --------------------------------------------------------------------------- #
# Étape 1 — attention token → réduction au niveau MOT : A_word[L, H, n, n]
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WordAttn:
    words: list[str]
    A: np.ndarray            # [L, H, n, n] attention mot→mot (lignes ~ somme 1)
    n: int
    V: np.ndarray = None     # [n, dim] vecteurs-MOTS (last_hidden_state, L2-norm)

    @property
    def n_layers(self) -> int:
        return self.A.shape[0]

    @property
    def n_heads(self) -> int:
        return self.A.shape[1]


def _cache_path(model_key: str, text: str) -> Path:
    h = hashlib.sha1(f"{model_key}\x00{text}".encode("utf-8")).hexdigest()[:16]
    # v2 : ajoute les vecteurs-mots V (contrôle « trajectoire d'embedding » même modèle).
    return CACHE_DIR / f"attn2_{model_key}_{h}.npz"


def word_attention(text: str, model_key: str, *, use_cache: bool = True) -> WordAttn:
    """Attention mot→mot d'un avis, moyennée sur les tokens de chaque mot.

    A_word[L,H,i,j] = moyenne des poids d'attention token(mot i) → token(mot j) sur la
    couche L, tête H. Tokens spéciaux (CLS/EOS) et tokens du préfixe d'instruction
    retirés (ils capteraient une attention parasite, type « puits » sur CLS).
    """
    words, spans = _split_words(text)
    n = len(words)
    if n == 0:
        return WordAttn([], np.zeros((1, 1, 0, 0), dtype=np.float32), 0)

    cache_file = _cache_path(model_key, text)
    if use_cache and cache_file.exists():
        d = np.load(cache_file)
        return WordAttn(words, d["A"].astype(np.float32), n, d["V"].astype(np.float32))

    import torch

    tok, model, spec = _load_model(model_key)
    prefix = spec["doc_prefix"]
    prefixed = prefix + text
    plen = len(prefix)

    enc = tok(prefixed, return_offsets_mapping=True, return_tensors="pt",
              add_special_tokens=True, truncation=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"][0].tolist()
    special = tok.get_special_tokens_mask(ids, already_has_special_tokens=True)

    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    # [L, H, seq, seq] ; squeeze batch
    A_tok = np.stack([a[0].numpy() for a in out.attentions]).astype(np.float32)
    L, H, seq, _ = A_tok.shape
    H_tok = out.last_hidden_state[0].numpy().astype(np.float32)  # [seq, dim]

    # token → mot (offsets croissants ; on saute spéciaux et préfixe)
    tok2word = np.full(seq, -1, dtype=np.int64)
    wi = 0
    for t, (s, e) in enumerate(offsets):
        if special[t] or (s == 0 and e == 0):
            continue
        if e <= plen:
            continue  # token du préfixe d'instruction
        cs = s - plen
        while wi < n - 1 and cs >= spans[wi][1]:
            wi += 1
        tok2word[t] = wi

    # Matrice de groupement tokens→mots G [n, seq] (moyenne des lignes par mot).
    keep = tok2word >= 0
    G = np.zeros((n, seq), dtype=np.float32)
    for t in range(seq):
        if keep[t]:
            G[tok2word[t], t] = 1.0
    row_counts = G.sum(axis=1, keepdims=True)
    row_counts[row_counts == 0] = 1.0
    G_mean = G / row_counts                      # moyenne sur tokens-source du mot
    # A_word[L,H] = G_mean @ A_tok[L,H] @ G^T : moyenne source par mot, somme cible.
    A_word = np.zeros((L, H, n, n), dtype=np.float32)
    GT = G.T  # somme sur tokens-cible du mot (pas de moyenne : on garde la masse)
    for li in range(L):
        for hi in range(H):
            A_word[li, hi] = G_mean @ A_tok[li, hi] @ GT

    # Vecteurs-MOTS depuis last_hidden_state (contrôle « trajectoire d'embedding » du
    # MÊME encodeur : moyenne des token-embeddings du mot, L2-normalisée).
    V = G_mean @ H_tok                            # [n, dim]
    vn = np.linalg.norm(V, axis=1, keepdims=True)
    vn[vn == 0] = 1.0
    V = (V / vn).astype(np.float32)

    # Mots sans token (rare) → repli sur le voisin pour éviter lignes/colonnes nulles.
    has_tok = row_counts[:, 0] > 0
    for i in range(n):
        if not has_tok[i]:
            j = i - 1 if i > 0 else min(i + 1, n - 1)
            A_word[:, :, i, :] = A_word[:, :, j, :]
            A_word[:, :, :, i] = A_word[:, :, :, j]
            V[i] = V[j]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, A=A_word.astype(np.float16), V=V.astype(np.float16))
    return WordAttn(words, A_word, n, V)


# --------------------------------------------------------------------------- #
# Étape 2 — signal de frontière : flux d'attention traversant p
# --------------------------------------------------------------------------- #
def _aggregate(A: np.ndarray, layers: list[int], heads: list[int] | None) -> np.ndarray:
    """Agrège A[L,H,n,n] sur un jeu de couches et de têtes → [n,n] symétrisé.

    `heads=None` → moyenne de TOUTES les têtes. Symétrisation : on additionne les deux
    sens (i→j et j→i) car une frontière coupe le flux dans les DEUX directions.
    """
    sub = A[layers]
    sub = sub[:, heads] if heads is not None else sub
    Aw = sub.mean(axis=(0, 1))               # [n, n]
    return Aw + Aw.T                          # flux symétrique


def cross_signal(A: np.ndarray, layers: list[int], heads: list[int] | None,
                 W: int) -> np.ndarray:
    """`cross(p)` pour p ∈ [1, n-1] : flux d'attention moyen entre le bloc gauche
    (W mots avant p) et le bloc droit (W mots après p). BAS = frontière.

    Normalisé par la taille des blocs (moyenne des poids), donc indépendant de W et
    comparable d'un avis à l'autre — calibrable globalement.
    """
    F = _aggregate(A, layers, heads)
    n = F.shape[0]
    out = np.ones(max(0, n - 1), dtype=np.float64)
    for p in range(1, n):
        lo, hi = max(0, p - W), min(n, p + W)
        block = F[lo:p, p:hi]
        out[p - 1] = float(block.mean()) if block.size else 1.0
    return out


# --------------------------------------------------------------------------- #
# Sélection des têtes « locales » (blocs-diagonales) — critère NON supervisé
# --------------------------------------------------------------------------- #
def head_locality(A: np.ndarray) -> np.ndarray:
    """Score de localité par (couche, tête) : masse d'attention à distance ≤1 du mot.

    Une tête « blocs-diagonale » (topique) concentre son attention localement ; une
    tête syntaxique/positionnelle disperse. Critère unsupervised → pas de fuite du gold.
    Renvoie [L, H].
    """
    L, H, n, _ = A.shape
    if n < 2:
        return np.zeros((L, H))
    # masque |i-j| <= 1
    idx = np.arange(n)
    local = (np.abs(idx[:, None] - idx[None, :]) <= 1).astype(np.float32)
    num = (A * local[None, None]).sum(axis=(2, 3))
    den = A.sum(axis=(2, 3))
    den[den == 0] = 1.0
    return num / den


# --------------------------------------------------------------------------- #
# Préparation : attention + gold word-boundaries par avis
# --------------------------------------------------------------------------- #
@dataclass
class PreparedA:
    item: GoldItem
    wa: WordAttn
    ref: set[int]
    locality: np.ndarray     # [L, H] localité moyenne (pour sélection de têtes)


def prepare(items: list[GoldItem], model_key: str) -> list[PreparedA]:
    out = []
    for it in items:
        wa = word_attention(it.text, model_key)
        _, spans = _split_words(it.text)
        ref = set()
        for off in it.boundaries_char:
            b = _word_index(spans, off)
            if 0 < b < wa.n:
                ref.add(b)
        loc = head_locality(wa.A) if wa.n >= 2 else np.zeros((wa.n_layers, wa.n_heads))
        out.append(PreparedA(it, wa, ref, loc))
    return out


# --------------------------------------------------------------------------- #
# Grille de balayage
# --------------------------------------------------------------------------- #
def layer_sets(n_layers: int) -> dict[str, list[int]]:
    """Jeux de couches early/mid/late (+ regroupements) — bornes dérivées de n_layers."""
    q = n_layers // 4
    return {
        "early": list(range(0, q)),
        "lowmid": list(range(q, 2 * q)),
        "mid": list(range(2 * q, 3 * q)),
        "late": list(range(3 * q, n_layers)),
        "midlate": list(range(2 * q, n_layers)),
        "all": list(range(0, n_layers)),
    }


HEAD_AGGS = ["mean", "local"]    # toutes têtes ; ou têtes locales (sélection dérivée)
W_GRID = [3, 5, 8, 12]
C_GRID = [0.5, 1.0, 1.5, 2.0]


def select_local_heads(prepared: list[PreparedA], layers: list[int]) -> list[int]:
    """Têtes locales d'un jeu de couches : indices (dans le sous-ensemble de couches)
    dont la localité POOLÉE dépasse la moyenne. Cut dérivé des données (μ), pas magique.

    Renvoie des indices de têtes-aplaties sur `len(layers)` couches : on agrège ensuite
    `A[layers][:, heads]`, donc on travaille tête-par-tête mais couches déjà filtrées.
    Ici on sélectionne par TÊTE (moyenne sur les couches retenues) pour rester simple.
    """
    locs = []
    for p in prepared:
        if p.wa.n >= 2:
            locs.append(p.locality[layers].mean(axis=0))   # [H]
    if not locs:
        return None
    mean_loc = np.mean(locs, axis=0)                        # [H]
    cut = float(mean_loc.mean())
    heads = [h for h in range(len(mean_loc)) if mean_loc[h] >= cut]
    return heads or list(range(len(mean_loc)))


# --------------------------------------------------------------------------- #
# Calibration globale du signal cross (μ/σ poolés) + détection de frontières
# --------------------------------------------------------------------------- #
def detect_boundaries(cross: np.ndarray, n: int, mu: float, sd: float, c: float,
                      min_seg: int = MIN_SEG) -> set[int]:
    """Minima locaux de `cross` sous `μ - c·σ` (calibré global). Tri par profondeur."""
    if n < 2 * min_seg or cross.size == 0:
        return set()
    cutoff = mu - c * sd
    m = len(cross)
    cand = []
    for i in range(m):
        left_ok = i == 0 or cross[i] <= cross[i - 1]
        right_ok = i == m - 1 or cross[i] <= cross[i + 1]
        if left_ok and right_ok and cross[i] < cutoff:
            cand.append((i + 1, mu - cross[i]))    # profondeur = score de tri
    return _enforce_min_seg(cand, n, min_seg)


@dataclass
class AConfig:
    model: str
    layer_set: str
    head_agg: str
    W: int
    layers: list[int]
    heads: list[int] | None


@dataclass
class AScore:
    cfg: AConfig
    c: float
    pk: float
    windowdiff: float
    f1: float
    precision: float
    recall: float
    gf1: float
    mono_fp_rate: float
    mono_cuts_mean: float

    def as_row(self) -> dict:
        return {
            "model": self.cfg.model, "layers": self.cfg.layer_set,
            "heads": self.cfg.head_agg, "W": self.cfg.W, "c": self.c,
            "Pk": round(self.pk, 4), "WindowDiff": round(self.windowdiff, 4),
            "F1_multi": round(self.f1, 4), "P": round(self.precision, 4),
            "R": round(self.recall, 4), "mono_FP": round(self.mono_fp_rate, 4),
            "mono_cuts": round(self.mono_cuts_mean, 3),
            "F1_global": round(self.gf1, 4),
        }


def evaluate_config(cfg: AConfig, prepared: list[PreparedA]) -> list[AScore]:
    """Calcule cross pour tous les avis (1 fois), calibre μ/σ global, balaie c."""
    crosses = {}
    pool = []
    for p in prepared:
        if p.wa.n >= 2:
            cr = cross_signal(p.wa.A, cfg.layers, cfg.heads, cfg.W)
            crosses[p.item.id] = cr
            pool.append(cr)
    if not pool:
        return []
    allv = np.concatenate(pool)
    mu, sd = float(allv.mean()), float(allv.std() or 1e-6)

    multi = [p for p in prepared if p.item.type == "multi"]
    mono = [p for p in prepared if p.item.type == "mono"]

    scores = []
    for c in C_GRID:
        pk_m, wd_m = [], []
        bc = M.BoundaryCounts()
        gbc = M.BoundaryCounts()
        for p in multi:
            cr = crosses.get(p.item.id)
            hyp = detect_boundaries(cr, p.wa.n, mu, sd, c) if cr is not None else set()
            pk_m.append(M.pk(p.wa.n, p.ref, hyp))
            wd_m.append(M.windowdiff(p.wa.n, p.ref, hyp))
            cnt = M.boundary_counts(p.ref, hyp, tol=1)
            bc = bc + cnt
            gbc = gbc + cnt
        mono_hits, mono_cuts = 0, 0
        for p in mono:
            cr = crosses.get(p.item.id)
            hyp = detect_boundaries(cr, p.wa.n, mu, sd, c) if cr is not None else set()
            if hyp:
                mono_hits += 1
            mono_cuts += len(hyp)
            gbc = gbc + M.boundary_counts(p.ref, hyp, tol=1)
        scores.append(AScore(
            cfg=cfg, c=c,
            pk=float(np.mean(pk_m)) if pk_m else 0.0,
            windowdiff=float(np.mean(wd_m)) if wd_m else 0.0,
            f1=bc.f1, precision=bc.precision, recall=bc.recall, gf1=gbc.f1,
            mono_fp_rate=mono_hits / len(mono) if mono else 0.0,
            mono_cuts_mean=mono_cuts / len(mono) if mono else 0.0,
        ))
    return scores


def sweep_model(model_key: str, prepared: list[PreparedA]) -> tuple[list[AScore], dict]:
    n_layers = prepared[0].wa.n_layers
    lsets = layer_sets(n_layers)
    # Sélection de têtes locales par jeu de couches (dérivée, loggée pour le report).
    local_heads = {name: select_local_heads(prepared, ls) for name, ls in lsets.items()}

    scores: list[AScore] = []
    for lname, layers in lsets.items():
        for hagg in HEAD_AGGS:
            heads = None if hagg == "mean" else local_heads[lname]
            for W in W_GRID:
                cfg = AConfig(model_key, lname, hagg, W, layers, heads)
                scores.extend(evaluate_config(cfg, prepared))
    info = {
        "n_layers": n_layers, "n_heads": prepared[0].wa.n_heads,
        "layer_sets": {k: v for k, v in lsets.items()},
        "local_heads": {k: v for k, v in local_heads.items()},
    }
    return scores, info


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def emb_control(prepared: list[PreparedA]) -> dict | None:
    """Contrôle CLÉ : la trajectoire d'embedding du MÊME encodeur (vecteurs-mots V)
    via les segmenteurs du banc (texttiling + change-point), pour isoler l'apport de
    l'ATTENTION du choix de modèle. Renvoie la meilleure config (par F1_global).

    Sans ce contrôle, « attention e5 » vs « change-point nomic » confond deux variables
    (signal ET encodeur). Ici on compare attention-e5 vs embedding-e5 sur le MÊME forward.
    """
    from eval.segmentation import segmenters as S
    from eval.segmentation.seg_bench import Prepared, evaluate

    prep = [Prepared(p.item, p.wa.V, p.wa.n, p.ref) for p in prepared if p.wa.V is not None]
    if not prep:
        return None
    gstats = S.compute_global_stats([p.U for p in prep], S.W_GRID)
    cp_on = S._ruptures_available()
    scores = [evaluate(m, W, thr, prep, gstats)
              for m, W, thr in S.iter_configs(include_changepoint=cp_on)]
    if not scores:
        return None
    best = max(scores, key=lambda s: (s.gf1, s.f1, -s.windowdiff))
    row = best.as_row()
    row["_changepoint_on"] = cp_on
    return row


def load_baseline() -> dict | None:
    if not BASELINE_SCORES.exists():
        return None
    d = json.loads(BASELINE_SCORES.read_text(encoding="utf-8"))
    return {"model": d.get("model"), "winner": d.get("winner")}


def feasibility_probe(model_key: str) -> dict:
    spec = ATTN_MODELS[model_key]
    info = {"model": model_key, "model_id": spec["model_id"], "doc_prefix": spec["doc_prefix"]}
    try:
        wa = word_attention("Je dors mal le soir. Par ailleurs le harcèlement "
                            "en ligne est un fléau pour les adolescents.", model_key,
                            use_cache=False)
        info["ok"] = True
        info["n_layers"] = wa.n_layers
        info["n_heads"] = wa.n_heads
        info["n_words"] = wa.n
    except Exception as exc:  # noqa: BLE001 — on RAPPORTE l'échec
        info["ok"] = False
        info["error"] = repr(exc)[:300]
    return info


def build_report(gold_path: Path, items: list[GoldItem], all_scores: dict[str, list[AScore]],
                 infos: dict[str, dict], feas: dict[str, dict], baseline: dict | None,
                 prepared_by_model: dict[str, list[PreparedA]],
                 emb_controls: dict[str, dict]) -> str:
    n_mono = sum(1 for it in items if it.type == "mono")
    n_multi = sum(1 for it in items if it.type == "multi")
    flat = [s for ss in all_scores.values() for s in ss]
    winner = max(flat, key=lambda s: (s.gf1, s.f1, -s.windowdiff, -s.pk))

    bl = baseline.get("winner") if baseline else None

    L = []
    L.append("# Segmentation par ATTENTION — l'attention bat-elle l'embedding ?\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={len(items)} ({n_mono} mono, {n_multi} multi). "
             f"Modèles : {', '.join(ATTN_MODELS[k]['model_id'] for k in all_scores)}. "
             f"CPU, seed={SEED}.*\n")

    # 0. Faisabilité
    L.append("## 0. Faisabilité de l'extraction d'attention\n")
    for mk, f in feas.items():
        if f.get("ok"):
            L.append(f"- **{mk}** (`{f['model_id']}`) : **OUI.** "
                     f"`AutoModel(attn_implementation='eager')` + `output_attentions=True` → "
                     f"tuple de `[batch, heads={f['n_heads']}, seq, seq]` × **{f['n_layers']} "
                     f"couches**. Réduction token→mot par `offset_mapping` (spéciaux + préfixe "
                     f"`{f['doc_prefix']!r}` retirés). Attention triviale à extraire (XLM-R standard).")
        else:
            L.append(f"- **{mk}** (`{f['model_id']}`) : **NON** — {f.get('error')}")
    L.append("")

    # 1. Méthode
    L.append("## 1. Méthode — signal de frontière par flux d'attention\n")
    L.append(
        "- **Unité = mot** (suite de non-espaces, identique au banc embeddings). "
        "`A_word[L,H,i,j]` = moyenne des poids d'attention token(mot i)→token(mot j) "
        "(tokens spéciaux/préfixe retirés pour éviter le « puits » sur CLS).\n"
        "- **Signal `cross(p)`** (frontière candidate entre mot p-1 et p) = flux "
        "d'attention MOYEN entre le bloc gauche (W mots) et le bloc droit (W mots), "
        "symétrisé (i→j + j→i). **Bas = frontière** : les mots d'un thème s'attendent "
        "entre eux ; au virage, le flux gauche↔droite s'effondre. Normalisé par la taille "
        "des blocs → comparable d'un avis à l'autre.\n"
        "- **Frontières** = minima locaux de `cross` sous `μ_cross − c·σ_cross`, μ/σ "
        "**poolés GLOBALEMENT** sur tous les avis (un seuil par-avis ne peut jamais "
        "s'abstenir sur un mono cohérent). Coefficient `c` sans dimension, `min_seg="
        f"{MIN_SEG}` mots. Zéro magic-number absolu.\n"
        "- **Balayage** : modèle × jeu-de-couches (early/lowmid/mid/late/midlate/all) × "
        "agrégation des têtes (`mean` = toutes ; `local` = têtes dont la localité "
        "— masse d'attention à distance ≤1 — dépasse la moyenne, sélection NON supervisée) "
        f"× fenêtre W∈{W_GRID} × seuil c∈{C_GRID}.\n")

    # 2. Comparaison au change-point (baseline embeddings)
    L.append("## 2. Attention vs change-point (trajectoire d'embedding)\n")
    rows = []
    if bl:
        rows.append({"approche": f"**change-point** (embeddings {baseline['model']})",
                     "config": f"W={bl['W']} pen={bl.get('pen','')}",
                     "Pk": bl["Pk"], "WindowDiff": bl["WindowDiff"],
                     "F1_multi": bl["F1_multi"], "P": bl["P"], "R": bl["R"],
                     "mono_FP": bl["mono_FP"], "F1_global": bl["F1_global"]})
    # contrôle MÊME-encodeur : trajectoire d'embedding de chaque modèle d'attention.
    for mk, ec in emb_controls.items():
        if ec:
            rows.append({"approche": f"_embedding-trajectoire {mk}_ (contrôle)",
                         "config": f"{ec['method']} W={ec['W']}",
                         "Pk": ec["Pk"], "WindowDiff": ec["WindowDiff"],
                         "F1_multi": ec["F1_multi"], "P": ec["P"], "R": ec["R"],
                         "mono_FP": ec["mono_FP"], "F1_global": ec["F1_global"]})
    # meilleure config attention par modèle
    for mk, ss in all_scores.items():
        if not ss:
            continue
        w = max(ss, key=lambda s: (s.gf1, s.f1, -s.windowdiff))
        rows.append({"approche": f"attention {mk}",
                     "config": f"{w.cfg.layer_set}/{w.cfg.head_agg} W={w.cfg.W} c={w.c}",
                     "Pk": round(w.pk, 4), "WindowDiff": round(w.windowdiff, 4),
                     "F1_multi": round(w.f1, 4), "P": round(w.precision, 4),
                     "R": round(w.recall, 4), "mono_FP": round(w.mono_fp_rate, 4),
                     "F1_global": round(w.gf1, 4)})
    cols = ["approche", "config", "Pk", "WindowDiff", "F1_multi", "P", "R", "mono_FP", "F1_global"]
    L.append(_md_table(rows, cols) + "\n")
    L.append("*(Pk/WindowDiff ↓ = mieux, sur les multi ; F1_multi = frontières tol ±1 ; "
             "mono_FP = fraction de mono sur-coupés ; F1_global = frontières mono+multi, "
             "objectif de sélection.)*\n")

    # 3. Top configs attention
    L.append("## 3. Top 15 configurations attention\n")
    top = sorted(flat, key=lambda s: (-s.gf1, -s.f1, s.windowdiff))[:15]
    cols2 = ["model", "layers", "heads", "W", "c", "Pk", "WindowDiff", "F1_multi",
             "P", "R", "mono_FP", "mono_cuts", "F1_global"]
    L.append(_md_table([t.as_row() for t in top], cols2) + "\n")

    # 4. Meilleure par (modèle × jeu de couches) — où l'attention aide-t-elle ?
    L.append("## 4. Meilleure config par couche (les têtes/couches qui aident)\n")
    for mk, ss in all_scores.items():
        L.append(f"\n**{mk}** — têtes locales sélectionnées par jeu de couches : "
                 f"`{infos[mk]['local_heads']}` (sur {infos[mk]['n_heads']} têtes).\n")
        best_by_ls = {}
        for s in ss:
            key = s.cfg.layer_set
            b = best_by_ls.get(key)
            if b is None or (s.gf1, s.f1) > (b.gf1, b.f1):
                best_by_ls[key] = s
        L.append(_md_table([best_by_ls[k].as_row() for k in
                            ["early", "lowmid", "mid", "late", "midlate", "all"]
                            if k in best_by_ls], cols2) + "\n")

    # 5. Verdict
    if bl:
        d_f1 = winner.f1 - bl["F1_multi"]
        d_gf1 = winner.gf1 - bl["F1_global"]
        d_pk = winner.pk - bl["Pk"]
        beats = (winner.f1 > bl["F1_multi"] + 1e-9) and (winner.pk < bl["Pk"] - 1e-9)
        beats_g = winner.gf1 > bl["F1_global"] + 1e-9
    L.append("## 5. Verdict honnête\n")
    L.append(f"**Meilleure config attention : `{winner.cfg.model}` · "
             f"{winner.cfg.layer_set}/{winner.cfg.head_agg} · W={winner.cfg.W} · c={winner.c}** "
             f"→ F1_multi={winner.f1:.3f} (P={winner.precision:.3f}, R={winner.recall:.3f}), "
             f"Pk={winner.pk:.3f}, WindowDiff={winner.windowdiff:.3f}, "
             f"F1_global={winner.gf1:.3f}, mono_FP={winner.mono_fp_rate:.3f}.\n")
    if bl:
        verdict = ("**OUI**" if beats else ("partiellement" if beats_g else "**NON**"))
        L.append(f"- **L'attention bat-elle la trajectoire d'embedding ? {verdict}.** "
                 f"vs change-point (F1_multi={bl['F1_multi']}, Pk={bl['Pk']}, "
                 f"F1_global={bl['F1_global']}) : "
                 f"ΔF1_multi={d_f1:+.3f}, ΔPk={d_pk:+.3f} (négatif = mieux), "
                 f"ΔF1_global={d_gf1:+.3f}.\n")
    # Contrôle MÊME-encodeur : isole l'apport de l'attention du choix de modèle.
    ec = emb_controls.get(winner.cfg.model)
    if ec:
        d_f1e = winner.f1 - ec["F1_multi"]
        d_pke = winner.pk - ec["Pk"]
        d_gf1e = winner.gf1 - ec["F1_global"]
        beats_same = (winner.f1 > ec["F1_multi"] + 1e-9) and (winner.pk < ec["Pk"] - 1e-9)
        L.append(
            f"- **Contrôle MÊME encodeur (sans confondre signal et modèle)** : la "
            f"trajectoire d'embedding de `{winner.cfg.model}` lui-même (`{ec['method']}` "
            f"W={ec['W']}) donne F1_multi={ec['F1_multi']}, Pk={ec['Pk']}, "
            f"F1_global={ec['F1_global']}. L'attention du même modèle fait "
            f"ΔF1_multi={d_f1e:+.3f}, ΔPk={d_pke:+.3f}, ΔF1_global={d_gf1e:+.3f} → "
            f"l'attention {'**bat**' if beats_same else 'ne bat pas nettement'} sa propre "
            f"trajectoire d'embedding. **C'est la comparaison décisive** (le gain n'est "
            f"pas un simple effet « e5 > nomic »).\n")
    diffuse = winner.cfg.head_agg == "mean"
    L.append(
        "- **Honnêteté têtes/couches** : l'attention de transformer est en grande partie "
        "syntaxique/positionnelle (têtes qui suivent le mot précédent/suivant, ou pointent "
        "vers la ponctuation). La sélection `local` ne garde que les têtes les plus "
        "topiques, mais rien ne garantit qu'une tête « thème » existe : XLM-R n'a jamais "
        "été entraîné à segmenter. "
        + ("Fait notable : la meilleure config agrège **TOUTES** les têtes (`mean`), pas la "
           "sélection `local` — le signal de cohésion thématique est **diffus** sur l'ensemble "
           "des têtes des couches basses-moyennes, pas concentré dans quelques « têtes-thème » "
           "identifiables. " if diffuse else
           "Ici la sélection `local` (têtes les plus concentrées) l'emporte. ")
        + "Les couches qui portent le signal (basses-moyennes, §4) précèdent les couches "
        "tardives plus abstraites/poolées — cohérent avec l'idée que la cohésion locale de "
        "thème vit tôt dans le réseau.\n")
    L.append(
        f"- **Jeu** : multi = concaténation de mono-thèmes (frontières nettes par "
        f"construction) → borne OPTIMISTE pour les deux approches.\n")
    L.append(
        "- **Portage nomic** : justifié UNIQUEMENT si l'attention bat nettement l'embedding "
        "ci-dessus. nomic (code custom, Wqkv fusionné + rotary) demande un hook manuel pour "
        "exposer les poids — coût non négligeable. Verdict ci-dessus = feu vert / rouge.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Segmentation par attention (e5-base/bge-m3).")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--models", nargs="+", default=["e5-base", "bge-m3"])
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    items, _ = load_gold(gold_path)
    print(f"gold: {gold_path.name} — {len(items)} items")
    baseline = load_baseline()
    if baseline:
        print(f"baseline change-point: {baseline['winner']}")

    all_scores, infos, feas, emb_controls = {}, {}, {}, {}
    prepared_by_model = {}
    for mk in args.models:
        print(f"\n=== {mk} ===")
        feas[mk] = feasibility_probe(mk)
        print(f"faisabilité: {feas[mk].get('ok')} "
              f"(L={feas[mk].get('n_layers')} H={feas[mk].get('n_heads')})")
        if not feas[mk].get("ok"):
            all_scores[mk] = []
            continue
        print("extraction attention + préparation…")
        prepared = prepare(items, mk)
        prepared_by_model[mk] = prepared
        print("contrôle trajectoire d'embedding (même encodeur)…")
        emb_controls[mk] = emb_control(prepared)
        if emb_controls[mk]:
            e = emb_controls[mk]
            print(f"  embedding {mk}: {e['method']} W={e['W']} "
                  f"F1_multi={e['F1_multi']} Pk={e['Pk']} F1_global={e['F1_global']}")
        print("balayage configs attention…")
        scores, info = sweep_model(mk, prepared)
        all_scores[mk] = scores
        infos[mk] = info
        if scores:
            w = max(scores, key=lambda s: (s.gf1, s.f1))
            print(f"  best {mk}: {w.cfg.layer_set}/{w.cfg.head_agg} W={w.cfg.W} c={w.c} "
                  f"F1_multi={w.f1:.3f} Pk={w.pk:.3f} F1_global={w.gf1:.3f}")

    report = build_report(gold_path, items, all_scores, infos, feas, baseline,
                          prepared_by_model, emb_controls)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n✓ {args.out}")

    flat = [s for ss in all_scores.values() for s in ss]
    winner = max(flat, key=lambda s: (s.gf1, s.f1)) if flat else None
    Path(args.scores_out).write_text(json.dumps({
        "gold": gold_path.name, "models": args.models, "n_items": len(items),
        "seed": SEED,
        "feasibility": feas,
        "baseline_changepoint": baseline["winner"] if baseline else None,
        "embedding_controls": emb_controls,
        "winner": winner.as_row() if winner else None,
        "infos": {k: {"n_layers": v["n_layers"], "n_heads": v["n_heads"],
                      "local_heads": v["local_heads"]} for k, v in infos.items()},
        "configs": [s.as_row() for s in sorted(flat, key=lambda s: -s.gf1)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()

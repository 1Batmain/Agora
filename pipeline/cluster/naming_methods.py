"""Nommage SWITCHABLE des thèmes — c-TF-IDF (défaut) | centroïde | LLM local.

Orthogonal au clustering (`leiden`/`hdbscan`) et au dataset : la MÉTHODE de
clustering décide *quels* groupes existent ; la méthode de NOMMAGE décide
*comment on les titre*. Trois méthodes, toutes langue-agnostiques et souveraines
(aucune API externe) :

1. **`ctfidf`** (défaut) — c-TF-IDF + mots-vides corpus-dérivés (`naming.py`).
   Inchangé, déterministe, reproductible. Le complément des deux autres : ses
   keywords servent d'entrée au LLM et de repli.

2. **`centroid`** — label = le **témoignage le plus représentatif** : le membre
   (médoïde) dont l'embedding est le plus proche du centroïde du cluster (cosinus
   max). Un verbatim citoyen réel comme en-tête, tronqué proprement. Cheap,
   déterministe, zéro dépendance.

3. **`llm`** — titre court généré par un **LLM LOCAL via Ollama** (défaut
   `llama3.2:3b`, multilingue, surchargeable par `AGORA_OLLAMA_MODEL`). Prompt
   **langue-agnostique** : « donne un titre court dans LEUR langue ». Entrée =
   top-membres (proches du centroïde) + keywords c-TF-IDF. **Repli gracieux sur
   ctfidf** si Ollama est injoignable / en timeout — `naming` reflète alors la
   méthode RÉELLEMENT appliquée.

Interface unique : `name_clusters_method(cluster_docs, method=..., members=...,
vecs=..., ideas=..., corpus_stopwords=...) -> (names, naming_meta)`. `names` a la
même shape que `name_clusters` (`{cid: {label, keywords}}`) ; `naming_meta` trace
la méthode réelle, le modèle, la latence et un éventuel repli.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np

from pipeline.cluster.naming import name_clusters

NAMING_METHODS = ("ctfidf", "centroid", "llm")
DEFAULT_NAMING = "ctfidf"

# --- Ollama local (souverain, configurable par env — zéro valeur de corpus) ----
OLLAMA_URL = os.environ.get("AGORA_OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Défaut = modèle léger NON-raisonneur, multilingue, qui répond DIRECTEMENT un
# titre (~3 s). NB : `qwen3:4b` (suggéré au contrat) est un modèle *à raisonnement*
# que l'Ollama installé (0.30) ne sait pas désactiver (`think:false` / `/no_think`
# non honorés) → il consomme tout le budget en « pensée » et ne sort pas de titre
# exploitable. On garde donc un défaut qui MARCHE, surchargeable par env pour tout
# autre modèle local. Cf. NAMING_SWITCH_NOTE.md.
OLLAMA_MODEL = os.environ.get("AGORA_OLLAMA_MODEL", "llama3.2:3b")
# Timeout COURT par appel (Ollama partagé) ; sonde de dispo encore plus courte.
OLLAMA_TIMEOUT = float(os.environ.get("AGORA_OLLAMA_TIMEOUT", "25"))
OLLAMA_PROBE_TIMEOUT = float(os.environ.get("AGORA_OLLAMA_PROBE_TIMEOUT", "2.5"))
# Parcimonie : quelques appels en parallèle au plus (Ollama est partagé).
LLM_MAX_WORKERS = int(os.environ.get("AGORA_LLM_MAX_WORKERS", "4"))

# Nb de top-membres injectés au LLM ; longueur max d'un en-tête verbatim/LLM.
TOP_MEMBERS = 6
MEMBER_SNIPPET = 240
LABEL_MAXLEN = 80
LLM_MAX_WORDS = 8  # garde-fou : "titre court (≤6 mots)" + marge


# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------
def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _truncate(text: str, maxlen: int = LABEL_MAXLEN) -> str:
    """Tronque proprement sur une frontière de mot, suffixe « … »."""
    text = _normalize_ws(text)
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen].rsplit(" ", 1)[0] or text[:maxlen]
    return cut.rstrip(" ,;:.!?—-") + "…"


def _members_by_proximity(member_idxs: list[int], vecs: np.ndarray) -> list[int]:
    """Membres triés par proximité au centroïde (cosinus décroissant).

    Vecteurs supposés L2-normalisés (sortie de l'embedder) ; le plus proche du
    centroïde = le plus représentatif (médoïde en tête)."""
    sub = vecs[member_idxs]
    centroid = sub.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    sims = sub @ centroid
    order = np.argsort(sims)[::-1]
    return [member_idxs[int(i)] for i in order]


# ---------------------------------------------------------------------------
# 2) Centroïde — verbatim représentatif
# ---------------------------------------------------------------------------
def _name_centroid(
    cids: list[int], members: dict[int, list[int]], vecs: np.ndarray, ideas: list,
    base: dict[int, dict],
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for cid in cids:
        idxs = members.get(cid, [])
        if not idxs:
            out[cid] = base.get(cid, {"label": f"thème {cid}", "keywords": []})
            continue
        medoid = _members_by_proximity(idxs, vecs)[0]
        text = ideas[medoid].text_clean or ideas[medoid].text
        label = _truncate(text) or base.get(cid, {}).get("label", f"thème {cid}")
        out[cid] = {"label": label, "keywords": base.get(cid, {}).get("keywords", [])}
    return out


# ---------------------------------------------------------------------------
# 3) LLM local (Ollama) — titre court, langue-agnostique, repli ctfidf
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _ollama_available(url: str = OLLAMA_URL, timeout: float = OLLAMA_PROBE_TIMEOUT) -> bool:
    """Sonde rapide : Ollama répond-il ? Évite N timeouts si le service est down."""
    try:
        import httpx

        r = httpx.get(f"{url}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _build_prompt(samples: list[str], keywords: list[str]) -> str:
    """Prompt LANGUE-AGNOSTIQUE : titre court dans la langue des contributions."""
    body = "\n".join(f"- {s}" for s in samples)
    kw = ", ".join(keywords[:6])
    return (
        "Voici des contributions citoyennes regroupées par thème commun.\n"
        "Donne UNIQUEMENT un titre court (≤ 6 mots) qui résume ce thème, "
        "rédigé dans LA MÊME LANGUE que les contributions ci-dessous. "
        "Pas de guillemets, pas de ponctuation finale, pas d'explication, "
        "pas de préfixe — seulement le titre.\n\n"
        f"Contributions :\n{body}\n"
        + (f"\nMots-clés distinctifs : {kw}\n" if kw else "")
        + "\nTitre :"
    )


def _clean_title(raw: str) -> str:
    """Nettoie la sortie LLM : retire le raisonnement, garde une ligne de titre."""
    raw = _THINK_RE.sub("", raw or "")
    for line in raw.splitlines():
        line = _normalize_ws(line)
        # retire guillemets/puces/préfixes éventuels
        line = line.strip("\"'«»`*•-–—:. ")
        if line.lower().startswith(("titre", "title", "titel")):
            line = line.split(":", 1)[-1].strip("\"'«»`*•-–—:. ")
        if not line:
            continue
        words = line.split()
        if len(words) > LLM_MAX_WORDS:
            line = " ".join(words[:LLM_MAX_WORDS])
        return _truncate(line)
    return ""


def _ollama_title(prompt: str, model: str, url: str, timeout: float) -> str:
    """Un appel Ollama `/api/generate` (non-stream, thinking désactivé)."""
    import httpx

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # qwen3 & co. : pas de bloc <think>, réponse directe
        "options": {"temperature": 0.2, "num_predict": 40, "top_p": 0.9},
    }
    r = httpx.post(f"{url}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return _clean_title(r.json().get("response", ""))


def _name_llm(
    cids: list[int], members: dict[int, list[int]], vecs: np.ndarray, ideas: list,
    base: dict[int, dict], *, model: str, url: str, timeout: float,
) -> tuple[dict[int, dict], dict]:
    """Titre LLM par cluster, parallélisé et parcimonieux ; repli ctfidf par item.

    Si Ollama est injoignable (sonde), repli GLOBAL immédiat sur ctfidf
    (`naming="ctfidf"`). Sinon chaque échec isolé retombe sur le label ctfidf de
    SON cluster et est compté dans `n_fallback`.
    """
    out = {cid: dict(base.get(cid, {"label": f"thème {cid}", "keywords": []})) for cid in cids}

    if not _ollama_available(url):
        return out, {
            "naming": "ctfidf", "requested": "llm", "fallback": True,
            "reason": "ollama_unreachable", "model": model, "url": url,
            "n_llm": 0, "n_fallback": len(cids),
        }

    def title_for(cid: int) -> tuple[int, str | None]:
        idxs = members.get(cid, [])
        if not idxs:
            return cid, None
        top = _members_by_proximity(idxs, vecs)[:TOP_MEMBERS]
        samples = [_truncate(ideas[i].text_clean or ideas[i].text, MEMBER_SNIPPET) for i in top]
        kw = base.get(cid, {}).get("keywords", [])
        try:
            return cid, _ollama_title(_build_prompt(samples, kw), model, url, timeout)
        except Exception:
            return cid, None

    n_llm = 0
    workers = max(1, min(LLM_MAX_WORKERS, len(cids)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cid, title in pool.map(title_for, cids):
            if title:
                out[cid]["label"] = title
                n_llm += 1

    n_fb = len(cids) - n_llm
    return out, {
        "naming": "llm" if n_llm else "ctfidf",
        "requested": "llm",
        "fallback": n_fb > 0,
        "reason": None if n_fb == 0 else "partial_fallback",
        "model": model, "url": url,
        "n_llm": n_llm, "n_fallback": n_fb,
    }


# ---------------------------------------------------------------------------
# Dispatcher public
# ---------------------------------------------------------------------------
def name_clusters_method(
    cluster_docs: dict[int, list[str]],
    *,
    method: str = DEFAULT_NAMING,
    members: dict[int, list[int]] | None = None,
    vecs: np.ndarray | None = None,
    ideas: list | None = None,
    corpus_stopwords: set[str] | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> tuple[dict[int, dict], dict]:
    """Nomme des clusters selon `method`. Renvoie `(names, naming_meta)`.

    `cluster_docs` = `{cid: [avis...]}` (comme `name_clusters`). `centroid`/`llm`
    requièrent `members` (`{cid: [idx...]}`), `vecs` (embeddings alignés) et
    `ideas` (pour le verbatim). c-TF-IDF est TOUJOURS calculé (keywords +
    repli + entrée LLM). Une méthode inconnue retombe sur ctfidf.
    """
    method = (method or DEFAULT_NAMING).lower()
    t0 = perf_counter()

    base = name_clusters(cluster_docs, corpus_stopwords=corpus_stopwords)
    cids = sorted(cluster_docs.keys())

    def stamp(meta: dict) -> dict:
        meta["took_ms"] = round((perf_counter() - t0) * 1000)
        return meta

    if method == "ctfidf" or not cids:
        return base, stamp({"naming": "ctfidf", "requested": method, "fallback": False})

    can_member = members is not None and vecs is not None and ideas is not None
    if method in ("centroid", "llm") and not can_member:
        # Sécurité : sans le contexte d'embeddings, on ne peut pas faire mieux.
        return base, stamp({
            "naming": "ctfidf", "requested": method, "fallback": True,
            "reason": "missing_embeddings_context",
        })

    if method == "centroid":
        out = _name_centroid(cids, members, vecs, ideas, base)
        return out, stamp({"naming": "centroid", "requested": "centroid", "fallback": False})

    if method == "llm":
        out, meta = _name_llm(
            cids, members, vecs, ideas, base,
            model=model or OLLAMA_MODEL,
            url=(ollama_url or OLLAMA_URL).rstrip("/"),
            timeout=timeout or OLLAMA_TIMEOUT,
        )
        return out, stamp(meta)

    return base, stamp({"naming": "ctfidf", "requested": method, "fallback": True,
                        "reason": "unknown_method"})

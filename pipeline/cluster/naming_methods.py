"""Nommage SWITCHABLE des thèmes — c-TF-IDF (défaut) | centroïde | LLM (Mistral).

Orthogonal au clustering (`leiden`/`hdbscan`) et au dataset : la MÉTHODE de
clustering décide *quels* groupes existent ; la méthode de NOMMAGE décide
*comment on les titre*. Trois méthodes, toutes langue-agnostiques :

1. **`ctfidf`** (défaut) — c-TF-IDF + mots-vides corpus-dérivés (`naming.py`).
   Inchangé, déterministe, reproductible, souverain. Le complément des deux
   autres : ses keywords servent d'entrée au LLM et de **repli**.

2. **`centroid`** — label = le **témoignage le plus représentatif** : le membre
   (médoïde) dont l'embedding est le plus proche du centroïde du cluster (cosinus
   max). Un verbatim citoyen réel comme en-tête, tronqué proprement. Cheap,
   déterministe, zéro dépendance, souverain.

3. **`llm`** — titres courts générés par l'**API Mistral** (EU, `mistral-small-
   latest` par défaut, surchargeable par `AGORA_MISTRAL_MODEL`). **Batché** : UN
   seul appel renvoie un JSON `{cluster_id: titre}` pour TOUS les clusters.
   Entrée par cluster = keywords c-TF-IDF + quelques témoignages représentatifs
   (médoïdes). Prompt **langue-agnostique** : titre dans la langue dominante du
   cluster. **Repli gracieux sur ctfidf** si pas de clé / erreur / timeout —
   `naming` reflète alors la méthode RÉELLEMENT appliquée. (L'ancien backend
   Ollama local est abandonné : le serveur de déploiement ne peut pas l'exécuter.)

Interface unique : `name_clusters_method(cluster_docs, method=..., members=...,
vecs=..., ideas=..., corpus_stopwords=...) -> (names, naming_meta)`. `names` a la
même shape que `name_clusters` (`{cid: {label, keywords}}`) ; `naming_meta` trace
la méthode réelle, le modèle, la latence et un éventuel repli.
"""

from __future__ import annotations

import json
import os
import re
from time import perf_counter

import numpy as np

from pipeline.cluster import mistral_client
from pipeline.cluster.mistral_client import NAMING_MODEL as MISTRAL_MODEL
from pipeline.cluster.naming import name_clusters

NAMING_METHODS = ("ctfidf", "centroid", "llm")
DEFAULT_NAMING = "ctfidf"

# Timeout de l'appel Mistral batché (un seul appel pour tous les clusters).
LLM_TIMEOUT = float(os.environ.get("AGORA_MISTRAL_TIMEOUT", "60"))

# Nb de témoignages représentatifs (médoïdes) injectés par cluster ; longueur
# max d'un snippet et d'un en-tête verbatim/LLM.
TOP_MEMBERS = 3
MEMBER_SNIPPET = 200
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
# 3) LLM (API Mistral) — titres courts BATCHÉS, langue-agnostiques, repli ctfidf
# ---------------------------------------------------------------------------
def _cluster_block(cid: int, samples: list[str], keywords: list[str]) -> str:
    """Bloc d'UN cluster dans le prompt batché : id + mots-clés + témoignages."""
    body = "\n".join(f"  - {s}" for s in samples)
    kw = ", ".join(keywords[:6])
    head = f"Cluster {cid}"
    if kw:
        head += f" (mots-clés : {kw})"
    return f"{head}\n{body}" if body else head


def _build_batch_prompt(blocks: list[str]) -> str:
    """Prompt LANGUE-AGNOSTIQUE batché → JSON {cluster_id: titre court}.

    Un SEUL appel pour tous les clusters. Demande explicitement un objet JSON
    dont les clés sont les identifiants de cluster (en chaîne) et les valeurs des
    titres courts, chacun dans la langue dominante de SON cluster.
    """
    joined = "\n\n".join(blocks)
    return (
        "Voici des groupes (« clusters ») de contributions citoyennes, chacun "
        "identifié par un numéro. Pour CHAQUE cluster, propose un titre court "
        "(≤ 6 mots) qui en résume le thème, rédigé dans LA MÊME LANGUE que les "
        "contributions de ce cluster.\n\n"
        "Réponds UNIQUEMENT par un objet JSON dont les clés sont les numéros de "
        "cluster (en chaîne de caractères) et les valeurs les titres. "
        "Pas de texte autour, pas de guillemets dans les titres, pas de "
        "ponctuation finale, pas d'explication.\n"
        'Exemple de format : {"0": "titre du thème", "3": "autre titre"}\n\n'
        f"Clusters :\n{joined}\n"
    )


def _clean_title(raw: str) -> str:
    """Nettoie un titre LLM : une ligne, sans puces/guillemets/préfixe, bornée."""
    for line in (raw or "").splitlines():
        line = _normalize_ws(line)
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


def _parse_titles(content: str) -> dict[str, str]:
    """Parse le JSON `{cluster_id: titre}` renvoyé par Mistral (tolérant)."""
    if not content:
        return {}
    text = content.strip()
    # Retire un éventuel bloc de code markdown ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Repli : isole le premier objet {...} du texte.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _name_llm(
    cids: list[int], members: dict[int, list[int]], vecs: np.ndarray, ideas: list,
    base: dict[int, dict], *, model: str, timeout: float,
) -> tuple[dict[int, dict], dict]:
    """Titres LLM BATCHÉS via Mistral ; repli ctfidf (global ou par item).

    Un SEUL appel Mistral renvoie un JSON {cid: titre}. Si pas de clé / erreur /
    timeout → repli GLOBAL sur ctfidf (`naming="ctfidf"`, `reason` détaillé).
    Si l'appel réussit mais qu'un cluster manque/est vide → ce cluster retombe
    sur son label ctfidf et est compté dans `n_fallback`.
    """
    out = {cid: dict(base.get(cid, {"label": f"thème {cid}", "keywords": []})) for cid in cids}

    if not mistral_client.available():
        return out, {
            "naming": "ctfidf", "requested": "llm", "fallback": True,
            "reason": "no_api_key", "model": model,
            "n_llm": 0, "n_fallback": len(cids),
        }

    # Construit UN bloc par cluster (keywords + médoïdes représentatifs).
    blocks: list[str] = []
    for cid in cids:
        idxs = members.get(cid, [])
        samples: list[str] = []
        if idxs:
            top = _members_by_proximity(idxs, vecs)[:TOP_MEMBERS]
            samples = [_truncate(ideas[i].text_clean or ideas[i].text, MEMBER_SNIPPET) for i in top]
        kw = base.get(cid, {}).get("keywords", [])
        blocks.append(_cluster_block(cid, samples, kw))

    messages = [{"role": "user", "content": _build_batch_prompt(blocks)}]
    try:
        content = mistral_client.chat(
            messages, model=model, temperature=0.2,
            max_tokens=min(2048, 64 * max(1, len(cids))), json_mode=True,
            timeout=timeout,
        )
    except mistral_client.MistralError as exc:
        return out, {
            "naming": "ctfidf", "requested": "llm", "fallback": True,
            "reason": f"api_error:{exc.status}:{exc.reason}", "model": model,
            "n_llm": 0, "n_fallback": len(cids),
        }

    titles = _parse_titles(content)
    n_llm = 0
    for cid in cids:
        raw = titles.get(str(cid))
        title = _clean_title(raw) if raw else ""
        if title:
            out[cid]["label"] = title
            n_llm += 1

    n_fb = len(cids) - n_llm
    return out, {
        "naming": "llm" if n_llm else "ctfidf",
        "requested": "llm",
        "fallback": n_fb > 0,
        "reason": None if n_fb == 0 else ("empty_response" if n_llm == 0 else "partial_fallback"),
        "model": model,
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
            model=model or MISTRAL_MODEL,
            timeout=timeout or LLM_TIMEOUT,
        )
        return out, stamp(meta)

    return base, stamp({"naming": "ctfidf", "requested": method, "fallback": True,
                        "reason": "unknown_method"})

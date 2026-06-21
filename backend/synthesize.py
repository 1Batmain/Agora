"""Synthèse LLM d'une consultation — rapport Markdown court via l'API Mistral.

`POST /synthesize` (cf. `server.py`) reclusterise le dataset (réutilise
`recluster`), construit un **résumé compact de TOUS les clusters** (label,
keywords, taille/poids, témoignages représentatifs = médoïdes) puis fait **UN
appel Mistral** qui rédige un rapport court en deux parties :

  (a) **synthèse** des grands thèmes / de la parole citoyenne ;
  (b) **feedback sur la PERTINENCE** des clusters (cohérence, redondance,
      couverture, qualité du découpage, clusters douteux).

Langue-agnostique : le rapport est rédigé dans la **langue dominante** du corpus.
Générique : aucune valeur de corpus en dur. Repli gracieux : sans clé Mistral,
renvoie un message clair (`fallback=True`) — jamais un crash.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from backend.recluster import recluster
from pipeline.cluster import mistral_client

# Garde-fous de taille de prompt (généric : bornes structurelles, pas de corpus).
MAX_CLUSTERS_IN_PROMPT = 40   # au-delà, on garde les mieux classés (rank)
REP_PER_CLUSTER = 2           # témoignages représentatifs par cluster
REP_SNIPPET = 220             # longueur max d'un témoignage cité
SYNTH_MAX_TOKENS = 1400       # rapport COURT


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _truncate(text: str, maxlen: int = REP_SNIPPET) -> str:
    text = _normalize_ws(text)
    if len(text) <= maxlen:
        return text
    cut = text[:maxlen].rsplit(" ", 1)[0] or text[:maxlen]
    return cut.rstrip(" ,;:.!?—-") + "…"


def _dominant_lang(languages: list[str] | None, nodes: list[dict]) -> str | None:
    """Langue dominante : descripteur dataset sinon dérivée des nœuds."""
    if languages:
        return languages[0]
    from collections import Counter

    c = Counter(n.get("props", {}).get("lang") for n in nodes if n.get("props", {}).get("lang"))
    return c.most_common(1)[0][0] if c else None


def _representatives(theme: dict, id2vec: dict, id2text: dict, k: int) -> list[str]:
    """Témoignages les plus proches du centroïde du thème (médoïdes), tronqués."""
    centroid = theme.get("centroid")
    members = theme.get("member_ids", [])
    if not members:
        return []
    if not centroid:
        return [_truncate(id2text.get(m, "")) for m in members[:k] if id2text.get(m)]
    c = np.asarray(centroid, dtype=np.float32)
    scored: list[tuple[float, str]] = []
    for mid in members:
        v = id2vec.get(mid)
        if v is None:
            continue
        scored.append((float(v @ c), mid))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    out: list[str] = []
    for _, mid in scored[:k]:
        t = _truncate(id2text.get(mid, ""))
        if t:
            out.append(t)
    return out


def _build_summary(payload: dict, id2vec: dict, lang: str | None) -> tuple[str, int, bool]:
    """Résumé compact texte de TOUS les macro-thèmes. Renvoie (texte, n, tronqué?)."""
    nodes = payload.get("nodes", [])
    id2text = {
        n["id"]: (n.get("props", {}).get("text_clean") or n.get("props", {}).get("text") or n.get("label", ""))
        for n in nodes
    }
    themes = payload.get("themes", [])
    # Macro-thèmes réels (exclut le bruit HDBSCAN cluster_id=-1), dans l'ordre de rang.
    macros = [t for t in themes if t.get("level", 0) == 0 and t.get("cluster_id", 0) >= 0]
    truncated = len(macros) > MAX_CLUSTERS_IN_PROMPT
    macros = macros[:MAX_CLUSTERS_IN_PROMPT]

    meta = payload.get("meta", {})
    stats = meta.get("stats", {})
    lines = [
        f"Consultation : {meta.get('dataset', '?')}",
        f"Contributions analysées : {meta.get('n_nodes', len(nodes))}",
        f"Méthode de clustering : {meta.get('method', '?')} · nommage : {meta.get('naming', '?')}",
        f"Nombre de grands thèmes : {len(macros)}"
        + (f" (sur {stats.get('n_macros', len(macros))}, tronqué)" if truncated else ""),
    ]
    if lang:
        lines.append(f"Langue dominante : {lang}")
    lines.append("")
    lines.append("THÈMES :")

    for t in macros:
        kw = ", ".join((t.get("keywords") or [])[:6])
        n_sub = len(t.get("children") or [])
        head = (
            f"\n[{t.get('cluster_id')}] {t.get('label', '')} "
            f"— taille {t.get('size')}, poids {t.get('weight_sum')}"
        )
        if n_sub:
            head += f", {n_sub} sous-thèmes"
        consensus = t.get("consensus")
        diversity = t.get("diversity")
        if consensus is not None and diversity is not None:
            head += f", cohérence {consensus}, diversité {diversity}"
        lines.append(head)
        if kw:
            lines.append(f"  mots-clés : {kw}")
        reps = _representatives(t, id2vec, id2text, REP_PER_CLUSTER)
        for r in reps:
            lines.append(f"  • {r}")

    return "\n".join(lines), len(macros), truncated


def _build_messages(summary: str, lang: str | None) -> list[dict]:
    """Messages Mistral : consigne langue-agnostique + résumé des clusters."""
    lang_clause = (
        f"Rédige le rapport dans la langue dominante du corpus (code : {lang}). "
        if lang else
        "Rédige le rapport dans la langue dominante des contributions ci-dessous. "
    )
    system = (
        "Tu es analyste de consultations citoyennes. Tu produis des synthèses "
        "neutres, factuelles et concises à partir de thèmes déjà regroupés "
        "automatiquement. Tu ne fais aucune supposition hors des données fournies."
    )
    user = (
        "À partir du résumé des thèmes ci-dessous (issus d'un regroupement "
        "automatique de contributions citoyennes), rédige un RAPPORT MARKDOWN "
        "COURT en deux parties, avec ces deux titres exacts de niveau 2 :\n\n"
        "## Synthèse\n"
        "Les grands thèmes qui ressortent de la parole citoyenne, leur importance "
        "relative, les points de convergence et de tension. Quelques phrases par "
        "grand thème, pas une liste exhaustive.\n\n"
        "## Pertinence du découpage\n"
        "Un regard critique sur la QUALITÉ du regroupement : les thèmes sont-ils "
        "cohérents ? Y a-t-il des redondances (thèmes qui se recoupent) ? La "
        "couverture est-elle bonne ? Des clusters paraissent-ils douteux, trop "
        "génériques ou mal découpés ? Sois bref et concret.\n\n"
        + lang_clause
        + "Reste COURT (l'ensemble doit tenir en une page). N'invente aucun chiffre "
        "absent du résumé.\n\n"
        f"Résumé des thèmes :\n\n{summary}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip_code_fence(text: str) -> str:
    """Retire un fence Markdown enveloppant (``` ou ```markdown) si le LLM en met un,
    sinon le rapport s'afficherait comme un bloc de code brut côté front."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def synthesize(
    ideas: list,
    vecs: np.ndarray,
    weights: np.ndarray,
    *,
    dataset: str,
    method: str,
    naming: str,
    languages: list[str] | None = None,
) -> dict:
    """Construit le rapport de synthèse. Renvoie
    `{report_markdown, meta:{model, took_ms, n_clusters, fallback?, reason?}}`.

    Repli gracieux : sans clé Mistral, `fallback=True` + message clair (pas de
    crash). `ideas`/`vecs`/`weights` = superset CACHÉ aligné (comme recluster).
    """
    t0 = perf_counter()
    model = mistral_client.SYNTHESIS_MODEL

    # Reclusterise (réutilise tout le pipeline) pour disposer des thèmes à jour.
    payload = recluster(ideas, vecs, weights, method=method, naming=naming, dataset=dataset)

    # Map id → vecteur (cache complet aligné) pour sélectionner les médoïdes.
    id2vec = {idea.id: vecs[i] for i, idea in enumerate(ideas)}
    lang = _dominant_lang(languages, payload.get("nodes", []))
    summary, n_clusters, truncated = _build_summary(payload, id2vec, lang)

    def _stamp(extra: dict) -> dict:
        return {"took_ms": round((perf_counter() - t0) * 1000),
                "n_clusters": n_clusters, "truncated": truncated, **extra}

    if not mistral_client.available():
        return {
            "report_markdown": (
                "_Synthèse indisponible : clé Mistral manquante "
                "(`MISTRAL_API_KEY` non configurée)._"
            ),
            "meta": _stamp({"model": model, "fallback": True, "reason": "no_api_key"}),
        }

    messages = _build_messages(summary, lang)
    try:
        content = mistral_client.chat(
            messages, model=model, temperature=0.3, max_tokens=SYNTH_MAX_TOKENS,
        )
    except mistral_client.MistralError as exc:
        return {
            "report_markdown": (
                f"_Synthèse indisponible : l'appel à Mistral a échoué "
                f"(statut {exc.status})._"
            ),
            "meta": _stamp({"model": model, "fallback": True,
                            "reason": f"api_error:{exc.status}:{exc.reason}"}),
        }

    return {
        "report_markdown": _strip_code_fence(content),
        "meta": _stamp({"model": model, "fallback": False, "lang": lang}),
    }

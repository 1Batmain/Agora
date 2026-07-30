"""Endpoint `/insights` — synthèses LLM Markdown LIÉES AU NIVEAU DE ZOOM (B3 du contrat).

Le panneau droit du front affiche un texte généré par LLM qui dépend du niveau courant :

  - `level="global"`        → synthèse de TOUTE la consultation (grands thèmes,
                              convergences/tensions, points saillants) ;
  - `level="theme", id=…`  → synthèse d'UN thème (sa parole, ses sous-thèmes, son poids).

Réutilise l'arbre variance-adaptatif (`backend.analysis`) pour le contenu et le client
`pipeline.cluster.mistral_client` (**API Mistral par défaut**) pour la rédaction.
**Caché par `(dataset, level, id)`** (mémoire + disque) → le 2ᵉ appel est instantané
(acceptance). Repli gracieux : sans clé Mistral, renvoie un message clair (`fallback=True`),
jamais un crash. Langue-agnostique : rédigé dans la langue dominante du corpus.

    GET /insights {dataset, level:global|theme, id?} -> {markdown, meta}
"""

from __future__ import annotations

import json
import re
from time import perf_counter

from backend.analysis import (
    ThemeNode,
    ThemeTree,
    _dataset_context,
)
from backend.recluster import dataset_dir
from pipeline.cluster import mistral_client

# UN SEUL cache d'insights : BAKÉ au build (`analysis_store.insights_path` →
# <dataset>/analysis/insights/<global|theme_id>.json, nom SÉMANTIQUE), servi tel quel par
# `/insights`. L'ancien étage LIVE (calcul à la demande, caché par hash) a été retiré : le
# serveur ne calcule plus d'insight, il lit le précalculé (`server.py` → `read_insights`).


# ─────────────────────────────────────────────────────────────────────────────
# HARNESS de structure — chaque synthèse a une STRUCTURE FIXE, et chaque section est
# générée par un appel LLM FOCALISÉ (ou dérivée). Segmenter la génération par section
# donne au LLM un contexte précis pour CETTE sous-tâche et garantit la cohérence (fin du
# « drift » de format). Les sections de NAVIGATION (sous-thématiques / thèmes) sont
# rendues par le FRONT et s'intercalent aux positions prévues — pas générées ici.
#
#   THÉMATIQUE : ## Vue générale (LLM) · [Thèmes distincts — front] · ## À relever (LLM)
#   GLOBALE    : ## Contexte (LLM) · ## Profil du panel (dérivé) · [Thèmes identifiés — front]
# ─────────────────────────────────────────────────────────────────────────────

_HARNESS_SYSTEM = (
    "Tu es analyste de consultations citoyennes pour des parlementaires. Tu écris de "
    "façon neutre, factuelle et concise, sans jamais rien inventer hors des données "
    "fournies. Tu réponds UNIQUEMENT le contenu demandé, sans titre de section ni "
    "préambule, dans la langue dominante des contributions."
)
_TASK_IDENTITE = (
    "Décris en 2 à 4 phrases ce qui fait l'IDENTITÉ de cette thématique : de quoi parlent "
    "les citoyens ici, ce qui les rassemble, la tonalité. UN seul paragraphe, sans liste, "
    "sans énumérer les sous-thématiques."
)
_TASK_TENSION = (
    "À partir des données d'opinion fournies (objet de clivage, profil, répartition "
    "favorable/défavorable), RELÈVE ce qui ressort au sein de cette thématique : indique "
    "si l'opinion est plutôt CONSENSUELLE, plutôt CLIVANTE, ou partagée, puis développe "
    "EN CONSÉQUENCE (accords larges et/ou lignes de fracture, points saillants). Adapte "
    "librement le fond ET la forme au signal réel — n'impose PAS de rubriques figées "
    "« Consensus »/« Tensions » si l'un domine largement. 2 à 4 phrases ou puces, "
    "factuel, sans recopier les chiffres bruts."
)
_TASK_CONTEXTE = (
    "En 2 à 3 phrases, présente le CONTEXTE de cette consultation : son OBJET (de quoi il "
    "s'agit, ce sur quoi les citoyens se sont exprimés) ET le but/cadre dans lequel elle a "
    "été émise (quand, par qui, pourquoi), d'après le contexte fourni."
)
_SECTION_MAX_TOKENS = 320


def _dataset_meta(dataset_id: str) -> dict:
    """meta.json du dataset (léger) — repli {} si absent/illisible."""
    try:
        return json.loads((dataset_dir(dataset_id) / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _section_of(md: str | None, heading: str) -> str | None:
    """Extrait le corps de la section « ## <heading> » d'un markdown (None si absente)."""
    if not md:
        return None
    m = re.search(rf"(?mi)^#{{1,4}}\s*{re.escape(heading)}\b[^\n]*$", md)
    if not m:
        return None
    start = m.end()
    nxt = re.search(r"(?m)^#{1,4}\s", md[start:])
    body = md[start: start + nxt.start()] if nxt else md[start:]
    return body.strip() or None


def _node_header(node: ThemeNode) -> list[str]:
    lines = [f"Thématique : {node.title or node.label}",
             f"Avis : {node.n_avis} · cohésion interne : {round(node.consensus or 0, 2)}"]
    if node.keywords:
        lines.append(f"Mots-clés : {', '.join(node.keywords[:8])}")
    return lines


def _identity_data(tree: ThemeTree, node: ThemeNode,
                   child_insights: dict[str, str] | None) -> str:
    """Entrée de « Vue générale » : bottom-up (vues générales des enfants) sinon claims."""
    lines = _node_header(node)
    kids = [(tree.nodes[c], _section_of((child_insights or {}).get(c), "Vue générale"))
            for c in node.children]
    if any(v for _, v in kids):
        lines += ["", "Vues générales des sous-thématiques (à AGRÉGER en une identité "
                  "commune, sans les recopier) :"]
        for c, v in kids:
            lines.append(f"— {c.title or c.label} ({c.n_avis} avis) : {v or '(—)'}")
    else:
        lines += ["", "Témoignages représentatifs :"]
        lines += [f"  • {rep}" for rep in node.representative_claims[:8]]
    return "\n".join(lines)


def _tension_data(tree: ThemeTree, node: ThemeNode, opinion: dict | None,
                  child_insights: dict[str, str] | None) -> str:
    """Entrée de « À relever » : données d'OPINION (clivage/consensus) si dispo, sinon claims."""
    lines = _node_header(node) + [""]
    if opinion:
        lines.append("Données d'opinion :")
        lines.append(f"  objet de clivage : « {opinion.get('proposition', '')} »")
        lines.append(f"  profil : {opinion.get('profil', '?')} — favorable {opinion.get('fav', 0)}"
                     f" · défavorable {opinion.get('def', 0)} · nuance {opinion.get('nuance', 0)}")
        if opinion.get("is_aggregate") and opinion.get("child_propositions"):
            lines.append("  objets de clivage des sous-thématiques :")
            lines += [f"    – {p}" for p in opinion["child_propositions"][:12]]
    else:
        lines.append(f"(Pas d'opinion bakée — infère depuis la cohésion "
                     f"[{round(node.consensus or 0, 2)}] et les témoignages.)")
        lines.append("Témoignages représentatifs :")
        lines += [f"  • {rep}" for rep in node.representative_claims[:8]]
    for c in [tree.nodes[x] for x in node.children]:
        sec = _section_of((child_insights or {}).get(c.id), "À relever")
        if sec:
            lines.append(f"\nÀ relever — {c.title or c.label} :\n{sec}")
    return "\n".join(lines)


def _global_data(tree: ThemeTree) -> str:
    """Entrée commune Introduction/Contexte : label + question + contexte + volumes."""
    meta = _dataset_meta(tree.dataset)
    prep = tree.prepared
    lines = [f"Consultation : {meta.get('label', tree.dataset)}"]
    if meta.get("question"):
        lines.append(f"Question posée : « {meta['question']} »")
    ctx = _dataset_context(tree.dataset)
    if ctx:
        lines.append(f"Contexte de collecte : {ctx}")
    lines.append(f"Témoignages analysés : {len(prep.avis)} · claims : {len(prep.claim_texts)}"
                 f" · grands thèmes : {len(tree.macros)}")
    return "\n".join(lines)


def _profil_panel(tree: ThemeTree) -> str:
    """Section DÉRIVÉE (pas de LLM) : profil du panel si définissable, sinon anonyme."""
    meta = _dataset_meta(tree.dataset)
    langs = [l.upper() for l in (meta.get("languages") or [])]
    s = ("Panel anonyme — participation volontaire, aucune donnée démographique collectée. "
         f"{len(tree.prepared.avis)} témoignages analysés")
    if langs:
        s += f" · langues : {', '.join(langs)}"
    return s + "."


def render_insight(tree: ThemeTree, level: str, theme_id: str | None = None,
                   *, model: str | None = None,
                   child_insights: dict[str, str] | None = None,
                   opinion: dict | None = None) -> dict:
    """Synthèse Markdown STRUCTURÉE d'un niveau (harness, sans cache).

    `child_insights` (BOTTOM-UP) : synthèses déjà générées des sous-thématiques (id →
    markdown) → « Vue générale »/« À relever » d'un PARENT agrègent celles des enfants.
    `opinion` : enregistrement d'opinion de CETTE thématique (opinion.json) → nourrit
    « À relever ». Absents → repli gracieux sur les claims."""
    return _render_insight(tree, level, theme_id, model=model,
                           child_insights=child_insights, opinion=opinion)


def _render_insight(tree: ThemeTree, level: str, theme_id: str | None = None,
                    *, model: str | None = None,
                    child_insights: dict[str, str] | None = None,
                    opinion: dict | None = None) -> dict:
    """Génère (sans cache) la synthèse STRUCTURÉE d'un niveau via le harness de sections.

    Chaque section LLM est un appel FOCALISÉ (contexte précis pour CETTE sous-tâche) ;
    les sections dérivées (Profil du panel) ne coûtent aucun appel. Repli gracieux
    (`meta.fallback`) sans clé Mistral ou sur erreur API — on n'écrit jamais une synthèse
    partielle. `level` vaut `global` (toute la consultation) ou `theme` (+`theme_id`).
    """
    t0 = perf_counter()
    level = (level or "global").strip().lower()
    if level not in ("global", "theme"):
        raise ValueError(f"level inconnu: {level!r} (attendu: global | theme).")
    synth_model = model or mistral_client.SYNTHESIS_MODEL

    if level == "theme":
        if not theme_id:
            raise ValueError("level='theme' exige un `id` de thème.")
        node = tree.get(theme_id)
        if node is None:
            raise ValueError(f"thème inconnu: {theme_id!r} (dataset {tree.dataset!r}).")
        target_label = node.title or node.label
    else:
        node = None
        target_label = "global"

    def _stamp(extra: dict) -> dict:
        return {"dataset": tree.dataset, "level": level, "id": theme_id,
                "target": target_label, "model": synth_model,
                "took_ms": round((perf_counter() - t0) * 1000), **extra}

    if not mistral_client.available():
        return {
            "markdown": "_Synthèse indisponible : clé Mistral manquante "
                        "(`MISTRAL_API_KEY` non configurée)._",
            "meta": _stamp({"fallback": True, "reason": "no_api_key"}),
        }

    def _llm(task: str, data: str) -> str:
        messages = [{"role": "system", "content": _HARNESS_SYSTEM},
                    {"role": "user", "content": f"{task}\n\nDonnées :\n\n{data}\n"}]
        return _strip_code_fence(mistral_client.chat(
            messages, model=synth_model, temperature=0.3, max_tokens=_SECTION_MAX_TOKENS))

    try:
        if level == "theme":
            sections = [
                ("Vue générale", _llm(_TASK_IDENTITE, _identity_data(tree, node, child_insights))),
                ("À relever", _llm(_TASK_TENSION, _tension_data(tree, node, opinion, child_insights))),
            ]
        else:
            # Introduction retirée (redondante avec Contexte) → Contexte auto-suffisant.
            sections = [
                ("Contexte", _llm(_TASK_CONTEXTE, _global_data(tree))),
                ("Profil du panel", _profil_panel(tree)),  # dérivé — pas de LLM
            ]
    except mistral_client.MistralError as exc:
        return {
            "markdown": f"_Synthèse indisponible : l'appel à Mistral a échoué "
                        f"(statut {exc.status})._",
            "meta": _stamp({"fallback": True, "reason": f"api_error:{exc.status}"}),
        }

    markdown = "\n\n".join(f"## {h}\n{(c or '').strip()}" for h, c in sections if c and c.strip())
    return {"markdown": markdown,
            "meta": _stamp({"fallback": False, "sections": [h for h, _ in sections]})}


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()



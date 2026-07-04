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

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

from backend.analysis import (
    DEFAULT_RESOLUTION,
    ThemeNode,
    ThemeTree,
    _dataset_context,
    get_or_build_tree,
)
from backend.analysis_store import INSIGHTS_DIRNAME  # même DIRNAME, source unique
from backend.llm_cache import DISK, MEMORY, cached_llm
from backend.recluster import dataset_dir
from pipeline.cluster import mistral_client

# DEUX caches d'insights partagent le DIRNAME "insights" mais vivent sous des parents
# DIFFÉRENTS et avec des schémas de nom DIFFÉRENTS — ce n'est pas un doublon, ce sont deux
# étages :
#   • BAKÉ  (analysis_store.insights_path) : <dataset>/analysis/insights/<global|theme_id>.json
#     — précalculé au BUILD, servi tel quel quand l'analyse est prête (nom SÉMANTIQUE) ;
#   • LIVE  (_disk_path ci-dessous)        : <dataset>/insights/<key_hash>.json
#     — repli à la demande hors analyse bakée, caché par HASH de contenu.
# On importe le DIRNAME depuis analysis_store pour n'avoir qu'un seul littéral "insights".
INSIGHTS_MAX_TOKENS = 1200          # rapport COURT (tient dans le panneau)
MAX_THEMES_IN_GLOBAL = 40           # garde-fou structurel de taille de prompt
REP_PER_THEME = 2                   # claims représentatives citées par thème

# Cache MÉMOIRE : (dataset, level, id, model, resolution) -> payload. Le disque
# persiste entre redémarrages ; la mémoire sert l'acceptance « 2ᵉ appel rapide ».
_MEM_CACHE: dict[tuple, dict] = {}


def _theme_block(node: ThemeNode, children: list[ThemeNode]) -> str:
    """Bloc texte compact d'un thème (label, poids, mots-clés, sous-thèmes, verbatims)."""
    kw = ", ".join((node.keywords or [])[:6])
    head = (f"[{node.id}] {node.label} — {node.n_avis} avis, poids {node.weight}, "
            f"cohérence {node.consensus}, dispersion {node.dispersion}")
    lines = [head]
    if kw:
        lines.append(f"  mots-clés : {kw}")
    for rep in node.representative_claims[:REP_PER_THEME]:
        lines.append(f"  • {rep}")
    for c in children:
        ckw = ", ".join((c.keywords or [])[:4])
        lines.append(f"  ↳ sous-thème [{c.id}] {c.label} ({c.n_avis} avis"
                     + (f" ; {ckw}" if ckw else "") + ")")
    return "\n".join(lines)


def _global_summary(tree: ThemeTree) -> str:
    """Résumé compact de la consultation : tous les macro-thèmes (triés par poids)."""
    macros = [tree.nodes[mid] for mid in tree.macros]
    macros = [m for m in macros if m.n_claims > 0]
    truncated = len(macros) > MAX_THEMES_IN_GLOBAL
    macros = macros[:MAX_THEMES_IN_GLOBAL]
    prep = tree.prepared
    lines = [
        f"Consultation : {tree.dataset}",
        f"Avis analysés : {len(prep.avis)} · claims extraites : {len(prep.claim_texts)}",
        f"Grands thèmes : {len(macros)}" + (" (tronqué)" if truncated else ""),
        "",
        "THÈMES (du plus au moins porté) :",
    ]
    for m in macros:
        children = [tree.nodes[c] for c in m.children]
        lines.append("")
        lines.append(_theme_block(m, children))
    return "\n".join(lines)


def _theme_summary(tree: ThemeTree, node: ThemeNode,
                   child_insights: dict[str, str] | None = None) -> str:
    """Résumé compact d'UN thème pour la synthèse ciblée.

    BOTTOM-UP : si le thème a des sous-thèmes ET que leurs synthèses Markdown sont
    déjà générées (`child_insights` : id_enfant → markdown), le résumé du PARENT est
    bâti par AGRÉGATION des synthèses de ses enfants — pas de ses propres claims. Une
    FEUILLE (ou un parent dont aucun enfant n'a encore de synthèse) retombe sur ses
    claims représentatives (comportement historique). Cf. `_summary_from_children` /
    `_summary_from_claims`."""
    children = [tree.nodes[c] for c in node.children]
    have_children_md = bool(children) and bool(child_insights) and any(
        (child_insights or {}).get(c.id) for c in children)
    return (_summary_from_children(node, children, child_insights or {})
            if have_children_md else _summary_from_claims(node, children))


def _summary_header(node: ThemeNode) -> list[str]:
    """Entête commune (label + métriques + mots-clés) d'un résumé de thème."""
    lines = [
        f"Thème : {node.label}",
        f"Poids social : {node.weight} · avis : {node.n_avis} · claims : {node.n_claims}",
        f"Cohérence interne : {node.consensus} · dispersion : {node.dispersion}",
    ]
    if node.keywords:
        lines.append(f"Mots-clés : {', '.join(node.keywords[:8])}")
    return lines


def _summary_from_claims(node: ThemeNode, children: list[ThemeNode]) -> str:
    """Résumé « historique » d'un thème : ses claims représentatives + aperçu enfants."""
    lines = _summary_header(node)
    lines.append("")
    lines.append("Claims représentatives :")
    for rep in node.representative_claims:
        lines.append(f"  • {rep}")
    if children:
        lines.append("")
        lines.append("Sous-thèmes :")
        for c in children:
            ckw = ", ".join((c.keywords or [])[:5])
            lines.append(f"  - [{c.id}] {c.label} ({c.n_avis} avis"
                         + (f" ; {ckw}" if ckw else "") + ")")
            for rep in c.representative_claims[:1]:
                lines.append(f"      • {rep}")
    return "\n".join(lines)


def _summary_from_children(node: ThemeNode, children: list[ThemeNode],
                           child_insights: dict[str, str]) -> str:
    """Résumé BOTTOM-UP : les synthèses déjà rédigées des sous-thèmes, à agréger.

    Chaque sous-thème apporte sa synthèse Markdown complète (repli sur ses mots-clés +
    2 claims s'il n'en a pas). Le prompt parent (`_theme_messages(from_children=True)`)
    demande d'AGRÉGER, pas de recopier."""
    lines = _summary_header(node)
    lines.append("")
    lines.append("Synthèses des sous-thèmes (à AGRÉGER en une synthèse du thème "
                 "parent, sans les recopier telles quelles) :")
    for c in children:
        md = (child_insights.get(c.id) or "").strip()
        lines.append("")
        lines.append(f"— Sous-thème [{c.id}] {c.label} — {c.n_avis} avis :")
        if md:
            lines.append(md)
        else:  # enfant sans synthèse (repli/erreur) → mots-clés + 2 claims
            ckw = ", ".join((c.keywords or [])[:5])
            if ckw:
                lines.append(f"  mots-clés : {ckw}")
            for rep in c.representative_claims[:2]:
                lines.append(f"  • {rep}")
    return "\n".join(lines)


def _global_messages(summary: str) -> list[dict]:
    system = (
        "Tu es analyste de consultations citoyennes pour des parlementaires. Tu "
        "produis des synthèses neutres, factuelles et concises à partir de thèmes "
        "déjà regroupés automatiquement. Tu n'inventes rien hors des données fournies."
    )
    user = (
        "À partir du résumé des thèmes ci-dessous (regroupement automatique de "
        "contributions citoyennes), rédige une SYNTHÈSE GLOBALE d'introduction, en "
        "Markdown TRÈS COURT.\n\n"
        "IMPORTANT — NE liste PAS et n'énumère PAS les thèmes : ils sont affichés "
        "séparément, sous forme cliquable, juste après ta synthèse. Ne fais donc AUCUNE "
        "puce par thème, AUCUNE liste de sujets.\n\n"
        "Écris seulement UN à DEUX paragraphes (sans titre de section) qui posent :\n"
        "1. le CONTEXTE de la consultation (de quoi il s'agit, ce sur quoi les citoyens "
        "se sont exprimés) ;\n"
        "2. une ANALYSE BRÈVE de la tonalité générale et du MESSAGE CENTRAL qui ressort "
        "des contributions — par ex. « Les contributions révèlent une critique massive "
        "de … perçue comme … ».\n\n"
        "TERMINE sur ton analyse (une phrase de conclusion). N'ajoute SURTOUT AUCUNE "
        "phrase d'amorce vers une liste de thèmes (du type « Les principaux thèmes "
        "sont : ») : cette amorce est ajoutée séparément, l'écrire ferait doublon. "
        "Rédige dans la langue dominante des contributions. Reste TRÈS COURT. N'invente "
        "aucun chiffre absent du résumé.\n\n"
        f"Résumé des thèmes :\n\n{summary}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _theme_messages(summary: str, *, from_children: bool = False) -> list[dict]:
    system = (
        "Tu es analyste de consultations citoyennes pour des parlementaires. Tu "
        "synthétises la parole citoyenne d'UN thème, de façon neutre et factuelle, "
        "sans rien inventer hors des données fournies."
    )
    if from_children:
        # BOTTOM-UP : le résumé fourni est fait des synthèses déjà rédigées des
        # sous-thèmes → on demande une AGRÉGATION (vue d'ensemble), pas une redite.
        user = (
            "Ci-dessous, les SYNTHÈSES déjà rédigées des SOUS-THÈMES d'un thème (issu "
            "d'un regroupement automatique de contributions citoyennes). Rédige la "
            "SYNTHÈSE DU THÈME PARENT en AGRÉGEANT ces synthèses, en Markdown COURT, "
            "structurée ainsi :\n\n"
            "## Ce que disent les citoyens\n"
            "La vue d'ensemble du thème : ce que ses sous-thèmes ont en commun, ce qui "
            "les distingue, leur poids relatif, les points de convergence et de tension. "
            "N'énumère pas mécaniquement les sous-thèmes : fais-en une vraie synthèse.\n\n"
            "## À retenir\n"
            "2 à 4 puces : l'essentiel pour un décideur.\n\n"
            "Rédige dans la langue dominante des contributions. Reste COURT. N'invente "
            "aucun chiffre ni fait absent des synthèses fournies.\n\n"
            f"Résumé du thème et synthèses des sous-thèmes :\n\n{summary}\n"
        )
    else:
        user = (
            "À partir du résumé d'UN thème ci-dessous (issu d'un regroupement automatique "
            "de contributions citoyennes), rédige une SYNTHÈSE DU THÈME en Markdown COURT, "
            "structurée ainsi :\n\n"
            "## Ce que disent les citoyens\n"
            "Le cœur de la parole sur ce thème : préoccupations, propositions, nuances. "
            "Si des sous-thèmes existent, montre comment ils se répartissent.\n\n"
            "## À retenir\n"
            "2 à 4 puces : l'essentiel pour un décideur.\n\n"
            "Rédige dans la langue dominante des contributions. Reste COURT. N'invente "
            "aucun chiffre absent du résumé.\n\n"
            f"Résumé du thème :\n\n{summary}\n"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _attach_global_context(out: dict, dataset_id: str) -> dict:
    """B2 : la synthèse GLOBALE s'OUVRE sur le contexte de collecte du dataset.

    Le contexte (descripteur d'ingestion, cf. `_dataset_context`) est préfixé en
    italique au Markdown de l'insight global → une seule synthèse qui commence par le
    contexte (le front F2 n'affiche plus de bloc intro séparé). `dataset_context` reste
    par ailleurs exposé dans `/analysis` (repli). Sans contexte, Markdown inchangé.
    """
    if out.get("meta", {}).get("level") != "global":
        return out
    ctx = _dataset_context(dataset_id)
    if not ctx:
        return out
    md = out.get("markdown", "")
    if md.startswith(f"_{ctx}_"):          # idempotent (déjà préfixé)
        return out
    return {**out, "markdown": f"_{ctx}_\n\n{md}"}


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


def _disk_path(dataset: str, key_hash: str) -> Path:
    # Cache LIVE (repli hors analyse bakée) : <dataset>/insights/<hash>.json — à distinguer
    # du cache BAKÉ `analysis_store.insights_path` (<dataset>/analysis/insights/<name>.json).
    return dataset_dir(dataset) / INSIGHTS_DIRNAME / f"{key_hash}.json"


def _cache_key(dataset: str, level: str, theme_id: str | None, model: str,
               resolution: float, child_insights: dict[str, str] | None = None) -> tuple:
    """Clé de cache d'un insight.

    BOTTOM-UP : la synthèse d'un thème PARENT dépend des MARKDOWNS de ses ENFANTS.
    Sans les inclure, un re-bake avec des synthèses enfants CHANGÉES ferait un cache
    HIT sur l'ANCIENNE synthèse du parent. On ajoute donc un marqueur `bottomup` + un
    hash sha256 court des markdowns enfants (triés par id → déterministe, indépendant de
    l'ordre d'insertion). Une FEUILLE (ou tout appel sans `child_insights`) garde la clé
    STABLE (base seule) — aucun changement de comportement."""
    base = (dataset, level, theme_id or "", model, round(resolution, 4))
    if child_insights:
        joined = "\x00".join(f"{cid}\x01{child_insights[cid]}"
                             for cid in sorted(child_insights))
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
        return (*base, "bottomup", digest)
    return base


def _key_hash(key: tuple) -> str:
    return hashlib.sha256("\x00".join(str(k) for k in key).encode("utf-8")).hexdigest()[:16]


def insights_payload(ds, **kwargs) -> dict:
    """Synthèse Markdown LLM cachée ; la GLOBALE s'ouvre sur le contexte (B2).

    Le contexte est attaché à la VOLÉE (après cache) : le Markdown caché reste « pur »
    (insensible à un changement de descripteur), le contexte est toujours frais.
    """
    out = _insights_payload(ds, **kwargs)
    return _attach_global_context(out, ds.id)


def _insights_payload(
    ds,
    *,
    level: str = "global",
    theme_id: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    embedder: str | None = None,
    resolution: float = DEFAULT_RESOLUTION,
    refresh: bool = False,
    child_insights: dict[str, str] | None = None,
) -> dict:
    """Synthèse Markdown LLM d'un niveau de zoom, CACHÉE par (dataset, level, id).

    `level="global"` synthétise toute la consultation ; `level="theme"` + `theme_id`
    synthétise un thème. Renvoie `{markdown, meta}`. L'arbre est réutilisé depuis le
    cache mémoire (`get_or_build_tree`). 2ᵉ appel identique = servi du cache (mémoire
    puis disque) sans rappeler le LLM. Repli gracieux sans clé Mistral.
    """
    t0 = perf_counter()
    level = (level or "global").strip().lower()
    if level not in ("global", "theme"):
        raise ValueError(f"level inconnu: {level!r} (attendu: global | theme).")
    if level == "theme" and not theme_id:
        raise ValueError("level='theme' exige un `id` de thème.")

    synth_model = mistral_client.SYNTHESIS_MODEL
    key = _cache_key(ds.id, level, theme_id, synth_model, resolution, child_insights)

    # Arbre + résumé + libellé construits PARESSEUSEMENT : seulement sur cache miss
    # (réutilisés depuis le cache mémoire de `get_or_build_tree`). `state` mémorise pour
    # ne bâtir qu'une fois même si messages ET repli sont sollicités.
    state: dict = {}

    def _prepare() -> dict:
        if not state:
            kw = {} if embedder is None else {"embedder": embedder}
            tree = get_or_build_tree(ds, backend=backend, model=model,
                                     resolution=resolution, **kw)
            if level == "global":
                summary = _global_summary(tree)
                target_label = "global"
            else:
                node = tree.get(theme_id)
                if node is None:
                    raise ValueError(f"thème inconnu: {theme_id!r} (dataset {ds.id!r}).")
                summary = _theme_summary(tree, node)
                target_label = node.label
            build = _global_messages if level == "global" else _theme_messages
            state.update(messages=build(summary), target_label=target_label)
        return state

    def _stamp(extra: dict) -> dict:
        return {"dataset": ds.id, "level": level, "id": theme_id,
                "target": _prepare()["target_label"], "model": synth_model,
                "took_ms": round((perf_counter() - t0) * 1000), **extra}

    def _fallback(reason: str, exc=None) -> dict:
        if reason == "no_api_key":
            return {"markdown": "_Synthèse indisponible : clé Mistral manquante "
                                "(`MISTRAL_API_KEY` non configurée)._",
                    "meta": _stamp({"fallback": True, "reason": "no_api_key",
                                    "cache": "miss"})}
        return {"markdown": f"_Synthèse indisponible : l'appel à Mistral a échoué "
                            f"(statut {exc.status})._",
                "meta": _stamp({"fallback": True, "reason": f"api_error:{exc.status}",
                                "cache": "miss"})}

    value, source = cached_llm(
        mem_cache=_MEM_CACHE,
        key=key,
        disk_path=_disk_path(ds.id, _key_hash(key)),
        build_messages=lambda: _prepare()["messages"],
        fallback_fn=_fallback,                    # repli NON caché (réessai dès la clé revenue)
        model=synth_model,
        max_tokens=INSIGHTS_MAX_TOKENS,
        temperature=0.3,
        postprocess=lambda content: {"markdown": _strip_code_fence(content),
                                     "meta": _stamp({"fallback": False, "cache": "miss"})},
        cache_fallback=False,
        refresh=refresh,
    )
    # Sur HIT, re-tamponne la provenance et le temps (le Markdown caché reste « pur »).
    if source in (MEMORY, DISK):
        return {**value, "meta": {**value["meta"], "cache": source,
                                  "took_ms": round((perf_counter() - t0) * 1000)}}
    return value

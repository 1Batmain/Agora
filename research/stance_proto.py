"""PROTO — STANCE sur le SUJET DU CLUSTER (vs cibles per-claim). Agora · R&D pur.

Hypothèse testée
----------------
La stance citoyenne s'AGRÈGE proprement si le **sujet** envers lequel on classe la
position vient du **CLUSTER** (canonique, un seul sujet par thème) plutôt que des
**cibles per-claim** (idiosyncrasiques : ~la moitié sont des pronoms / bouts
d'argument / jugements, non agrégeables). On le prouve sur 2 macros TikTok à sujet
clair : un autour de l'ADDICTION / temps d'écran, un autour du HARCÈLEMENT.

Chemin (aucun fichier produit modifié — tout est en lecture seule + research/) :
  1. CLUSTER  : `backend.live_cluster.build_live_tree('tiktok', k=défaut)` → macros.
  2. SUJET    : titre LLM canonique du nœud (`backend.titles.title_for_node`).
  3. STANCE   : pour chaque membre (prise de position citoyenne verbatim), un appel
                LLM (`pipeline.cluster.mistral_client`, mistral-small) classe la
                position ENVERS le sujet ∈ {favorable, defavorable, nuance} + justif.
                Multilingue, batché.
  4. AGRÉGAT  : « N défavorables / M favorables / K nuancés sur [sujet] » par cluster.
  5. COMPARE  : les cibles per-claim des MÊMES avis (depuis `claims.json`, lecture
                seule) — dispersion + part inexploitable (déictiques/courtes) — VS le
                sujet unique du cluster.

Lancement (depuis la racine du worktree) :
  MISTRAL_API_KEY=$(cat var/mistral.key) \
  uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
  python research/stance_proto.py

Sorties : research/stance_proto_results.json (chiffres bruts) ; le verdict rédigé
est dans research/stance_proto_note.md.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from backend.recluster import load_cache
from backend.live_cluster import build_live_tree
from backend.titles import title_for_node
from pipeline.cluster import mistral_client

SEED = 42
DATASET = "tiktok"
RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = RESEARCH_DIR / "stance_proto_results.json"

STANCE_MODEL = os.environ.get("AGORA_MISTRAL_MODEL", "mistral-small-latest")
BATCH = 10                      # contributions par appel LLM (batché pour le coût)
REP_FOR_TITLE = 8               # membres représentatifs passés au titreur LLM

# Le secret + l'extraction `claims.json` vivent dans le dépôt principal ; ce worktree
# R&D n'embarque ni `var/` ni le cache claims. On les lit en SEULE LECTURE.
MAIN_REPO = Path("/home/bat/projects/Analyse-des-consultations-citoyennes")
CLAIMS_JSON = MAIN_REPO / "backend" / "cache" / DATASET / "claims.json"
KEY_FALLBACK = MAIN_REPO / "var" / "mistral.key"

# Familles lexicales pour repérer les 2 macros « à sujet clair » (générique : on
# SÉLECTIONNE par mots-clés, on n'IMPOSE aucun contenu).
THEME_LEXICON = {
    "addiction": {"addiction", "dependance", "dépendance", "scroller", "scroll",
                  "accro", "temps", "ecran", "écran", "heures", "arreter", "arrêter",
                  "procrastination", "telephone", "téléphone"},
    "harcelement": {"harcelement", "harcèlement", "haine", "insultes", "insulte",
                    "harceler", "commentaires", "mechant", "méchant", "cyberharcelement"},
}

# Déictiques / pronoms / connecteurs : une cible réduite à ça est INEXPLOITABLE comme
# sujet agrégeable (elle ne dit pas DE QUOI on parle hors contexte de l'avis).
DEICTIC = {
    "ça", "ca", "cela", "ceci", "c'", "c", "ce", "cet", "cette", "ces", "il", "ils",
    "elle", "elles", "on", "nous", "je", "j'", "tu", "vous", "eux", "leur", "leurs",
    "le", "la", "les", "lui", "y", "en", "ceux", "celles", "celui", "celle", "qui",
    "que", "quoi", "dont", "où", "ou", "et", "mais", "donc", "car", "this", "that",
    "it", "they", "them", "we", "i", "he", "she",
}


# --------------------------------------------------------------------------- #
# Prompt STANCE — le cœur du proto.
# --------------------------------------------------------------------------- #
STANCE_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne UN SUJET (l'objet de "
    "débat d'un thème) et des CONTRIBUTIONS citoyennes verbatim (multilingues). Pour "
    "chaque contribution, classe la PRISE DE POSITION de l'auteur ENVERS LE SUJET en "
    "exactement une étiquette :\n"
    "  - \"favorable\"   : la contribution VALORISE, défend, minimise, ou s'adonne "
    "sans regret au sujet (le voit positivement / en veut plus) ;\n"
    "  - \"defavorable\" : la contribution DÉNONCE, critique, déplore le sujet ou veut "
    "le limiter / s'en protéger (le voit négativement) ;\n"
    "  - \"nuance\"      : position ambivalente, conditionnelle, ou aucune position "
    "claire envers le sujet.\n"
    "Juge la position envers le SUJET, pas la qualité de l'écriture. Réponds en JSON "
    "strict : {\"results\":[{\"i\":<int>,\"stance\":\"favorable|defavorable|nuance\","
    "\"justif\":\"<≤15 mots, langue de la contribution>\"}]}. Une entrée par "
    "contribution, dans l'ordre, rien d'autre."
)


def stance_batch(subject: str, items: list[tuple[int, str]]) -> dict[int, dict]:
    """Classe un lot de contributions ENVERS `subject`. Renvoie {i: {stance, justif}}."""
    lines = [f"[{i}] {text}" for i, text in items]
    user = (
        f"SUJET : {subject}\n\n"
        f"CONTRIBUTIONS (réponds pour chaque [indice]) :\n" + "\n".join(lines)
    )
    messages = [
        {"role": "system", "content": STANCE_SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = mistral_client.chat(
        messages, model=STANCE_MODEL, temperature=0.0, max_tokens=1400, json_mode=True
    )
    data = json.loads(raw)
    out: dict[int, dict] = {}
    for rec in data.get("results", []):
        try:
            idx = int(rec["i"])
        except (KeyError, ValueError, TypeError):
            continue
        stance = str(rec.get("stance", "")).strip().lower()
        if stance not in {"favorable", "defavorable", "nuance"}:
            stance = "nuance"
        out[idx] = {"stance": stance, "justif": str(rec.get("justif", "")).strip()}
    return out


def run_stance(subject: str, members: list[tuple[int, str]]) -> dict[int, dict]:
    """STANCE sur TOUS les membres (batché). Repli unitaire si un lot casse."""
    results: dict[int, dict] = {}
    for start in range(0, len(members), BATCH):
        batch = members[start:start + BATCH]
        try:
            got = stance_batch(subject, batch)
        except (mistral_client.MistralError, json.JSONDecodeError) as exc:
            got = {}
            print(f"    ⚠ lot {start}: {type(exc).__name__} — repli unitaire", flush=True)
        # Repli : tout membre non rendu par le lot est re-tenté seul.
        for i, text in batch:
            if i not in got:
                try:
                    single = stance_batch(subject, [(i, text)])
                    got.update(single)
                except (mistral_client.MistralError, json.JSONDecodeError):
                    got[i] = {"stance": "nuance", "justif": "(échec LLM)"}
        results.update(got)
        print(f"    stance {min(start + BATCH, len(members))}/{len(members)}", flush=True)
        time.sleep(0.05)
    return results


# --------------------------------------------------------------------------- #
# Sélection des 2 macros + sujet canonique.
# --------------------------------------------------------------------------- #
def _norm(tok: str) -> str:
    return re.sub(r"[^a-zà-ÿ']", "", tok.lower())


def pick_macros(tree) -> dict[str, str]:
    """Pour chaque thème visé, le macro dont les mots-clés matchent le plus le lexique."""
    chosen: dict[str, str] = {}
    used: set[str] = set()
    for theme, lex in THEME_LEXICON.items():
        best, best_score = None, -1
        for mid in tree.macros:
            if mid in used:
                continue
            kws = {_norm(k) for k in (tree.nodes[mid].keywords or [])}
            score = len(kws & {_norm(w) for w in lex})
            if score > best_score:
                best, best_score = mid, score
        chosen[theme] = best
        used.add(best)
    return chosen


def subject_for(node, ideas) -> str:
    """Sujet canonique = titre LLM du nœud. Peuple d'abord ses claims représentatives
    (membres les plus proches du centroïde) — build_live_tree ne les pose pas."""
    members = node.members
    cent = np.asarray(node.centroid, dtype=np.float64)
    # cos au centroïde via les textes : on n'a pas les vecs ici → approxime par les
    # membres de plus haut poids ; suffisant pour donner du contexte au titreur.
    reps = [(ideas[i].text_clean or ideas[i].text) for i in members[:REP_FOR_TITLE]]
    node.representative_claims = [r[:240] for r in reps]
    title = title_for_node(DATASET, node)
    return title or node.label


# --------------------------------------------------------------------------- #
# Comparaison : cibles per-claim (claims.json) des MÊMES avis.
# --------------------------------------------------------------------------- #
def load_claims() -> dict[str, list[dict]]:
    if not CLAIMS_JSON.exists():
        print(f"⚠ claims.json absent ({CLAIMS_JSON}) — comparaison cibles désactivée.")
        return {}
    rec = json.loads(CLAIMS_JSON.read_text(encoding="utf-8"))
    return rec.get("claims", {})


def target_text(idea, span) -> str | None:
    if not span:
        return None
    t0, t1 = span
    src = idea.text_clean or idea.text
    if 0 <= t0 < t1 <= len(src):
        return src[t0:t1].strip()
    return None


def is_unusable(cible: str | None) -> bool:
    """Une cible est INEXPLOITABLE comme sujet agrégeable si absente, déictique pure,
    ou trop courte/générique (1 mot non substantiel)."""
    if not cible:
        return True
    toks = [_norm(t) for t in cible.split() if _norm(t)]
    if not toks:
        return True
    content = [t for t in toks if t not in DEICTIC]
    if not content:                     # que des déictiques/pronoms/connecteurs
        return True
    if len(content) == 1 and len(content[0]) <= 3:   # mono-mot très court
        return True
    return False


def cibles_for(members: list[int], ideas, claims_by_avis) -> dict:
    """Agrège les cibles per-claim des avis membres → stats de dispersion + part inutile."""
    cibles: list[str | None] = []
    n_with_claim = 0
    for i in members:
        avis_id = ideas[i].id
        avis_claims = claims_by_avis.get(avis_id, [])
        if avis_claims:
            n_with_claim += 1
        for c in avis_claims:
            cibles.append(target_text(ideas[i], c.get("target")))
    usable = [c for c in cibles if not is_unusable(c)]
    distinct_usable = {c.lower() for c in usable}
    return {
        "n_members": len(members),
        "n_members_with_claim": n_with_claim,
        "n_claims": len(cibles),
        "n_cibles_unusable": sum(1 for c in cibles if is_unusable(c)),
        "n_cibles_usable": len(usable),
        "n_distinct_usable_cibles": len(distinct_usable),
        "frac_unusable": round(sum(1 for c in cibles if is_unusable(c)) / max(1, len(cibles)), 3),
        "top_distinct": Counter(c.lower() for c in usable).most_common(12),
        "sample_cibles": [c for c in cibles[:25]],
    }


# --------------------------------------------------------------------------- #
def analyse_cluster(theme: str, mid: str, tree, ideas, claims_by_avis) -> dict:
    node = tree.nodes[mid]
    subject = subject_for(node, ideas)
    print(f"\n=== {theme} :: {mid} (n={len(node.members)}) ===", flush=True)
    print(f"    sujet canonique (titre LLM) : {subject!r}", flush=True)

    members = [(i, (ideas[i].text_clean or ideas[i].text)) for i in node.members]
    stance = run_stance(subject, members)

    counts = Counter(stance[i]["stance"] for i, _ in members if i in stance)
    examples = []
    for i, text in members[:12]:
        if i in stance:
            examples.append({
                "avis_id": ideas[i].id,
                "claim": text[:200],
                "stance": stance[i]["stance"],
                "justif": stance[i]["justif"],
            })

    cibles = cibles_for(node.members, ideas, claims_by_avis)
    n = len(members)
    return {
        "theme": theme,
        "node_id": mid,
        "label_ctfidf": node.label,
        "keywords": node.keywords[:8],
        "subject": subject,
        "n_members": n,
        "stance_counts": {
            "favorable": counts.get("favorable", 0),
            "defavorable": counts.get("defavorable", 0),
            "nuance": counts.get("nuance", 0),
        },
        "stance_dominant_frac": round(max(counts.values()) / max(1, n), 3) if counts else 0.0,
        "examples": examples,
        "cibles_per_claim": cibles,
    }


def main() -> None:
    # Bootstrap clé : si l'env n'a pas la clé (worktree sans var/), tenter le dépôt
    # principal en lecture seule. R&D uniquement.
    if not mistral_client.available() and KEY_FALLBACK.exists():
        os.environ["MISTRAL_API_KEY"] = KEY_FALLBACK.read_text(encoding="utf-8").strip()
    if not mistral_client.available():
        sys.exit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")

    print("Chargement cache + clustering live (k défaut)…", flush=True)
    ideas, vecs, weights = load_cache(DATASET)
    tree = build_live_tree(ideas, vecs, weights, seed=SEED)
    print(f"  {len(ideas)} idées, {len(tree.macros)} macros", flush=True)

    chosen = pick_macros(tree)
    for theme, mid in chosen.items():
        kw = ", ".join(tree.nodes[mid].keywords[:6])
        print(f"  {theme} → {mid} [{kw}]", flush=True)

    claims_by_avis = load_claims()

    clusters = [analyse_cluster(theme, mid, tree, ideas, claims_by_avis)
                for theme, mid in chosen.items()]

    results = {
        "dataset": DATASET,
        "seed": SEED,
        "stance_model": STANCE_MODEL,
        "stance_prompt_system": STANCE_SYSTEM,
        "n_ideas": len(ideas),
        "n_macros": len(tree.macros),
        "clusters": clusters,
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ écrit {RESULTS_PATH}", flush=True)

    # Récap console.
    for c in clusters:
        sc = c["stance_counts"]
        cb = c["cibles_per_claim"]
        print(f"\n── {c['theme']} : « {c['subject']} » (n={c['n_members']})")
        print(f"   STANCE  favorable={sc['favorable']}  defavorable={sc['defavorable']}  "
              f"nuance={sc['nuance']}  (dominant {c['stance_dominant_frac']*100:.0f}%)")
        print(f"   CIBLES  {cb['n_claims']} claims → {cb['n_distinct_usable_cibles']} cibles "
              f"distinctes exploitables, {cb['frac_unusable']*100:.0f}% inexploitables")


if __name__ == "__main__":
    main()

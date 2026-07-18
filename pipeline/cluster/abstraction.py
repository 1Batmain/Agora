"""Moteur d'ABSTRACTION — couche macro par étiquette canonique + affectation embedding.

Au-dessus de la couche PLATE (γ, pic de modularité), on regroupe les thèmes REDONDANTS en
macros SANS souder les sujets distincts. Validé (`research/abstraction_note.md`) :

  1. le LLM normalise chaque thème en une ÉTIQUETTE canonique (3-6 mots) — surface → sens,
     ce qui rapproche les redondants (les ~5 « addiction » deviennent la même catégorie) ;
  2. le LLM PROPOSE un petit jeu de CATÉGORIES abstraites (sa force : nommer l'abstrait) ;
  3. l'affectation thème → catégorie se fait par EMBEDDING (produit scalaire dans l'espace
     recentré) — ce qui GARANTIT une partition stricte, là où le regroupement LLM libre
     double-assigne les thèmes ambigus.

Fonctions PURES (chat_fn / embed_fn injectés → testable, découplé de Mistral). Le résultat est
DÉTERMINISÉ par le cache disque (`compute` une fois au build, `load` par les autres étapes) —
indispensable à la cohérence de l'arbre entre build_analysis / build_opinion / build_arguments.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MIN_THEMES = 4        # en dessous, la couche plate suffit (pas d'abstraction)
LABEL_MAX_TOKENS = 25


def _canonical_label(claims: list[str], *, chat_fn, model: str) -> str:
    ex = "\n".join(f"- {c[:180]}" for c in claims[:15])
    msg = [
        {"role": "system", "content":
         "Donne le SUJET de ce groupe de témoignages en 3 à 6 mots, sous forme de CATÉGORIE "
         "canonique et générique (ex: « dépendance aux réseaux sociaux », « protection des "
         "mineurs »). Juste la catégorie, rien d'autre."},
        {"role": "user", "content": f"Témoignages :\n{ex}\n\nCatégorie :"},
    ]
    return chat_fn(msg, model=model, temperature=0.0, max_tokens=LABEL_MAX_TOKENS).strip()


def _group_labels(labels: list[str], *, chat_fn, model: str) -> list[dict]:
    """Le LLM regroupe les étiquettes par SUJET DE FOND (fusionne les redondances). Renvoie une
    liste `[{"titre":..., "indices":[...]}]`. Raisonne sur le SENS (pas la surface) → route bien
    les cas ambigus (« utilisation excessive » = addiction, pas « risque social »)."""
    lst = "\n".join(f"{i}. {l}" for i, l in enumerate(labels))
    msg = [
        {"role": "system", "content":
         "On te donne une liste de SUJETS de thèmes. Regroupe ceux qui traitent du MÊME sujet de "
         "FOND (fusionne les redondances : plusieurs formulations d'une même idée = un groupe), et "
         "garde DISTINCTS les sujets réellement différents (même s'ils partagent un contexte). Vise "
         "plusieurs groupes équilibrés, pas un ou deux fourre-tout. Réponds en JSON : "
         '{"groupes": [{"titre": "catégorie courte et neutre", "indices": [numéros]}]}.'},
        {"role": "user", "content": lst},
    ]
    raw = chat_fn(msg, model=model, temperature=0.0, max_tokens=500, json_mode=True)
    try:
        return json.loads(raw).get("groupes", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def compute(cluster_texts: list[list[str]], *, chat_fn, embed_fn=None, model: str) -> dict | None:
    """Calcule la couche macro. `cluster_texts[i]` = claims représentatifs du thème i.

    Étiquette canonique par thème (surface→sens), PUIS regroupement LLM par sujet de fond. Le
    regroupement libre peut double-assigner un thème AMBIGU : on DÉDUPLIQUE (premier groupe qui
    le réclame gagne) → partition stricte. `embed_fn` n'est plus requis (gardé pour compat).

    Renvoie `{"labels":[...], "macros":[...], "assign":[macro par thème]}` ou `None` si trop peu
    de thèmes, ou si tout retombe dans un seul macro (pas d'abstraction utile).
    """
    n = len(cluster_texts)
    if n < MIN_THEMES:
        return None
    labels = [_canonical_label(c, chat_fn=chat_fn, model=model) for c in cluster_texts]
    groupes = _group_labels(labels, chat_fn=chat_fn, model=model)

    assign = [-1] * n
    titles: list[str] = []
    for g in groupes:
        gi = len(titles)
        placed = False
        for idx in g.get("indices", []):
            if isinstance(idx, int) and 0 <= idx < n and assign[idx] == -1:
                assign[idx] = gi
                placed = True
        if placed:
            titles.append(str(g.get("titre") or "").strip() or labels[
                next(i for i in range(n) if assign[i] == gi)])
    # Thèmes oubliés par le LLM → chacun son propre macro (jamais perdus).
    for i in range(n):
        if assign[i] == -1:
            assign[i] = len(titles)
            titles.append(labels[i])

    used = sorted(set(assign))
    if len(used) < 2:
        return None                                  # tout dans un macro = pas d'abstraction
    remap = {m: i for i, m in enumerate(used)}
    return {"labels": labels, "macros": [titles[m] for m in used],
            "assign": [remap[a] for a in assign]}


# --- Cache disque : DÉTERMINISE l'abstraction entre les étapes du build ------------------- #
def signature(clusters: list[list[int]]) -> str:
    """Empreinte STABLE de la partition plate (indépendante de l'ordre des clusters)."""
    key = repr(sorted(tuple(sorted(c)) for c in clusters))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def load(path: Path, clusters: list[list[int]]) -> dict | None:
    """Relit l'abstraction cachée si elle correspond à CETTE partition (sinon None)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("signature") != signature(clusters):
        return None                                  # partition changée → cache périmé
    return data.get("result")


def save(path: Path, clusters: list[list[int]], result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": signature(clusters), "result": result},
                               ensure_ascii=False, indent=2), encoding="utf-8")

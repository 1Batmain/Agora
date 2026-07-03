"""CALIBRATION des poids sauce_magique (backend/recut.py) contre les golds.

But : remplacer les poids v1 posés à la main (α=1, β=0.5, γ=1, δ=1) par des poids
CALIBRÉS — ceux dont le CLASSEMENT des coupes candidates par la fonction objectif
corrèle le mieux (Spearman) avec le classement d'un score GOLD indépendant, agrégé
sur 3 datasets. AUCUN rebuild, AUCUN appel LLM : on ne fait que varier la COUPE des
arbres existants (antichaînes) et scorer.

Coupes candidates (par dataset, ~20-40 visées, dédupliquées) :
  - coupes de NIVEAU fixe (racines, niveau 1, 2, … feuilles) ;
  - coupes GLOUTONNES `best_cut` sous ~35 rayons de poids variés (grille) ;
  - coupes ALÉATOIRES (expansions aléatoires depuis les racines, arrêts aléatoires).
  Cas granddebat : l'arbre en cache est DÉJÀ re-coupé (n0 dissous) et quasi plat
  (2 nœuds internes) → on RECONSTITUE le géant pré-recoupe comme racine virtuelle
  (n_avis=22 172, cohésion 0.606, chiffres tracés dans params.recut) pour inclure
  la façade pathologique dans les candidats. Peu de coupes distinctes possibles
  (~5) : granddebat = point de contrôle grossier, le signal fin vient de
  xstance/repnum.

Scores GOLD (indépendants de la fonction objectif) :
  - xstance : NMI(affectation des claims à la coupe, topic officiel du gold
    x-stance porté par chaque avis) — calculable SANS LLM ;
  - granddebat : F1 d'appariement par EMBEDDINGS (nomic local) entre les nœuds de
    la coupe (titre+mots-clés) et les 14 sous-thèmes officiels OpinionWay
    (recall = couverture des officiels, precision = pertinence pondérée voix) ;
  - republique-numerique : idem embeddings vs les sous-thèmes officiels du projet
    de loi (Titres I-III, cf. research/repnum_note.md) ;
  - VALIDATION du proxy embeddings : sur xstance on calcule AUSSI le F1 embeddings
    (vs les 12 topics) et sa corrélation avec le NMI — si elle est haute, le proxy
    utilisé pour granddebat/repnum est crédible.

Calibration : grille α,β,γ,δ ∈ {0.25,0.5,1,2}^4, dédupliquée par RAYON (le
classement est invariant à l'échelle des poids). Chaque terme de la fonction
objectif ne dépend que de la coupe → 4 termes précalculés par coupe, score sous
tout rayon = produit scalaire. Sélection = meilleur Spearman moyen sur datasets.

Usage :
  uv run --extra contender --extra embed-contender --extra faiss --with fastapi \
      python research/sauce_calibration.py

Sortie : research/sauce_calibration_results.json ; verdict rédigé dans
research/sauce_magique_calibration.md.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.recut import DUST_SHARE, W as W_V1, best_cut  # noqa: E402

CACHE = ROOT / "backend" / "cache"
OUT = Path(__file__).resolve().parent / "sauce_calibration_results.json"
GRID = (0.25, 0.5, 1.0, 2.0)
SEED = 42

# --------------------------------------------------------------------------- #
# GOLDS — descriptions officielles (témoins, pas du code pipeline).
# --------------------------------------------------------------------------- #

# granddebat : les 14 sous-thèmes officiels OpinionWay (source unique :
# research/granddebat_witness.py, dict OFFICIAL), importés ci-dessous.


def _load_official_granddebat() -> dict[str, str]:
    import ast
    import re

    src = (Path(__file__).resolve().parent / "granddebat_witness.py").read_text()
    m = re.search(r"^OFFICIAL = (\{.*?^\})", src, re.S | re.M)
    return ast.literal_eval(m.group(1))


# republique-numerique : sous-thèmes officiels du plan du projet de loi
# (Titres I-III), transcrits de research/repnum_note.md (gold thématique).
OFFICIAL_REPNUM = {
    "open_data": "Ouverture des données publiques : open data par défaut des "
        "administrations, transparence de l'action publique.",
    "donnees_interet_general": "Données d'intérêt général : ouverture de données "
        "privées d'utilité publique (délégataires, subventions).",
    "service_public_donnee": "Service public de la donnée : données de référence "
        "publiques, qualité et disponibilité.",
    "open_access": "Économie du savoir : libre accès (open access) aux publications "
        "scientifiques issues de la recherche publique.",
    "tdm_fouille": "Fouille de textes et de données (text and data mining, TDM) "
        "pour la recherche.",
    "domaine_commun": "Domaine commun informationnel : biens communs de la "
        "connaissance, domaine public, logiciels libres et licences libres.",
    "neutralite_net": "Neutralité du net : égalité de traitement du trafic par les "
        "opérateurs, non-discrimination des contenus.",
    "portabilite": "Portabilité et récupération des données par l'utilisateur "
        "(changer de service sans perdre ses données).",
    "loyaute_plateformes": "Loyauté des plateformes : transparence des plateformes "
        "en ligne, information du consommateur, avis en ligne.",
    "vie_privee_cnil": "Protection de la vie privée et des données personnelles : "
        "pouvoirs et sanctions de la CNIL, consentement.",
    "droit_oubli": "Droit à l'oubli, notamment pour les mineurs : déréférencement, "
        "effacement des données.",
    "mort_numerique": "Mort numérique : sort des données personnelles après le "
        "décès, directives.",
    "confidentialite_correspondances": "Confidentialité des correspondances privées "
        "électroniques (courriels, messageries).",
    "couverture_reseaux": "Couverture et connectivité du territoire : réseaux, "
        "fibre, mobile, débit internet, zones blanches.",
    "inclusion_numerique": "Inclusion numérique des publics fragiles : maintien de "
        "la connexion internet, accompagnement, éducation au numérique.",
    "accessibilite_handicap": "Accessibilité des services numériques et "
        "téléphoniques aux personnes handicapées (sourds, malvoyants).",
}

# xstance : les 12 topics officiels (gold PAR AVIS dans les props ; les
# descriptions ne servent qu'au F1-embeddings de VALIDATION du proxy).
OFFICIAL_XSTANCE = {
    "Infrastructure & Environment": "Infrastructure et environnement : transports, "
        "énergie, aménagement, protection de l'environnement et du climat.",
    "Welfare": "Protection sociale : retraites, âge de la retraite, assurances "
        "sociales, aides, prestations.",
    "Education": "Éducation : école, enseignement, formation, universités.",
    "Economy": "Économie : entreprises, marché du travail, salaires, régulation "
        "économique.",
    "Immigration": "Immigration : étrangers, asile, naturalisation, intégration.",
    "Society": "Société : famille, égalité, religion, culture, modes de vie.",
    "Security": "Sécurité : police, armée, criminalité, défense.",
    "Healthcare": "Santé : assurance maladie, soins, hôpitaux, prévention.",
    "Foreign Policy": "Politique étrangère : relations internationales, accords "
        "avec l'UE, libre-échange, coopération.",
    "Finances": "Finances publiques : impôts, taxes, budget de l'État, dépenses.",
    "Political System": "Système politique : institutions, droits populaires, "
        "votations, démocratie directe.",
    "Digitisation": "Numérisation : numérique, internet, e-government, "
        "cybersécurité, données.",
}


# --------------------------------------------------------------------------- #
# Chargement des arbres + reconstruction du géant granddebat pré-recoupe.
# --------------------------------------------------------------------------- #

def load_themes(ds: str) -> list[dict]:
    a = json.loads((CACHE / ds / "analysis" / "analysis.json").read_text())
    return [
        {"id": t["id"], "parent_id": t.get("parent_id"), "n_avis": t["n_avis"],
         "cohesion": t.get("consensus") or 0.0, "title": t.get("title") or "",
         "keywords": t.get("keywords") or []}
        for t in a["themes"]
    ]


def granddebat_with_virtual_root(themes: list[dict]) -> list[dict]:
    """Rajoute le géant n0 dissous par la re-coupe v1 comme racine virtuelle.

    Chiffres tracés dans params.recut (avant : top1=0.999, cohésion pondérée
    0.606 ≈ cohésion propre de n0 vu sa part) — les 17 racines réelles (>1 avis)
    étaient ses enfants, les 20 singletons de bruit restent racines.
    """
    out = [dict(t) for t in themes]
    real_roots = [t for t in out if t["parent_id"] is None and t["n_avis"] > 1]
    for t in real_roots:
        t["parent_id"] = "_virt_n0"
    out.append({"id": "_virt_n0", "parent_id": None, "n_avis": 22172,
                "cohesion": 0.606, "title": "", "keywords": []})
    return out


# --------------------------------------------------------------------------- #
# Génération des coupes candidates (antichaînes couvrant les feuilles).
# --------------------------------------------------------------------------- #

def children_map(themes: list[dict]) -> dict:
    kids = defaultdict(list)
    for t in themes:
        kids[t["parent_id"]].append(t["id"])
    return kids


def level_cuts(themes: list[dict], kids: dict) -> list[frozenset]:
    """Coupe « à profondeur d » : nœud si depth==d, ou feuille moins profonde."""
    depth = {}
    stack = [(i, 0) for i in kids[None]]
    while stack:
        nid, d = stack.pop()
        depth[nid] = d
        stack.extend((c, d + 1) for c in kids.get(nid, []))
    maxd = max(depth.values())
    cuts = []
    for d in range(maxd + 1):
        cut = frozenset(nid for nid, dd in depth.items()
                        if dd == d or (dd < d and not kids.get(nid)))
        cuts.append(cut)
    return cuts


def greedy_cuts(themes: list[dict], rays: list[dict]) -> list[frozenset]:
    return [frozenset(n["id"] for n in best_cut(themes, weights=w)[0]) for w in rays]


def random_cuts(kids: dict, n_wanted: int, rng: random.Random) -> list[frozenset]:
    cuts = []
    for _ in range(n_wanted * 3):
        cut = set(kids[None])
        expandable = [n for n in cut if kids.get(n)]
        steps = rng.randint(1, max(1, len(kids))) if expandable else 0
        for _ in range(steps):
            expandable = [n for n in cut if kids.get(n)]
            if not expandable:
                break
            n = rng.choice(expandable)
            cut.discard(n)
            cut.update(kids[n])
            if rng.random() < 0.15:      # arrêt anticipé aléatoire
                break
        cuts.append(frozenset(cut))
        if len(set(cuts)) >= n_wanted:
            break
    return cuts


def gen_candidates(themes: list[dict], target: int = 40) -> list[frozenset]:
    kids = children_map(themes)
    rng = random.Random(SEED)
    # rayons de génération : coins {0.25,2}^4 + milieux {0.5,1}^4 + v1 (~33)
    gen_rays = ([dict(zip(("alpha", "beta", "gamma", "delta"), v))
                 for v in itertools.product((0.25, 2.0), repeat=4)]
                + [dict(zip(("alpha", "beta", "gamma", "delta"), v))
                   for v in itertools.product((0.5, 1.0), repeat=4)]
                + [dict(W_V1)])
    cand = level_cuts(themes, kids) + greedy_cuts(themes, gen_rays)
    seen = list(dict.fromkeys(cand))          # dédup en préservant l'ordre
    if len(seen) < target:
        for c in random_cuts(kids, (target - len(seen)) * 2, rng):
            if c not in seen:
                seen.append(c)
            if len(seen) >= target:
                break
    return seen[:target]


# --------------------------------------------------------------------------- #
# Termes de la fonction objectif — précalculés par coupe (score = w·termes).
# --------------------------------------------------------------------------- #

def objective_terms(cut_nodes: list[dict]) -> tuple[float, float, float, float]:
    voices = [max(1, n["n_avis"]) for n in cut_nodes]
    total = sum(voices)
    shares = [v / total for v in voices]
    coh = sum(s * (n.get("cohesion") or 0.0) for s, n in zip(shares, cut_nodes))
    ent = -sum(s * math.log(s) for s in shares if s > 0)
    n_eff = math.exp(ent)
    n_cible = max(6.0, math.log(total) * 1.4)
    dust = sum(s for s in shares if s < DUST_SHARE)
    top1 = max(shares)
    return (1 - coh, abs(math.log(n_eff / n_cible)), dust, top1)


# --------------------------------------------------------------------------- #
# Scores GOLD.
# --------------------------------------------------------------------------- #

def ancestor_in_cut(cut: frozenset, parent: dict[str, str | None]) -> dict[str, str]:
    """leaf/nœud → son ancêtre (ou lui-même) membre de la coupe."""
    out = {}
    for nid in parent:
        cur = nid
        while cur is not None and cur not in cut:
            cur = parent[cur]
        if cur is not None:
            out[nid] = cur
    return out


def nmi(labels_a: list, labels_b: list) -> float:
    """NMI (normalisation arithmétique), implémentation locale sans sklearn."""
    n = len(labels_a)
    ca, cb, cab = defaultdict(int), defaultdict(int), defaultdict(int)
    for a, b in zip(labels_a, labels_b):
        ca[a] += 1
        cb[b] += 1
        cab[(a, b)] += 1
    ha = -sum(c / n * math.log(c / n) for c in ca.values())
    hb = -sum(c / n * math.log(c / n) for c in cb.values())
    mi = sum(c / n * math.log(n * c / (ca[a] * cb[b])) for (a, b), c in cab.items())
    denom = (ha + hb) / 2
    return mi / denom if denom > 0 else 0.0


def gold_nmi_xstance(cut: frozenset, parent: dict, claim_leaves: list[str],
                     claim_topics: list[str]) -> float:
    anc = ancestor_in_cut(cut, parent)
    assign = [anc[leaf] for leaf in claim_leaves]
    return nmi(assign, claim_topics)


def gold_embed_f1(cut: frozenset, node_vec: dict[str, np.ndarray],
                  node_voice: dict[str, int], gold_mat: np.ndarray) -> float:
    """F1 d'appariement embeddings coupe ↔ sous-thèmes officiels.

    recall = moyenne sur les officiels du max de cos vers les nœuds de la coupe
    (couverture) ; precision = moyenne PONDÉRÉE VOIX sur les nœuds du max de cos
    vers les officiels (pertinence de la façade). Nœuds sans titre (racine
    virtuelle) : similarité 0 (façade illisible).
    """
    ids = sorted(cut)
    mat = np.stack([node_vec[i] for i in ids])            # (n_cut, d)
    sims = mat @ gold_mat.T                               # (n_cut, n_gold)
    recall = float(sims.max(axis=0).mean())
    voices = np.array([node_voice[i] for i in ids], dtype=float)
    shares = voices / voices.sum()
    precision = float((sims.max(axis=1) * shares).sum())
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


# --------------------------------------------------------------------------- #
# Spearman (rangs moyens pour les ex-æquo).
# --------------------------------------------------------------------------- #

def _ranks(x: list[float]) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x))
    xs = np.asarray(x)[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum()) / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #

def main() -> None:
    from pipeline.embed.embedder import Embedder
    emb = Embedder()          # nomic-v2 (défaut du pipeline), local CPU

    def embed_texts(texts: list[str]) -> np.ndarray:
        return emb.embed(texts)

    officials = {
        "granddebat": _load_official_granddebat(),
        "republique-numerique": OFFICIAL_REPNUM,
        "xstance": OFFICIAL_XSTANCE,
    }

    datasets = {}
    for ds in ("granddebat", "xstance", "republique-numerique"):
        themes = load_themes(ds)
        if ds == "granddebat":
            themes = granddebat_with_virtual_root(themes)
        cands = gen_candidates(themes)
        node_texts = {
            t["id"]: (f"{t['title']}. Mots-clés : {', '.join(t['keywords'][:10])}"
                      if t["title"] else "")
            for t in themes
        }
        with_text = [i for i, txt in node_texts.items() if txt]
        vecs = embed_texts([node_texts[i] for i in with_text])
        node_vec = {i: v for i, v in zip(with_text, vecs)}
        dim = vecs.shape[1]
        for i, txt in node_texts.items():          # racine virtuelle sans titre
            if not txt:
                node_vec[i] = np.zeros(dim, dtype=np.float32)
        gold_mat = embed_texts(list(officials[ds].values()))
        datasets[ds] = {
            "themes": themes,
            "by_id": {t["id"]: t for t in themes},
            "parent": {t["id"]: t["parent_id"] for t in themes},
            "cuts": cands,
            "node_vec": node_vec,
            "node_voice": {t["id"]: t["n_avis"] for t in themes},
            "gold_mat": gold_mat,
        }
        print(f"[{ds}] {len(themes)} nœuds, {len(cands)} coupes candidates "
              f"(tailles {sorted({len(c) for c in cands})[:12]}…)")

    # gold xstance = NMI sur les claims (label topic officiel par avis)
    ideas_topic = {}
    with open(CACHE / "xstance" / "ideas.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ideas_topic[d["id"]] = d["props"].get("topic")
    avis = json.loads((CACHE / "xstance" / "analysis" / "avis.json").read_text())
    claim_leaves, claim_topics = [], []
    for aid, a in avis.items():
        for c in a.get("claims") or []:
            claim_leaves.append(c["leaf_id"])
            claim_topics.append(ideas_topic[aid])

    # -------- scores gold + termes objectif, par coupe --------------------- #
    results = {}
    for ds, D in datasets.items():
        golds, terms, sizes = [], [], []
        embf_check = []                     # proxy embeddings (validation xstance)
        for cut in D["cuts"]:
            cut_nodes = [D["by_id"][i] for i in cut]
            terms.append(objective_terms(cut_nodes))
            sizes.append(len(cut))
            f1 = gold_embed_f1(cut, D["node_vec"], D["node_voice"], D["gold_mat"])
            if ds == "xstance":
                golds.append(gold_nmi_xstance(cut, D["parent"], claim_leaves,
                                              claim_topics))
                embf_check.append(f1)
            else:
                golds.append(f1)
        results[ds] = {"golds": golds, "terms": terms, "sizes": sizes,
                       "size_gold_spearman": round(spearman([float(s) for s in sizes],
                                                            golds), 3)}
        print(f"[{ds}] Spearman(taille de coupe, gold) = "
              f"{results[ds]['size_gold_spearman']:+.3f} — monotonie granularité")
        if ds == "xstance":
            rho = spearman(embf_check, golds)
            results[ds]["proxy_embed_vs_nmi_spearman"] = round(rho, 3)
            print(f"[xstance] validation proxy embeddings : "
                  f"Spearman(F1-embed, NMI) = {rho:.3f} sur {len(golds)} coupes")

    # -------- grille de poids (rayons dédupliqués) -------------------------- #
    rays, seen = [], set()
    for v in itertools.product(GRID, repeat=4):
        key = tuple(round(x / sum(v), 6) for x in v)
        if key not in seen:
            seen.add(key)
            rays.append(v)
    print(f"{len(rays)} rayons de poids uniques (grille {GRID}^4)")

    def eval_ray(v: tuple) -> dict:
        per_ds = {}
        for ds, R in results.items():
            scores = [sum(w * t for w, t in zip(v, tt)) for tt in R["terms"]]
            per_ds[ds] = spearman([-s for s in scores], R["golds"])
        per_ds["mean"] = sum(per_ds.values()) / len(results)
        return per_ds

    table = []
    for v in rays:
        r = eval_ray(v)
        table.append({"weights": dict(zip(("alpha", "beta", "gamma", "delta"), v)),
                      **{k: round(x, 4) for k, x in r.items()}})
    table.sort(key=lambda r: -r["mean"])
    v1_eval = eval_ray((W_V1["alpha"], W_V1["beta"], W_V1["gamma"], W_V1["delta"]))
    v1_row = {"weights": dict(W_V1), **{k: round(x, 4) for k, x in v1_eval.items()}}

    print("\n=== TOP 10 rayons (Spearman moyen) ===")
    for r in table[:10]:
        w = r["weights"]
        print(f"  α={w['alpha']:<5} β={w['beta']:<5} γ={w['gamma']:<5} δ={w['delta']:<5}"
              f" | mean={r['mean']:+.3f}  gd={r['granddebat']:+.3f}"
              f"  xs={r['xstance']:+.3f}  rn={r['republique-numerique']:+.3f}")
    print(f"\n  v1 (1, 0.5, 1, 1)                    | mean={v1_row['mean']:+.3f}"
          f"  gd={v1_row['granddebat']:+.3f}  xs={v1_row['xstance']:+.3f}"
          f"  rn={v1_row['republique-numerique']:+.3f}")

    # -------- façades produites : v1 vs meilleur rayon ---------------------- #
    best_w = table[0]["weights"]
    facades = {}
    for ds, D in datasets.items():
        row = {}
        for name, w in (("v1", W_V1), ("calibre", best_w)):
            cut, detail = best_cut(D["themes"], weights=w)
            fs = frozenset(n["id"] for n in cut)
            if ds == "xstance":
                g = gold_nmi_xstance(fs, D["parent"], claim_leaves, claim_topics)
            else:
                g = gold_embed_f1(fs, D["node_vec"], D["node_voice"], D["gold_mat"])
            row[name] = {**detail, "gold": round(g, 4)}
        row["same_cut"] = (row["v1"]["n_clusters"] == row["calibre"]["n_clusters"]
                           and row["v1"]["score"] != row["calibre"]["score"] or None)
        facades[ds] = row
        print(f"\n[{ds}] façade v1      : {row['v1']}")
        print(f"[{ds}] façade calibrée : {row['calibre']}")

    OUT.write_text(json.dumps({
        "grid": GRID, "n_rays": len(rays),
        "candidates": {ds: {"n_cuts": len(D["cuts"]),
                            "size_gold_spearman": results[ds]["size_gold_spearman"],
                            "rows": sorted(
                                ({"size": s, "gold": round(g, 4),
                                  "terms": [round(t, 4) for t in tt]}
                                 for s, g, tt in zip(results[ds]["sizes"],
                                                     results[ds]["golds"],
                                                     results[ds]["terms"])),
                                key=lambda r: r["size"])}
                       for ds, D in datasets.items()},
        "proxy_embed_vs_nmi_spearman_xstance":
            results["xstance"]["proxy_embed_vs_nmi_spearman"],
        "top20": table[:20], "v1": v1_row, "best": table[0],
        "facades": facades,
    }, ensure_ascii=False, indent=2))
    print(f"\nÉcrit → {OUT}")


if __name__ == "__main__":
    main()

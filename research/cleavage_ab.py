"""STANCE — track B : cible de clivage (a) actuel vs (b) « mots-clés distinctifs + contraste ».

Le verdict `research/stance_target_ab_verdict.md` a tranché sur TIKTOK (panel aveugle 7-2) :
adopter la variante (b) — cadrer les mots-clés c-TF-IDF comme « ce qui SÉPARE ce thème des
voisins » + consigne de contraste dans `cleavage_system`. RÉSERVE explicite du verdict :
« n=1 corpus (tiktok, FR, mono-sujet). Rien ne se transpose sur granddebat sans re-mesure. »

Ce proto TESTE LE TRANSFERT sur un corpus granddebat-style (lutte-contre-les-fausses-
informations) via la métrique objective du verdict : `cleavage_fit` = cos(emb(cible),
emb(titre)) dans l'espace claims (nomic-v2). On dérive (a) ET (b) sur des ENTRÉES IDENTIQUES
(titre + mots-clés + contributions représentatives de `analysis.json`) → on isole l'effet
PROMPT seul (exactement ce que change le verdict). On mesure : fit médian, nb de cibles
`fit_low` (< 0.75), et le signe du delta par thème.

ISOLATION : lit les caches servis, dérive via Mistral, n'écrit que sous `research/`. Ne touche
NI `backend/build_opinion.py` NI la prod NI les caches servis.

Usage :
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/cleavage_ab.py --dataset lutte-contre-les-fausses-informations
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parent.parent
CACHED = ROOT / "cached_data"
MODEL = "mistral-large-latest"
FIT_LOW = 0.75            # même seuil que build_opinion.CLEAVAGE_FIT_LOW


# --------------------------------------------------------------------------- #
# Prompts — (a) = REPRIS TEL QUEL de backend/build_opinion.cleavage_system.
# --------------------------------------------------------------------------- #
def cleavage_a(title: str) -> str:
    return (
        "Tu es analyste de consultations citoyennes. On te donne le TITRE d'un THÈME, ses "
        "MOTS-CLÉS et des CONTRIBUTIONS verbatim. Identifie l'OBJET DE CLIVAGE qui RÉSUME "
        f"le débat CENTRAL de CE thème, intitulé « {title} » : la proposition ou mesure "
        "PRÉCISE, au cœur du thème, sur laquelle des citoyens peuvent être POUR ou CONTRE. "
        "Elle doit capturer le SUJET CENTRAL du thème (ce dont parle le titre), PAS une "
        "facette secondaire ni le détail le plus bruyant. Formule-la comme une proposition "
        "polaire COURTE (≤12 mots), neutre et débattable, à l'infinitif ou nominale. "
        "Réponds en JSON strict : {\"objet\":\"<proposition>\",\"justif\":\"<≤14 mots>\"}."
    )


# (b) = variante VALIDÉE (tiktok) : mots-clés = « ce qui SÉPARE ce thème des voisins » +
# consigne de CONTRASTE. Seule différence avec (a) : le cadrage des mots-clés et l'ajout
# de la phrase de contraste (rien d'autre — « prompt only »).
def cleavage_b(title: str) -> str:
    return (
        "Tu es analyste de consultations citoyennes. On te donne le TITRE d'un THÈME, ses "
        "MOTS-CLÉS DISTINCTIFS (ce qui SÉPARE ce thème des thèmes voisins) et des "
        "CONTRIBUTIONS verbatim. Identifie l'OBJET DE CLIVAGE qui RÉSUME le débat CENTRAL de "
        f"CE thème, intitulé « {title} » : la proposition ou mesure PRÉCISE, au cœur du "
        "thème, sur laquelle des citoyens peuvent être POUR ou CONTRE. Elle doit capturer le "
        "SUJET CENTRAL du thème (ce dont parle le titre), PAS une facette secondaire ni le "
        "détail le plus bruyant. IMPORTANT — la proposition doit MOBILISER ce qui DISTINGUE "
        "ce thème (ses mots-clés distinctifs), pour ne pas être un énoncé passe-partout "
        "interchangeable avec un thème voisin. Formule-la comme une proposition polaire "
        "COURTE (≤12 mots), neutre et débattable, à l'infinitif ou nominale. "
        "Réponds en JSON strict : {\"objet\":\"<proposition>\",\"justif\":\"<≤14 mots>\"}."
    )


def derive(system: str, title: str, keywords: list[str], reps: list[str]) -> str:
    kw = ", ".join(keywords[:10])
    contribs = "\n".join(f"- {r[:160]}" for r in reps[:14])
    user = f"MOTS-CLÉS : {kw}\n\nCONTRIBUTIONS :\n{contribs}"
    try:
        raw = mistral_client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=MODEL, temperature=0.0, max_tokens=200, json_mode=True)
        return str(json.loads(raw).get("objet", "")).strip() or title
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return title


def run(dataset: str) -> dict:
    base = CACHED / dataset
    op = json.loads((base / "analysis" / "opinion.json").read_text())
    an = {t["id"]: t for t in json.loads((base / "analysis" / "analysis.json").read_text())["themes"]}
    leaves = [t for t in op["themes"] if not t.get("is_aggregate")]
    print(f"[cleavage-ab] {dataset} · {len(leaves)} feuilles")

    # Cache des dérivations LLM (48 appels) — l'embedding en aval ne re-paye jamais le LLM.
    deriv_cache = ROOT / "research" / f"cleavage_ab_deriv.{dataset}.json"
    if deriv_cache.exists():
        rows = json.loads(deriv_cache.read_text())
        print(f"[cleavage-ab] dérivations rechargées ({len(rows)}) depuis le cache")
    else:
        rows = []
        for lf in leaves:
            meta = an.get(lf["theme_id"], {})
            kw = meta.get("keywords") or []
            if isinstance(kw, str):
                kw = [k.strip() for k in kw.split(",")]
            reps = meta.get("representative_claims") or []
            title = lf.get("title") or meta.get("title") or lf["theme_id"]
            prop_a = derive(cleavage_a(title), title, kw, reps)
            prop_b = derive(cleavage_b(title), title, kw, reps)
            rows.append({"theme_id": lf["theme_id"], "title": title,
                         "prop_a": prop_a, "prop_b": prop_b})
        deriv_cache.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    # Embeddings : titres + (a) + (b) en un batch, espace claims (nomic-v2).
    from pipeline.claims.pipeline import embed_claim_texts
    titles = [r["title"] for r in rows]
    props_a = [r["prop_a"] for r in rows]
    props_b = [r["prop_b"] for r in rows]
    tv = embed_claim_texts(titles)
    av = embed_claim_texts(props_a)
    bv = embed_claim_texts(props_b)
    for r, t, a, b in zip(rows, tv, av, bv):
        r["fit_a"] = round(max(0.0, float(np.dot(a, t))), 4)
        r["fit_b"] = round(max(0.0, float(np.dot(b, t))), 4)
        r["delta"] = round(r["fit_b"] - r["fit_a"], 4)

    fa = [r["fit_a"] for r in rows]
    fb = [r["fit_b"] for r in rows]
    wins_b = sum(1 for r in rows if r["delta"] > 0.005)
    wins_a = sum(1 for r in rows if r["delta"] < -0.005)
    summary = {
        "dataset": dataset, "n_leaves": len(rows),
        "fit_a_median": round(statistics.median(fa), 4),
        "fit_b_median": round(statistics.median(fb), 4),
        "fit_a_mean": round(statistics.mean(fa), 4),
        "fit_b_mean": round(statistics.mean(fb), 4),
        "fit_low_a": sum(1 for x in fa if x < FIT_LOW),
        "fit_low_b": sum(1 for x in fb if x < FIT_LOW),
        "themes_b_better": wins_b, "themes_a_better": wins_a,
        "themes_tie": len(rows) - wins_b - wins_a,
    }
    print(f"[cleavage-ab] fit médian : a={summary['fit_a_median']} b={summary['fit_b_median']}")
    print(f"[cleavage-ab] fit_low (<{FIT_LOW}) : a={summary['fit_low_a']} b={summary['fit_low_b']}")
    print(f"[cleavage-ab] par thème : (b) mieux {wins_b} · (a) mieux {wins_a} · nul {summary['themes_tie']}")
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="AB cible de clivage (a) vs (b) — track stance (R&D).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "cleavage_ab_results.json"))
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral (MISTRAL_API_KEY). Abandon.")
    result = run(args.dataset)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[cleavage-ab] → {args.out}")


if __name__ == "__main__":
    main()

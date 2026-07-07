"""V-EXTRACT — argument mining verbatim par EXTRACTION de spans (fork B + ablation embedding).

Suite du proto `argmine_verbatim.py`. Idée (Bob) : une fois la CIBLE établie (cleavage T2
par thème, la version robuste), extraire les arguments pour/contre comme des SPANS verbatim
RESSERRÉS — même gate que les claims (`pipeline/claims/span.align_spans` + `Claim.is_verbatim`,
rejet si pas sous-chaîne exacte de l'avis). L'argument devient une unité fine, embeddable →
analyses plus précises.

Fork B (validé) : on extrait DEPUIS les claims sélectionnés par V-SELECT (composable, ancré,
pas de 2ᵉ passe sur l'avis complet). Repli gracieux : si l'extraction ne resserre pas (rien
d'ancrable), l'argument = le claim entier (comportement V-SELECT).

ABLATION EMBEDDING — le span extrait peut être trop court pour un vecteur représentatif. Deux
espaces DISTINCTS (cf. discussion) :
  * support (argument↔claim, `back_match`) = TOUJOURS span BRUT (même espace que les claims) ;
  * analyse inter-arguments (dedup/distinction/rollup) = vecteur ENRICHI, uniforme sur tous.
Escalade validée : (1) `raw` (brut) ; (2) `target-context` = `cible + span` (0 LLM, uniforme) ;
(3) `reflet` LLM UNIFORME — UNIQUEMENT si (2) échoue. Ce script mesure (1) vs (2) ; (3) tenu en
réserve.

Read-only caches + `argmine_verbatim.py`. N'écrit que sous `research/`. Zéro touche prod.
    MISTRAL_API_KEY=$(cat var/mistral.key) \
    uv run python research/argmine_extract.py --dataset lutte-contre-les-fausses-informations
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from pipeline.claims.span import align_spans
from pipeline.cluster import mistral_client
from research.argmine_verbatim import (
    CACHED, MIN_SUPPORT, SIM_THRESHOLD, MAX_K, SELECT_MODEL,
    load_corpus, build_groups, vselect, _exclusive_support, _LABEL,
)

ROOT = Path(__file__).resolve().parent.parent
DEDUP_THRESHOLD = 0.85           # même seuil que build_arguments


# --------------------------------------------------------------------------- #
# Extraction du span argumentatif d'un claim envers la cible (gate verbatim)
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = (
    "Tu es analyste de consultations citoyennes. On te donne une CIBLE (une proposition "
    "débattable) et UNE contribution citoyenne. Extrait la ou les PORTIONS EXACTES de la "
    "contribution qui expriment sa POSITION ou son ARGUMENT envers la cible — copiées MOT POUR "
    "MOT depuis la contribution (aucune reformulation, aucun mot ajouté). Choisis la portion la "
    "plus PORTEUSE de l'argument, sans le délayage. Si toute la contribution est l'argument, "
    "renvoie-la entière. Tu peux renvoyer plusieurs portions non-contiguës si l'argument est "
    'découpé. Réponds en JSON strict : {"parts":["<portion exacte>", ...]} — rien d\'autre.'
)


# Reflet — reformulation EXPLICITE de la position (ESPACE INTER-ARGUMENTS UNIQUEMENT, jamais
# servi). Escalade validée : ne s'emploie QUE si le 0-LLM (target-context) échoue, et UNIFORME.
_REFLET_SYSTEM = (
    "On te donne une CIBLE (proposition débattable) et un EXTRAIT verbatim d'une contribution "
    "citoyenne. Formule en UNE phrase claire et autonome la POSITION que cet extrait exprime "
    "ENVERS la cible (favorable/opposé et POURQUOI), en restant STRICTEMENT fidèle à l'extrait "
    "(n'ajoute aucun argument absent). But : rendre l'extrait interprétable hors contexte. "
    'Réponds en JSON : {"reflet":"<une phrase>"}.'
)


def reflet(cible: str, span: str, *, model: str) -> str:
    """Reformulation explicite de la position (repli = le span brut si échec)."""
    messages = [{"role": "system", "content": _REFLET_SYSTEM},
                {"role": "user", "content": f"CIBLE : {cible}\n\nEXTRAIT : {span}"}]
    try:
        raw = mistral_client.chat(messages, model=model, temperature=0.0,
                                  max_tokens=120, json_mode=True)
        return str(json.loads(raw).get("reflet", "")).strip() or span
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        return span


def extract_span(cible: str, claim_text: str, avis_text: str, *, model: str) -> tuple[str, list]:
    """(texte_argument, spans) resserré sur l'avis. Repli = le claim entier si rien d'ancrable.

    Le LLM renvoie des portions ; `align_spans` les VALIDE comme sous-chaînes exactes de l'avis
    (rejet sinon → zéro mot inventé). Une portion qui franchit un ` … ` d'un claim multi-span ne
    s'ancre pas sur l'avis → écartée, ce qui est correct (fail-closed).
    """
    messages = [{"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"CIBLE : {cible}\n\nCONTRIBUTION : {claim_text}"}]
    try:
        raw = mistral_client.chat(messages, model=model, temperature=0.0,
                                  max_tokens=300, json_mode=True)
        parts = json.loads(raw).get("parts", [])
    except (mistral_client.MistralError, json.JSONDecodeError, TypeError):
        parts = []
    parts = [p for p in parts if isinstance(p, str) and p.strip()]
    if parts:
        claims = align_spans(avis_text, [{"parts": parts, "target": None}])
        if claims and claims[0].is_verbatim(avis_text):
            return claims[0].text, [list(s) for s in claims[0].spans]
    return claim_text, None  # repli V-SELECT (claim entier, déjà verbatim)


# --------------------------------------------------------------------------- #
def run(dataset: str, model: str, *, reflet_on: bool = False) -> dict:
    corpus = load_corpus(dataset)
    groups = build_groups(corpus)
    op = json.loads((CACHED / dataset / "analysis" / "opinion.json").read_text())
    prop_by_theme = {t["theme_id"]: t.get("proposition") for t in op["themes"]
                     if not t.get("is_aggregate") and t.get("proposition")}

    args: list[dict] = []   # arguments extraits, tous groupes confondus
    n_narrowed = n_fallback = 0
    for (theme_id, stance), rows in sorted(groups.items()):
        cible = prop_by_theme.get(theme_id)
        if not cible:
            continue
        gvecs = corpus["vecs"][rows]
        texts = [corpus["texts"][r] for r in rows]
        # 1) sélection des claims représentatifs (V-SELECT), repli V-CLUSTER si vide.
        seed_local = vselect(texts, model=model, max_k=MAX_K)
        if not seed_local:
            continue
        support = _exclusive_support(seed_local, gvecs, sim_threshold=SIM_THRESHOLD)
        for s in seed_local:
            members = support[s]
            if len(members) < MIN_SUPPORT:
                continue
            gi = rows[s]
            avis_text = corpus["avis"].get(corpus["owner"][gi], {}).get("text", "")
            arg_text, spans = extract_span(cible, corpus["texts"][gi], avis_text, model=model)
            narrowed = arg_text != corpus["texts"][gi]
            n_narrowed += narrowed
            n_fallback += (not narrowed)
            args.append({
                "theme_id": theme_id, "stance": _LABEL[stance], "cible": cible,
                "argument": arg_text, "narrowed": bool(narrowed),
                "claim_text": corpus["texts"][gi],
                "claim_id": f"{corpus['owner'][gi]}#{gi}",
                "seed_row": s, "member_rows_global": [rows[m] for m in members],
                "n_support": len(members),
            })

    # ── Contrôle DUR : 100 % des textes servis sont verbatim (déjà garanti par extract_span).
    ok = 0
    for a in args:
        avis_text = corpus["avis"].get(a["claim_id"].split("#")[0], {}).get("text", "")
        # un span extrait est un join de sous-chaînes d'avis ; le repli (claim entier) l'est aussi
        joined_ok = all(part in avis_text for part in a["argument"].split(" … "))
        ok += joined_ok
    verbatim_rate = round(ok / len(args), 4) if args else 1.0

    # ── Métriques de fragmentation (compression span/claim).
    comp = [len(a["argument"]) / max(1, len(a["claim_text"])) for a in args]

    # ── ABLATION EMBEDDING (espace inter-arguments) : raw vs target-context.
    from pipeline.claims.pipeline import embed_claim_texts
    raw_txt = [a["argument"] for a in args]
    tc_txt = [f"{a['cible']}. {a['argument']}" for a in args]
    raw_vecs = embed_claim_texts(raw_txt).astype(np.float32)
    tc_vecs = embed_claim_texts(tc_txt).astype(np.float32)

    # Paires INTRA-(thème,stance) = candidats doublons que le dedup fusionnerait.
    def _dup_pairs(vecs) -> set:
        pairs = set()
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                if (args[i]["theme_id"], args[i]["stance"]) != (args[j]["theme_id"], args[j]["stance"]):
                    continue
                if float(np.dot(vecs[i], vecs[j])) >= DEDUP_THRESHOLD:
                    pairs.add((i, j))
        return pairs
    raw_pairs, tc_pairs = _dup_pairs(raw_vecs), _dup_pairs(tc_vecs)

    summary = {
        "dataset": dataset, "n_arguments": len(args),
        "n_narrowed": n_narrowed, "n_fallback_whole_claim": n_fallback,
        "narrow_rate": round(n_narrowed / len(args), 3) if args else 0.0,
        "verbatim_rate": verbatim_rate,
        "compression_median": round(statistics.median(comp), 3) if comp else None,
        "dedup_pairs_raw": len(raw_pairs), "dedup_pairs_targetctx": len(tc_pairs),
        "dedup_disagreements": sorted(raw_pairs ^ tc_pairs),
    }

    # ── Escalade REFLET (uniforme) — validée UNIQUEMENT parce que le 0-LLM (target-context)
    #    échoue (contamination par le préfixe cible commun). Fidélité gardée : cos(reflet, raw).
    if reflet_on:
        reflets = [reflet(a["cible"], a["argument"], model=model) for a in args]
        rf_vecs = embed_claim_texts(reflets).astype(np.float32)
        rf_pairs = _dup_pairs(rf_vecs)
        fidelity = [float(np.dot(rf_vecs[i], raw_vecs[i])) for i in range(len(args))]
        for a, r in zip(args, reflets):
            a["reflet"] = r
        summary["dedup_pairs_reflet"] = len(rf_pairs)
        summary["reflet_fidelity_median"] = round(statistics.median(fidelity), 3) if fidelity else None
        summary["reflet_fidelity_min"] = round(min(fidelity), 3) if fidelity else None

    print(f"[extract] {len(args)} args · resserrés {n_narrowed} ({summary['narrow_rate']*100:.0f}%) "
          f"· repli claim entier {n_fallback}")
    print(f"[extract] verbatim {verbatim_rate*100:.0f}% · compression médiane "
          f"{summary['compression_median']} (span/claim)")
    print(f"[extract] paires quasi-doublons (cos≥{DEDUP_THRESHOLD}) : raw {len(raw_pairs)} · "
          f"target-context {len(tc_pairs)}"
          + (f" · reflet {summary.get('dedup_pairs_reflet')}" if reflet_on else ""))
    if reflet_on:
        print(f"[extract] reflet fidélité cos(reflet,span) médiane "
              f"{summary['reflet_fidelity_median']} · min {summary['reflet_fidelity_min']}")
    return {"summary": summary, "arguments": args}


def main() -> None:
    ap = argparse.ArgumentParser(description="V-EXTRACT — argmining par extraction verbatim (R&D).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=SELECT_MODEL)
    ap.add_argument("--reflet", action="store_true", help="escalade reflet LLM uniforme (0-LLM échoué)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "argmine_extract_results.json"))
    args = ap.parse_args()
    if not mistral_client.available():
        raise SystemExit("Pas de clé Mistral. Abandon.")
    result = run(args.dataset, args.model, reflet_on=args.reflet)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[extract] → {args.out}")


if __name__ == "__main__":
    main()

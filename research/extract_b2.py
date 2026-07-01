"""ITÉRATION sur B : battre le BRUIT de B en GARDANT sa complétude/segmentation — R&D pur.

B (prompt relâché) a gagné l'A/B (+15 % rappel, segmentation 21-9, complétude 15-6) mais
ajoute un bruit modéré (juge propreté A 20 / B 13) et SUR-FRAGMENTE parfois (« moins d'élus »
/ « trop de strates » / « responsabilités dispersées » découpés alors que c'est UNE idée à
facettes). On raffine B en gardant cible OPTIONNELLE + COMPLÉTUDE n°1 + VERBATIM strict :

  • B'  = B + SÉLECTIVITÉ resserrée  : exclut explicitement micro-claims narratifs/redondants
                                        (cadrage, redites, phrases sans position propre).
  • B'' = B + ANTI-SUR-FRAGMENTATION : facettes d'une même idée / énumération sur un même
                                        objet = UN claim ; ne séparer que des idées distinctes.
  • Bc  = B' + B''                    : les deux resserrements combinés.

On réutilise tout l'outillage de `research.extract_ab` (load_sample, extract_arm, metrics) et
le cache disque (A et B DÉJÀ cachés → on ne re-paie QUE B', B'', Bc). Gate verbatim
`align_spans` partout. Juge = comparaisons PAIRWISE de chaque variante CONTRE B (référence),
sur le MÊME sous-corpus ~40 avis grand débat multi-thèmes, lots anonymisés/ordre alterné.

Usage :
  export MISTRAL_API_KEY=$(cat var/mistral.key)
  python -m research.extract_b2            # extraction B'/B''/Bc (cachée) + métriques
  python -m research.extract_b2 --judge    # + LLM-juge pairwise vs B
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from research.extract_ab import (
    CACHE_DIR,
    CLAIM_SYS_B,
    MODEL,
    extract_arm,
    load_sample,
    metrics,
)
from pipeline.claims.extract import CLAIM_SYS
from research.extract_ab_judge import JUDGE_SYS, _complete, _fmt, pick_judge_set
from pipeline.claims.ollama import parse_json_object

# --------------------------------------------------------------------------- #
# VARIANTES. On part du SQUELETTE EXACT de B (CLAIM_SYS_B) et on ne touche QUE
# la règle visée — règle n°2 (sélectivité) pour B', règle n°3 (regroupement)
# pour B'' — sans jamais affaiblir la complétude (n°1) ni ré-imposer la cible.
# --------------------------------------------------------------------------- #

# Règle n°2 d'origine (B) — pour repérage du remplacement.
_RULE2_B = (
    "2. SÉLECTIVITÉ — n'extrais que la SUBSTANCE : une PRISE DE POSITION (grief, opinion, "
    "proposition). Laisse de côté le pur cadrage, le narratif et les annonces qui ne "
    "portent aucune position par eux-mêmes (« pour illustrer… », « mes doléances sont "
    "triples : », politesses, anecdote de contexte). Pas de bruit, pas de redite.\n"
)

# Règle n°2 RESSERRÉE (B') — exclut explicitement micro-claims narratifs/redondants.
_RULE2_PRIME = (
    "2. SÉLECTIVITÉ STRICTE — chaque claim doit porter une PRISE DE POSITION PROPRE "
    "(grief, opinion, proposition) qui apporte une idée NOUVELLE. Avant de sortir une "
    "portion, demande-toi : « contient-elle une position de fond que je n'ai pas déjà "
    "captée ? » Si NON, ne l'extrais PAS. Écarte donc systématiquement :\n"
    "   – le CADRAGE et le narratif (« pour illustrer… », « je voudrais aborder… », "
    "« mes doléances sont triples : », politesses, anecdote de contexte, annonce de plan) ;\n"
    "   – la REDITE : une portion qui ne fait que reformuler, résumer ou ré-affirmer une "
    "position déjà extraite (garde la formulation la plus nette, pas les deux) ;\n"
    "   – les apartés rhétoriques/émotionnels SANS objet (« ça me dégoûte », « c'est "
    "scandaleux ») QUAND ils ne s'attachent à aucune position de fond identifiable.\n"
    "   Cette exigence ne RÉDUIT PAS la complétude (n°1) : tu gardes TOUTE position de fond "
    "distincte ; tu n'écartes QUE ce qui n'est pas une position propre ou n'est qu'une redite. "
    "Dans le doute entre « position faible » et « bruit », tranche pour le BRUIT (n'extrais pas).\n"
)

# Règle n°3 d'origine (B) — pour repérage du remplacement.
_RULE3_B = (
    "3. REGROUPEMENT — ne FRAGMENTE pas une même idée. Restent DANS UN SEUL claim : un "
    "contraste (« X et non Y »), une justification (« … parce que … »), une condition "
    "(« si …, alors … ») et une énumération qui DÉTAILLE une seule idée. Sépare les idées "
    "distinctes, mais ne coupe pas une idée unique en morceaux.\n"
)

# Règle n°3 RENFORCÉE (B'') — anti-sur-fragmentation : facettes = UN claim.
_RULE3_SECOND = (
    "3. REGROUPEMENT FORT — une idée à plusieurs FACETTES reste UN SEUL claim. Le défaut à "
    "éviter est de SUR-FRAGMENTER : ne crée un nouveau claim QUE pour un thème vraiment "
    "DISTINCT (objet différent, prise de position différente). Restent DANS UN SEUL claim :\n"
    "   – un contraste (« X et non Y »), une justification (« … parce que … »), une "
    "condition (« si …, alors … ») ;\n"
    "   – une ÉNUMÉRATION ou des facettes qui décrivent un MÊME grief / un MÊME objet "
    "(« X, Y et Z » portant sur la même cible) → UN claim, pas trois. Ex. : « trop d'élus, "
    "trop de strates, responsabilités dispersées » sont les facettes d'UNE idée (le "
    "mille-feuille territorial) → UN claim ;\n"
    "   – plusieurs symptômes d'un même diagnostic, ou plusieurs détails d'une même "
    "proposition → UN claim.\n"
    "   Ne sépare que si l'on passe à un OBJET ou un THÈME réellement autre. Si tu hésites "
    "entre couper et regrouper des facettes d'un même objet, REGROUPE. (Cette règle ne "
    "réduit pas la complétude n°1 : deux thèmes distincts restent deux claims ; on ne "
    "regroupe QUE les facettes d'une même idée.)\n"
)


def _swap(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"[{label}] bloc à remplacer introuvable — B a changé ?")
    return text.replace(old, new, 1)


CLAIM_SYS_BP = _swap(CLAIM_SYS_B, _RULE2_B, _RULE2_PRIME, "B'")
CLAIM_SYS_BS = _swap(CLAIM_SYS_B, _RULE3_B, _RULE3_SECOND, "B''")
CLAIM_SYS_BC = _swap(_swap(CLAIM_SYS_B, _RULE2_B, _RULE2_PRIME, "Bc/2"),
                     _RULE3_B, _RULE3_SECOND, "Bc/3")

ARMS = {"Bp": CLAIM_SYS_BP, "Bs": CLAIM_SYS_BS, "Bc": CLAIM_SYS_BC}


# --------------------------------------------------------------------------- #
# JUGE PAIRWISE GÉNÉRIQUE — chaque variante CONTRE B (réf), même JUDGE_SYS,
# lots anonymisés (« lot 1 / lot 2 »), ordre ALTERNÉ pour annuler le biais.
# --------------------------------------------------------------------------- #
def judge_pair(avis, ref, var, sel, ref_name="B", var_name="?"):
    def judge_one(idx_a):
        idx, a = idx_a
        ref_is_lot1 = (idx % 2 == 0)  # alterne
        cref, cvar = ref[a.id]["claims"], var[a.id]["claims"]
        lot1, lot2 = (cref, cvar) if ref_is_lot1 else (cvar, cref)
        user = (f"AVIS :\n{a.text}\n\n--- LOT 1 ---\n{_fmt(lot1)}\n\n"
                f"--- LOT 2 ---\n{_fmt(lot2)}")
        raw = _complete([{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content": user}])
        obj = parse_json_object(raw or "") or {}

        def deanon(v):
            if v in ("1", "2"):
                winner_lot1 = (v == "1")
                # lot1 == ref ?  → ref_is_lot1
                return ref_name if (winner_lot1 == ref_is_lot1) else var_name
            return "tie"

        return {
            "id": a.id, "ds": a.ds, "ref_is_lot1": ref_is_lot1,
            "n_ref": len(cref), "n_var": len(cvar),
            "completude": deanon(obj.get("completude")),
            "segmentation": deanon(obj.get("segmentation")),
            "bruit": deanon(obj.get("bruit")),
            "justif": obj.get("justif", ""),
        }

    verdicts = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(judge_one, (i, a)) for i, a in enumerate(sel)]
        for k, fut in enumerate(as_completed(futs), 1):
            verdicts.append(fut.result())
            print(f"[juge {var_name} vs {ref_name}] {k}/{len(sel)}")

    tally = {dim: {ref_name: 0, var_name: 0, "tie": 0}
             for dim in ("completude", "segmentation", "bruit")}
    for v in verdicts:
        for dim in tally:
            tally[dim][v[dim]] += 1
    return {"pair": f"{var_name}_vs_{ref_name}", "n": len(verdicts),
            "tally": tally, "verdicts": verdicts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    avis = load_sample()
    print(f"Échantillon : {len(avis)} avis")

    # Réutilise A et B cachés ; (ré)extrait seulement les variantes manquantes.
    A = extract_arm("A", CLAIM_SYS, avis)
    B = extract_arm("B", CLAIM_SYS_B, avis)
    data = {"A": A, "B": B}
    for name, sys_text in ARMS.items():
        data[name] = extract_arm(name, sys_text, avis)

    res = {"model": MODEL, "n_avis": len(avis),
           **{k: metrics(v, avis) for k, v in data.items()}}
    (CACHE_DIR / "metrics_b2.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: res[k]["ALL"] for k in ("B", "Bp", "Bs", "Bc")},
                     ensure_ascii=False, indent=2))

    if args.judge:
        # MÊME sous-corpus pour tous les pairs : sélectionné une fois sur (B, Bc).
        sel = pick_judge_set(avis, B, data["Bc"], n=40)
        out = {"n": len(sel), "pairs": {}}
        for name in ("Bp", "Bs", "Bc"):
            j = judge_pair(avis, B, data[name], sel, ref_name="B", var_name=name)
            out["pairs"][name] = j
            print(f"== {name} vs B ==")
            print(json.dumps(j["tally"], ensure_ascii=False, indent=2))
        (CACHE_DIR / "judge_b2.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

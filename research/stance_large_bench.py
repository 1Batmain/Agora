#!/usr/bin/env python3
"""BENCH stance mistral-small vs mistral-large sur le gold x-stance (FAVOR/AGAINST).

Reproduit EXACTEMENT le protocole de validation servi (research/run_stance_validation.py) :
pour chaque contribution xstance, on lance `stance_batch` (STANCE_SYSTEM, T=0) avec cible =
la PROPRE question fermée de la contribution, en batchant par question (BATCH=10), puis on
mappe favorable→FAVOR, defavorable→AGAINST, nuance→ABSTAIN et on compare au gold `props.label`.

Ce que ce bench ajoute vs la validation : on compare PLUSIEURS configs sur le MÊME échantillon
gold (échantillonné, seedé, pour borner le coût API à ~200-400 appels) :
  - `small`        : mistral-small-latest, STANCE_SYSTEM tel quel (= la config servie) ;
  - `large`        : mistral-large-latest, STANCE_SYSTEM tel quel (politique qualité-max) ;
  - `large_noabst` : mistral-large-latest + consigne ANTI-ABSTENTION (variante conditionnelle,
                     à mesurer si le taux de « nuance » de `large` explose — incident du 4/07).

Usage :
  python research/stance_large_bench.py plan                 # taille échantillon + nb d'appels/config
  python research/stance_large_bench.py run small large      # exécute ces configs → raw jsonl
  python research/stance_large_bench.py run large_noabst
  python research/stance_large_bench.py report               # métriques → stance_large_bench_metrics.md

Sortie brute : research/stance_large_bench_raw.jsonl (une ligne par (config, contribution)).
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.build_opinion import STANCE_SYSTEM, BATCH
from pipeline.cluster import mistral_client

DATA = "backend/cache/xstance/ideas.jsonl"
RAW = "research/stance_large_bench_raw.jsonl"
SEED = 42
PER_LANG = 170  # contributions échantillonnées par langue (borne le coût API)

# Consigne anti-abstention greffée à STANCE_SYSTEM pour la variante `large_noabst`.
# But : n'autoriser « nuance » que pour une contribution VRAIMENT sans position sur l'action,
# et TRANCHER dès qu'un penchant net existe (contrer le sur-classement « nuance » de large).
ANTI_ABST = (
    "\nCONSIGNE PRIORITAIRE — ne réserve « nuance » qu'aux contributions réellement SANS "
    "position identifiable sur l'action (hors-sujet, ou strictement les deux camps à parts "
    "égales). Dès qu'un penchant net se dégage — même exprimé indirectement, par une critique "
    "ou par une condition — TRANCHE en « favorable » ou « defavorable ». « nuance » doit rester "
    "RARE ; en cas d'hésitation entre nuance et un camp, choisis le camp."
)

CONFIGS = {
    "small":        {"model": "mistral-small-latest", "system": STANCE_SYSTEM},
    "large":        {"model": "mistral-large-latest", "system": STANCE_SYSTEM},
    "large_noabst": {"model": "mistral-large-latest", "system": STANCE_SYSTEM + ANTI_ABST},
}

STANCE2GOLD = {"favorable": "FAVOR", "defavorable": "AGAINST", "nuance": "ABSTAIN"}


def load_items():
    rows = []
    for line in open(DATA, encoding="utf-8"):
        d = json.loads(line)
        p = d["props"]
        rows.append({
            "id": d["id"],
            "text": (p.get("text_clean") or p.get("text") or "").strip(),
            "question": p["question"],
            "lang": p["lang"],
            "topic": p.get("topic", ""),
            "gold": p["label"],
        })
    return rows


def sample(rows):
    """Échantillon seedé : on tire des QUESTIONS ENTIÈRES par langue (jusqu'à PER_LANG
    contributions/langue) — batcher par question reste fidèle au protocole servi et évite
    les batches partiels qui gonfleraient le nombre d'appels. Ordre stable (seedé)."""
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    rng = random.Random(SEED)
    picked = []
    for lang in sorted(by_lang):
        qs = defaultdict(list)
        for r in by_lang[lang]:
            qs[r["question"]].append(r)
        qlist = sorted(qs)
        rng.shuffle(qlist)
        taken = 0
        for q in qlist:
            if taken >= PER_LANG:
                break
            picked.extend(qs[q])
            taken += len(qs[q])
    return picked


def build_tasks(rows):
    """Regroupe par question, chunk BATCH → liste de (cible, [(gi, text), ...])."""
    for gi, r in enumerate(rows):
        r["gi"] = gi
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["question"]].append(r)
    tasks = []
    for q, group in by_q.items():
        for s in range(0, len(group), BATCH):
            chunk = group[s:s + BATCH]
            tasks.append((q, [(c["gi"], c["text"]) for c in chunk]))
    return tasks


def stance_batch_cfg(cible, items, *, model, system):
    """Copie de backend.build_opinion.stance_batch mais avec system prompt injectable."""
    lines = [f"[{i}] {text}" for i, text in items]
    user = (f"CIBLE : {cible}\n\n"
            f"CONTRIBUTIONS (réponds pour chaque [indice]) :\n" + "\n".join(lines))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    raw = mistral_client.chat(messages, model=model, temperature=0.0,
                              max_tokens=1500, json_mode=True)
    data = json.loads(raw)
    out = {}
    for rec in data.get("results", []):
        try:
            idx = int(rec["i"])
        except (KeyError, ValueError, TypeError):
            continue
        stance = str(rec.get("stance", "")).strip().lower()
        if stance not in {"favorable", "defavorable", "nuance"}:
            stance = "nuance"
        conf = str(rec.get("confidence", "")).strip().lower()
        if conf not in {"high", "medium", "low"}:
            conf = "low"
        out[idx] = {"stance": stance, "confidence": conf,
                    "justif": str(rec.get("justif", "")).strip()}
    return out


def run_config(cfg_name, rows, tasks):
    cfg = CONFIGS[cfg_name]
    model, system = cfg["model"], cfg["system"]
    print(f"[bench:{cfg_name}] {len(rows)} contribs, {len(tasks)} batches, modèle={model}",
          flush=True)
    preds = {}
    done = 0

    def work(task):
        cible, items = task
        try:
            return cible, stance_batch_cfg(cible, items, model=model, system=system)
        except (mistral_client.MistralError, json.JSONDecodeError):
            out = {}
            for i, text in items:
                try:
                    out.update(stance_batch_cfg(cible, [(i, text)], model=model, system=system))
                except (mistral_client.MistralError, json.JSONDecodeError):
                    out[i] = {"stance": "nuance", "confidence": "low", "justif": "(échec LLM)"}
            return cible, out

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            cible, got = fut.result()
            items = dict(futs[fut][1])
            for gi, text in items.items():
                if gi not in got:
                    try:
                        got.update(stance_batch_cfg(cible, [(gi, text)], model=model, system=system))
                    except (mistral_client.MistralError, json.JSONDecodeError):
                        got[gi] = {"stance": "nuance", "confidence": "low", "justif": "(échec LLM)"}
            preds.update(got)
            done += 1
            if done % 20 == 0:
                print(f"[bench:{cfg_name}] {done}/{len(tasks)} batches", flush=True)

    out_rows = []
    for r in rows:
        p = preds.get(r["gi"], {"stance": "nuance", "confidence": "low", "justif": "(manquant)"})
        out_rows.append({
            "config": cfg_name, "model": model,
            "id": r["id"], "lang": r["lang"], "topic": r["topic"],
            "question": r["question"], "gold": r["gold"],
            "stance": p["stance"], "pred": STANCE2GOLD[p["stance"]],
            "confidence": p["confidence"], "justif": p.get("justif", ""),
        })
    return out_rows


def append_raw(new_rows, configs_run):
    """Réécrit RAW en retirant les configs qu'on vient de rejouer (idempotent) puis ajoute."""
    kept = []
    if os.path.exists(RAW):
        for line in open(RAW, encoding="utf-8"):
            row = json.loads(line)
            if row.get("config") not in configs_run:
                kept.append(row)
    with open(RAW, "w", encoding="utf-8") as f:
        for row in kept + new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[bench] écrit {RAW} ({len(kept)+len(new_rows)} lignes, +{len(new_rows)} neuves)",
          flush=True)


# --------------------------------------------------------------------------- #
# Rapport
# --------------------------------------------------------------------------- #
def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def block(rows, label):
    n = len(rows)
    if not n:
        return f"### {label}\n(aucune ligne)\n"
    abst = sum(1 for x in rows if x["pred"] == "ABSTAIN")
    decided = [x for x in rows if x["pred"] != "ABSTAIN"]
    correct_a = sum(1 for x in rows if x["pred"] == x["gold"])
    correct_b = sum(1 for x in decided if x["pred"] == x["gold"])
    acc_b = correct_b / len(decided) if decided else 0.0
    lines = [f"### {label}  (n={n}, abstention={abst} = {abst/n:.1%})",
             f"- **Accuracy sur décidés (abstention exclue)** : **{acc_b:.3f}**  ({correct_b}/{len(decided)})",
             f"- Accuracy brute (abstention=erreur) : {correct_a/n:.3f}  ({correct_a}/{n})",
             ""]
    lines.append("| classe | précision | rappel | F1 | support |")
    lines.append("|---|---|---|---|---|")
    for cls in ("FAVOR", "AGAINST"):
        tp = sum(1 for x in rows if x["pred"] == cls and x["gold"] == cls)
        fp = sum(1 for x in rows if x["pred"] == cls and x["gold"] != cls)
        fn = sum(1 for x in rows if x["pred"] != cls and x["gold"] == cls)
        sup = sum(1 for x in rows if x["gold"] == cls)
        p, r, f = prf(tp, fp, fn)
        lines.append(f"| {cls} | {p:.3f} | {r:.3f} | {f:.3f} | {sup} |")
    lines.append("")
    return "\n".join(lines)


def calib(rows, label):
    out = [f"#### Calibration confiance — {label}",
           "| confiance | n | %abstention | accuracy décidés |",
           "|---|---|---|---|"]
    for conf in ("high", "medium", "low"):
        sub = [x for x in rows if x["confidence"] == conf]
        if not sub:
            continue
        abst = sum(1 for x in sub if x["pred"] == "ABSTAIN")
        dec = [x for x in sub if x["pred"] != "ABSTAIN"]
        acc = sum(1 for x in dec if x["pred"] == x["gold"]) / len(dec) if dec else 0.0
        out.append(f"| {conf} | {len(sub)} | {abst/len(sub):.1%} | {acc:.3f} |")
    out.append("")
    return "\n".join(out)


def report():
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    out = ["# Bench stance small vs large — métriques (calculées)\n",
           f"Échantillon gold x-stance seedé (SEED={SEED}, ~{PER_LANG}/langue). "
           "Même échantillon pour toutes les configs.\n"]

    # Tableau de synthèse cross-config.
    out.append("## Synthèse cross-config\n")
    out.append("| config | modèle | n | %nuance | acc décidés | acc brute |")
    out.append("|---|---|---|---|---|---|")
    for cfg in ("small", "large", "large_noabst"):
        sub = by_cfg.get(cfg)
        if not sub:
            continue
        n = len(sub)
        abst = sum(1 for x in sub if x["pred"] == "ABSTAIN")
        dec = [x for x in sub if x["pred"] != "ABSTAIN"]
        acc_b = sum(1 for x in dec if x["pred"] == x["gold"]) / len(dec) if dec else 0.0
        acc_a = sum(1 for x in sub if x["pred"] == x["gold"]) / n
        model = sub[0]["model"]
        out.append(f"| {cfg} | {model} | {n} | {abst/n:.1%} | {acc_b:.3f} | {acc_a:.3f} |")
    out.append("")

    for cfg in ("small", "large", "large_noabst"):
        sub = by_cfg.get(cfg)
        if not sub:
            continue
        out.append(f"\n## Config : {cfg}\n")
        out.append(block(sub, f"{cfg} — GLOBAL"))
        out.append(calib(sub, cfg))
        out.append("\n**Par langue :**\n")
        for lang in ("de", "fr", "it"):
            out.append(block([x for x in sub if x["lang"] == lang], f"{cfg} — {lang}"))

    report_txt = "\n".join(out)
    with open("research/stance_large_bench_metrics.md", "w", encoding="utf-8") as f:
        f.write(report_txt)
    print(report_txt)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    rows = sample(load_items())
    tasks = build_tasks(rows)
    if cmd == "plan":
        lc = Counter(r["lang"] for r in rows)
        gc = Counter(r["gold"] for r in rows)
        print(f"Échantillon : {len(rows)} contributions, {len(tasks)} batches/config.")
        print(f"  langues : {dict(lc)}")
        print(f"  gold    : {dict(gc)}")
        print(f"  appels estimés : {len(tasks)}/config (small+large = {2*len(tasks)}, "
              f"+large_noabst = {3*len(tasks)}).")
        return
    if cmd == "run":
        configs_run = sys.argv[2:] or ["small", "large"]
        bad = [c for c in configs_run if c not in CONFIGS]
        if bad:
            print(f"config(s) inconnue(s): {bad} — dispo: {list(CONFIGS)}"); sys.exit(1)
        all_new = []
        for cfg in configs_run:
            all_new.extend(run_config(cfg, rows, tasks))
        append_raw(all_new, set(configs_run))
        return
    if cmd == "report":
        report()
        return
    print(f"commande inconnue: {cmd}"); sys.exit(1)


if __name__ == "__main__":
    main()

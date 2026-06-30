#!/usr/bin/env python3
"""Analyse research/stance_validation_raw.jsonl → métriques contre gold FAVOR/AGAINST.

Accuracy globale (2 conventions : abstention=erreur vs abstention exclue),
précision/rappel/F1 par classe, matrice de confusion 3x2, ventilation par langue,
par confiance, taux d'abstention. Imprime un rapport Markdown sur stdout.
"""
import json
from collections import Counter, defaultdict

RAW = "research/stance_validation_raw.jsonl"


def load():
    return [json.loads(l) for l in open(RAW, encoding="utf-8")]


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
    # Convention A : abstention comptée comme erreur (pred != gold).
    correct_a = sum(1 for x in rows if x["pred"] == x["gold"])
    acc_a = correct_a / n
    # Convention B : abstention exclue, accuracy sur les décidés.
    correct_b = sum(1 for x in decided if x["pred"] == x["gold"])
    acc_b = correct_b / len(decided) if decided else 0.0
    # P/R/F par classe (FAVOR / AGAINST), abstention = ni l'un ni l'autre.
    lines = [f"### {label}  (n={n}, abstention={abst} = {abst/n:.1%})",
             f"- **Accuracy** (abstention=erreur) : **{acc_a:.3f}**  ({correct_a}/{n})",
             f"- Accuracy sur décidés (abstention exclue) : {acc_b:.3f}  ({correct_b}/{len(decided)})",
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


def confusion(rows, label):
    cm = Counter((x["gold"], x["pred"]) for x in rows)
    preds = ["FAVOR", "AGAINST", "ABSTAIN"]
    out = [f"### Matrice de confusion — {label}", "",
           "gold ↓ / pred → | FAVOR | AGAINST | ABSTAIN | total |",
           "|---|---|---|---|---|"]
    for g in ("FAVOR", "AGAINST"):
        cells = [cm.get((g, p), 0) for p in preds]
        out.append(f"| **{g}** | {cells[0]} | {cells[1]} | {cells[2]} | {sum(cells)} |")
    out.append("")
    return "\n".join(out)


def main():
    rows = load()
    out = ["# Métriques de validation stance (calculées)\n"]
    out.append(block(rows, "GLOBAL"))
    out.append(confusion(rows, "GLOBAL"))

    out.append("\n## Par langue\n")
    for lang in ("de", "fr", "it"):
        out.append(block([x for x in rows if x["lang"] == lang], f"langue={lang}"))

    out.append("\n## Par confiance auto-déclarée\n")
    for conf in ("high", "medium", "low"):
        sub = [x for x in rows if x["confidence"] == conf]
        out.append(block(sub, f"confidence={conf}"))

    # Taux d'abstention par confiance + accuracy des décidés par confiance.
    out.append("\n## Synthèse confiance (les low se trompent-ils plus ?)\n")
    out.append("| confiance | n | %abstention | accuracy décidés |")
    out.append("|---|---|---|---|")
    for conf in ("high", "medium", "low"):
        sub = [x for x in rows if x["confidence"] == conf]
        if not sub:
            continue
        abst = sum(1 for x in sub if x["pred"] == "ABSTAIN")
        dec = [x for x in sub if x["pred"] != "ABSTAIN"]
        acc = sum(1 for x in dec if x["pred"] == x["gold"]) / len(dec) if dec else 0.0
        out.append(f"| {conf} | {len(sub)} | {abst/len(sub):.1%} | {acc:.3f} |")
    out.append("")

    report = "\n".join(out)
    print(report)
    with open("research/stance_validation_metrics.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()

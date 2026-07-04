"""DÉPOUILLEMENT (APRÈS le vote aveugle) — mappe chaque décision X/Y/tie du panel sur la
variante gagnante (a/b/c) via la clé, puis agrège les 3 duels a-b, a-c, b-c.

Lit `stance_target_ab_panel_votes.jsonl` (aveugle) + `stance_target_ab_panel_key.json`.
N'appelle AUCUN LLM. Sortie console = chiffres pour le verdict.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
VOTES = RESEARCH_DIR / "stance_target_ab_panel_votes.jsonl"
KEY = RESEARCH_DIR / "stance_target_ab_panel_key.json"

votes = [json.loads(l) for l in VOTES.read_text(encoding="utf-8").splitlines() if l.strip()]
key = json.loads(KEY.read_text(encoding="utf-8"))["key"]

# duel -> {winner_variant: count}, plus ties
duel = defaultdict(lambda: {"a": 0, "b": 0, "c": 0, "tie": 0})
per_pair = []
for rec in votes:
    pid = rec["pair_id"]
    k = key[pid]
    xv, yv = k["X"], k["Y"]
    duel_name = "".join(sorted([xv, yv]))  # 'ab' / 'ac' / 'bc'
    dec = rec["decision"]
    if dec == "X":
        winner = xv
    elif dec == "Y":
        winner = yv
    else:
        winner = "tie"
    duel[duel_name][winner] += 1
    per_pair.append((duel_name, k["theme_id"], k["title"], xv, yv, rec["votes"], dec, winner,
                     rec["option_X"], rec["option_Y"]))

print("=== DÉTAIL PAR PAIRE (variante gagnante après dé-anonymisation) ===")
for dn, tid, title, xv, yv, v, dec, win, ox, oy in sorted(per_pair):
    print(f"  [{dn}] {tid:<6} {title[:34]:<34} X={xv} Y={yv} votes={v} → {dec:<3} gagnant={win}")

print("\n=== DUELS (majorité de paire) ===")
for dn in ("ab", "ac", "bc"):
    d = duel[dn]
    v1, v2 = dn[0], dn[1]
    print(f"  {v1} vs {v2}:  {v1}={d[v1]}   {v2}={d[v2]}   nul/tie={d['tie']}   (total {sum(d.values())})")

# Score net de chaque variante contre (a) = référence.
print("\n=== VS ACTUEL (a) ===")
ab, ac = duel["ab"], duel["ac"]
print(f"  b vs a : b gagne {ab['b']}, a gagne {ab['a']}, nul {ab['tie']}")
print(f"  c vs a : c gagne {ac['c']}, a gagne {ac['a']}, nul {ac['tie']}")
bc = duel["bc"]
print(f"  b vs c : b gagne {bc['b']}, c gagne {bc['c']}, nul {bc['tie']}")

# Score global de Copeland : victoires - défaites par variante, tous duels.
score = defaultdict(int)
for dn in ("ab", "ac", "bc"):
    d = duel[dn]
    v1, v2 = dn[0], dn[1]
    score[v1] += d[v1] - d[v2]
    score[v2] += d[v2] - d[v1]
print("\n=== SCORE NET (victoires - défaites, tous duels confondus) ===")
for v in ("a", "b", "c"):
    print(f"  ({v}) net = {score[v]:+d}")

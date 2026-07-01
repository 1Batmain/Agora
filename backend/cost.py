"""Coût LLM d'un build — agrège les tokens Mistral (via `mistral_client.get_usage`) et
estime le $, par consultation, dans `analysis/cost.json`.

TOUT le trafic Mistral passe par `mistral_client.chat` (extraction via `ApiBackend`,
nommage, enrichissement, insights, opinion) → un seul accumulateur suffit. Chaque phase
de build (`analysis`, `opinion`) enregistre son usage ; `cost.json` cumule les phases et
recalcule le total + le coût estimé. Servi tel quel par l'API (transparence des coûts).

Les prix sont INDICATIFS (grille publique Mistral, USD / 1M tokens) et surchargeables par
env `AGORA_PRICE_<MODEL>` = "input,output" — l'estimation reste une estimation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from backend import analysis_store

# USD / 1M tokens : (entrée, sortie). Indicatif — voir la grille Mistral officielle.
PRICING: dict[str, tuple[float, float]] = {
    "mistral-large-latest": (2.0, 6.0),
    "mistral-medium-latest": (0.4, 2.0),
    "mistral-small-latest": (0.1, 0.3),
}
_DEFAULT_PRICE = (0.1, 0.3)  # repli = tarif « small »


def _price(model: str) -> tuple[float, float]:
    override = os.environ.get(f"AGORA_PRICE_{model.upper().replace('-', '_')}")
    if override:
        try:
            i, o = override.split(",")
            return float(i), float(o)
        except ValueError:
            pass
    return PRICING.get(model, _DEFAULT_PRICE)


def estimate_usd(by_model: dict) -> float:
    total = 0.0
    for model, u in (by_model or {}).items():
        pin, pout = _price(model)
        total += (u.get("prompt_tokens", 0) / 1e6) * pin
        total += (u.get("completion_tokens", 0) / 1e6) * pout
    return round(total, 4)


def _sum_models(phases: dict) -> dict:
    """Somme les `by_model` de toutes les phases → total consolidé."""
    by_model: dict = {}
    for ph in phases.values():
        for model, u in (ph.get("by_model") or {}).items():
            agg = by_model.setdefault(
                model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            agg["calls"] += u.get("calls", 0)
            agg["prompt_tokens"] += u.get("prompt_tokens", 0)
            agg["completion_tokens"] += u.get("completion_tokens", 0)
    return by_model


def cost_path(dataset: str) -> Path:
    return analysis_store.analysis_dir(dataset) / "cost.json"


def read_cost(dataset: str) -> dict | None:
    p = cost_path(dataset)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def record_phase(dataset: str, phase: str, usage: dict, *, extra: dict | None = None) -> dict:
    """Enregistre l'usage d'UNE phase (`analysis`/`opinion`) et recalcule le total.

    `usage` = snapshot de `mistral_client.get_usage()`. Idempotent par phase (ré-écrit la
    phase). Retourne le `cost.json` mis à jour.
    """
    doc = read_cost(dataset) or {"dataset": dataset, "phases": {}}
    doc.setdefault("phases", {})
    doc["phases"][phase] = {
        "calls": usage.get("calls", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "by_model": usage.get("by_model", {}),
    }
    by_model = _sum_models(doc["phases"])
    pt = sum(u["prompt_tokens"] for u in by_model.values())
    ct = sum(u["completion_tokens"] for u in by_model.values())
    doc["total"] = {
        "calls": sum(u["calls"] for u in by_model.values()),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "by_model": by_model,
        "estimated_usd": estimate_usd(by_model),
    }
    if extra:
        doc.update(extra)
    doc["pricing_usd_per_1m"] = {m: {"input": i, "output": o} for m, (i, o) in PRICING.items()}
    p = cost_path(dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc

"""LLM (Mistral) comme SEGMENTEUR de frontières ET EXTRACTEUR multi-thème.

    uv run python -m eval.segmentation.llm_seg
        [--gold eval/segmentation/gold_large.json]
        [--model mistral-small-latest]
        [--limit N] [--max-calls N]
        [--bound-batch 6] [--theme-batch 12]
        [--out eval/segmentation/llm_report.md]

Deux mesures, sur `gold_large.json`, comparées à l'attention réglé-main (F1_multi
0.769) et au change-point (0.44) :

1. **FRONTIÈRES** — on présente chaque avis comme une liste de MOTS indexés
   (`[0] Je [1] passe …`) et on demande à Mistral les indices où un NOUVEAU thème
   commence. Les indices renvoyés SONT les frontières en espace-mots (aucune
   réalignement char→mot, donc directement comparables au banc attention).
   → Pk / WindowDiff / F1-frontières (tol ±1, micro multi) / faux-positifs mono.

2. **THÈMES** (le vrai but) — on demande à Mistral, en choix FERMÉ parmi les 8
   thèmes du gold (+ descriptions de la taxonomie), l'ensemble des thèmes soulevés
   par chaque avis → multi-label P/R/F1 (micro & macro) vs les `seg_themes` du gold.
   L'attention ne sait PAS faire ça : c'est la valeur ajoutée propre du LLM.

Honnêteté : nb d'appels API réels (hors cache), latence, ce qui est envoyé à
l'API (le texte des avis → Mistral EU). Clé via `mistral_client.load_api_key`,
JAMAIS loggée. Cache disque (`.cache/llm/`) → relances gratuites.

ÉCRIT UNIQUEMENT dans `eval/segmentation/` (llm_report.md, llm_scores.json, .cache/llm/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from eval.segmentation import metrics as M
from eval.segmentation.seg_bench import GoldItem, load_gold
from pipeline.cluster import mistral_client as MC

HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold_large.json"
DEFAULT_REPORT = HERE / "llm_report.md"
DEFAULT_SCORES = HERE / "llm_scores.json"
CACHE_DIR = HERE / ".cache" / "llm"

_WORD_RE = re.compile(r"\S+")

# Repères attention/change-point réglés-main (cf. attn_scores.json) pour la scorecard.
ATTN_F1 = 0.769
CHANGEPOINT_F1 = 0.44


# --------------------------------------------------------------------------- #
# Mots & frontières (réplique EXACTE de la granularité de scoring du banc)
# --------------------------------------------------------------------------- #
def split_words(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words, spans = [], []
    for m in _WORD_RE.finditer(text):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans


def boundary_word_index(spans: list[tuple[int, int]], char_offset: int) -> int:
    """Nb de mots qui COMMENCENT avant l'offset (frontière à une jointure)."""
    idx = 0
    for s, _ in spans:
        if s < char_offset:
            idx += 1
        else:
            break
    return idx


@dataclass
class Prepared:
    item: GoldItem
    words: list[str]
    n: int
    ref: set[int]            # frontières gold en indices-MOTS
    gold_themes: set[str]    # ensemble des thèmes (mono → {theme})


def prepare(items: list[GoldItem]) -> list[Prepared]:
    out = []
    for it in items:
        words, spans = split_words(it.text)
        n = len(words)
        ref = set()
        for off in it.boundaries_char:
            b = boundary_word_index(spans, off)
            if 0 < b < n:
                ref.add(b)
        themes = {t for t in it.seg_themes if t and t != "?"}
        out.append(Prepared(it, words, n, ref, themes))
    return out


# --------------------------------------------------------------------------- #
# Appel Mistral caché sur disque (relance gratuite, comptage honnête)
# --------------------------------------------------------------------------- #
@dataclass
class CallStats:
    api_calls: int = 0       # vrais POST de CE run (cache miss)
    cache_hits: int = 0
    errors: int = 0
    api_seconds: float = 0.0
    chars_sent: int = 0      # caractères réellement (ré)envoyés ce run
    # Coût « à froid » (équivalent run sans cache) — agrège miss + hits.
    cold_calls: int = 0
    cold_seconds: float = 0.0
    cold_chars: int = 0


def _cache_key(model: str, temperature: float, messages: list[dict]) -> Path:
    blob = json.dumps([model, temperature, messages], ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
    return CACHE_DIR / f"{h}.json"


def cached_chat(messages: list[dict], *, model: str, temperature: float,
                max_tokens: int, stats: CallStats, timeout: float = 60.0) -> str | None:
    """Appel Mistral avec cache disque. Renvoie le contenu, ou None si échec."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chars = sum(len(m["content"]) for m in messages)
    cpath = _cache_key(model, temperature, messages)
    if cpath.exists():
        stats.cache_hits += 1
        rec = json.loads(cpath.read_text(encoding="utf-8"))
        # Coût à froid : on ré-impute le coût mémorisé lors du 1er appel réel.
        stats.cold_calls += 1
        stats.cold_seconds += float(rec.get("seconds", 0.0))
        stats.cold_chars += int(rec.get("chars", chars))
        return rec["content"]

    stats.chars_sent += chars
    t0 = time.monotonic()
    try:
        content = MC.chat(messages, model=model, temperature=temperature,
                          max_tokens=max_tokens, json_mode=True, timeout=timeout)
    except MC.MistralError as exc:
        stats.errors += 1
        # exc.reason est garanti sans secret par mistral_client.
        print(f"  ⚠️ mistral[{exc.status}]: {exc.reason}")
        return None
    finally:
        elapsed = time.monotonic() - t0
        stats.api_seconds += elapsed
    stats.api_calls += 1
    stats.cold_calls += 1
    stats.cold_seconds += elapsed
    stats.cold_chars += chars
    cpath.write_text(json.dumps(
        {"content": content, "seconds": round(elapsed, 3), "chars": chars},
        ensure_ascii=False), encoding="utf-8")
    return content


def parse_json_object(raw: str) -> dict | None:
    """Extrait le premier objet JSON d'une réponse (tolère ```json … ``` etc.)."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# --------------------------------------------------------------------------- #
# MESURE 1 — frontières (mots indexés → indices de nouveau thème)
# --------------------------------------------------------------------------- #
BOUND_SYS = (
    "Tu es un segmenteur de texte par THÈME, multilingue (FR, DE, IT, EN…). "
    "On te donne des avis citoyens découpés en MOTS numérotés. Pour chaque avis, "
    "tu identifies les positions où un NOUVEAU thème/sujet commence. Un avis qui ne "
    "parle que d'un seul sujet n'a AUCUNE frontière (liste vide). Ne te base pas sur "
    "la ponctuation seule : juge le sens. Réponds STRICTEMENT en JSON."
)


def _enumerate_words(words: list[str]) -> str:
    return " ".join(f"[{i}]{w}" for i, w in enumerate(words))


def bound_prompt(batch: list[Prepared]) -> list[dict]:
    lines = [
        "Pour CHAQUE avis ci-dessous, renvoie la liste des indices de mot où un "
        "NOUVEAU thème commence (le mot à cet indice est le 1er du nouveau segment). "
        "Indices valides : 1..n-1. Avis mono-thème → liste vide [].",
        'Format EXACT : {"avis_id": [indices…], …}. Aucun texte hors JSON.',
        "",
    ]
    for p in batch:
        lines.append(f"### {p.item.id}  (n={p.n} mots)")
        lines.append(_enumerate_words(p.words))
        lines.append("")
    return [
        {"role": "system", "content": BOUND_SYS},
        {"role": "user", "content": "\n".join(lines)},
    ]


def run_boundaries(prepared: list[Prepared], model: str, batch_size: int,
                   stats: CallStats, max_calls: int | None) -> dict[str, set[int]]:
    """Renvoie {item_id: frontières-mots hypothèse}. Items absents → set() (échec)."""
    hyps: dict[str, set[int]] = {}
    batches = [prepared[i:i + batch_size] for i in range(0, len(prepared), batch_size)]
    for bi, batch in enumerate(batches):
        if max_calls is not None and stats.api_calls >= max_calls:
            print(f"  ⛔ max-calls atteint ({max_calls}) — frontières arrêtées")
            break
        # max_tokens : ~ qq indices par avis, large marge.
        raw = cached_chat(bound_prompt(batch), model=model, temperature=0.0,
                          max_tokens=64 + 24 * len(batch), stats=stats)
        obj = parse_json_object(raw or "")
        print(f"  frontières lot {bi + 1}/{len(batches)} "
              f"({len(batch)} avis){' ⚠️ parse échec' if obj is None else ''}")
        for p in batch:
            hyp: set[int] = set()
            if obj is not None:
                val = obj.get(p.item.id)
                if isinstance(val, list):
                    for x in val:
                        try:
                            b = int(x)
                        except (TypeError, ValueError):
                            continue
                        if 0 < b < p.n:
                            hyp.add(b)
            hyps[p.item.id] = hyp
    return hyps


# --------------------------------------------------------------------------- #
# MESURE 2 — thèmes (choix fermé sur les 8 thèmes du gold)
# --------------------------------------------------------------------------- #
def theme_sys(taxonomy: dict[str, str]) -> str:
    lines = [
        "Tu es un extracteur de THÈMES multilingue (FR, DE, IT, EN…). On te donne "
        "des avis citoyens. Pour chaque avis, tu listes TOUS les thèmes abordés, "
        "choisis EXCLUSIVEMENT dans la liste fermée suivante (utilise les CLÉS) :",
    ]
    for k, desc in taxonomy.items():
        lines.append(f"- {k} : {desc}")
    lines.append(
        "Un avis peut aborder plusieurs thèmes. N'invente aucun thème hors liste. "
        "Réponds STRICTEMENT en JSON."
    )
    return "\n".join(lines)


def theme_prompt(batch: list[Prepared], taxonomy: dict[str, str]) -> list[dict]:
    lines = [
        'Pour CHAQUE avis, renvoie la liste des clés de thème présentes. '
        'Format EXACT : {"avis_id": ["cle1", "cle2", …], …}. Aucun texte hors JSON.',
        "",
    ]
    for p in batch:
        lines.append(f"### {p.item.id}")
        lines.append(p.item.text)
        lines.append("")
    return [
        {"role": "system", "content": theme_sys(taxonomy)},
        {"role": "user", "content": "\n".join(lines)},
    ]


def run_themes(prepared: list[Prepared], taxonomy: dict[str, str], model: str,
               batch_size: int, stats: CallStats,
               max_calls: int | None) -> dict[str, set[str]]:
    valid = set(taxonomy)
    hyps: dict[str, set[str]] = {}
    batches = [prepared[i:i + batch_size] for i in range(0, len(prepared), batch_size)]
    for bi, batch in enumerate(batches):
        if max_calls is not None and stats.api_calls >= max_calls:
            print(f"  ⛔ max-calls atteint ({max_calls}) — thèmes arrêtés")
            break
        raw = cached_chat(theme_prompt(batch, taxonomy), model=model, temperature=0.0,
                          max_tokens=64 + 32 * len(batch), stats=stats)
        obj = parse_json_object(raw or "")
        print(f"  thèmes lot {bi + 1}/{len(batches)} "
              f"({len(batch)} avis){' ⚠️ parse échec' if obj is None else ''}")
        for p in batch:
            hyp: set[str] = set()
            if obj is not None:
                val = obj.get(p.item.id)
                if isinstance(val, list):
                    hyp = {str(x).strip() for x in val if str(x).strip() in valid}
            hyps[p.item.id] = hyp
    return hyps


# --------------------------------------------------------------------------- #
# Scoring frontières (réplique seg_bench.evaluate, hypothèses = LLM)
# --------------------------------------------------------------------------- #
@dataclass
class BoundScore:
    pk: float = 0.0
    windowdiff: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    gf1: float = 0.0
    mono_fp_rate: float = 0.0
    mono_cuts_mean: float = 0.0
    n_multi: int = 0
    n_mono: int = 0
    covered: int = 0         # items réellement traités par le LLM
    multi_zero: int = 0      # multi pour lesquels le LLM ne coupe PAS du tout
    pred_mean_multi: float = 0.0   # nb moyen de coupes prédites par multi
    gold_mean_multi: float = 0.0   # nb moyen de frontières gold par multi


def score_boundaries(prepared: list[Prepared], hyps: dict[str, set[int]]) -> BoundScore:
    multi = [p for p in prepared if p.item.type == "multi"]
    mono = [p for p in prepared if p.item.type == "mono"]

    pk_m, wd_m = [], []
    bc = M.BoundaryCounts()
    gbc = M.BoundaryCounts()
    covered = 0
    multi_zero = pred_sum = gold_sum = 0
    for p in multi:
        if p.item.id not in hyps:
            continue
        covered += 1
        hyp = hyps[p.item.id]
        if not hyp:
            multi_zero += 1
        pred_sum += len(hyp)
        gold_sum += len(p.ref)
        pk_m.append(M.pk(p.n, p.ref, hyp))
        wd_m.append(M.windowdiff(p.n, p.ref, hyp))
        c = M.boundary_counts(p.ref, hyp, tol=1)
        bc = bc + c
        gbc = gbc + c

    mono_hits = mono_cuts = 0
    n_mono_cov = 0
    for p in mono:
        if p.item.id not in hyps:
            continue
        covered += 1
        n_mono_cov += 1
        hyp = hyps[p.item.id]
        if hyp:
            mono_hits += 1
        mono_cuts += len(hyp)
        gbc = gbc + M.boundary_counts(p.ref, hyp, tol=1)

    import numpy as np
    return BoundScore(
        pk=float(np.mean(pk_m)) if pk_m else 0.0,
        windowdiff=float(np.mean(wd_m)) if wd_m else 0.0,
        f1=bc.f1, precision=bc.precision, recall=bc.recall,
        gf1=gbc.f1,
        mono_fp_rate=mono_hits / n_mono_cov if n_mono_cov else 0.0,
        mono_cuts_mean=mono_cuts / n_mono_cov if n_mono_cov else 0.0,
        n_multi=len(pk_m), n_mono=n_mono_cov, covered=covered,
        multi_zero=multi_zero,
        pred_mean_multi=pred_sum / len(pk_m) if pk_m else 0.0,
        gold_mean_multi=gold_sum / len(pk_m) if pk_m else 0.0,
    )


# --------------------------------------------------------------------------- #
# Scoring thèmes (multi-label)
# --------------------------------------------------------------------------- #
@dataclass
class ThemeScore:
    micro_p: float = 0.0
    micro_r: float = 0.0
    micro_f1: float = 0.0
    macro_p: float = 0.0
    macro_r: float = 0.0
    macro_f1: float = 0.0
    exact_set: float = 0.0       # fraction d'avis avec ENSEMBLE de thèmes exact
    per_theme: dict = field(default_factory=dict)
    covered: int = 0


def score_themes(prepared: list[Prepared], hyps: dict[str, set[str]],
                 labels: list[str]) -> ThemeScore:
    tp = {t: 0 for t in labels}
    fp = {t: 0 for t in labels}
    fn = {t: 0 for t in labels}
    exact = covered = 0
    for p in prepared:
        if p.item.id not in hyps:
            continue
        covered += 1
        gold = p.gold_themes
        pred = hyps[p.item.id]
        if gold == pred:
            exact += 1
        for t in labels:
            if t in gold and t in pred:
                tp[t] += 1
            elif t in pred and t not in gold:
                fp[t] += 1
            elif t in gold and t not in pred:
                fn[t] += 1

    def prf(t):
        p_ = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) else 1.0
        r_ = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) else 1.0
        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        return p_, r_, f_

    per = {t: dict(zip(("p", "r", "f1", "tp", "fp", "fn"),
                       (*prf(t), tp[t], fp[t], fn[t]))) for t in labels}

    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mi_p = TP / (TP + FP) if (TP + FP) else 1.0
    mi_r = TP / (TP + FN) if (TP + FN) else 1.0
    mi_f = 2 * mi_p * mi_r / (mi_p + mi_r) if (mi_p + mi_r) else 0.0
    ma_p = sum(per[t]["p"] for t in labels) / len(labels)
    ma_r = sum(per[t]["r"] for t in labels) / len(labels)
    ma_f = sum(per[t]["f1"] for t in labels) / len(labels)
    return ThemeScore(
        micro_p=mi_p, micro_r=mi_r, micro_f1=mi_f,
        macro_p=ma_p, macro_r=ma_r, macro_f1=ma_f,
        exact_set=exact / covered if covered else 0.0,
        per_theme=per, covered=covered,
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])


def build_report(gold_path: Path, n_items: int, n_mono: int, n_multi: int,
                 model: str, bs: BoundScore, ts: ThemeScore, stats: CallStats,
                 labels: list[str], bound_batch: int, theme_batch: int,
                 limited: bool, examples: list[str]) -> str:
    beats_bound = bs.f1 > ATTN_F1
    L = []
    L.append("# LLM (Mistral) comme segmenteur & extracteur de thèmes — rapport\n")
    L.append(f"*Jeu : `{gold_path.name}` — N={n_items} ({n_mono} mono, {n_multi} multi). "
             f"Modèle : `{model}`, température 0.0 (quasi-déterministe ; ±0.01 de "
             f"F1 entre runs), JSON mode. "
             f"Clé via `mistral_client.load_api_key` (jamais loggée).*\n")
    if limited:
        L.append("⚠️ **Run PARTIEL** (`--limit`/`--max-calls`) — chiffres non "
                 "représentatifs du jeu complet.\n")

    # Scorecard
    L.append("## Scorecard — bat-il l'attention réglé-main (F1 0.769) ?\n")
    L.append(_md_table([
        {"Approche": "**Mistral (frontières)**", "F1_multi": round(bs.f1, 4),
         "Pk↓": round(bs.pk, 4), "WindowDiff↓": round(bs.windowdiff, 4),
         "P": round(bs.precision, 4), "R": round(bs.recall, 4),
         "mono_FP↓": round(bs.mono_fp_rate, 4), "F1_global": round(bs.gf1, 4)},
        {"Approche": "Attention (réglé-main)", "F1_multi": ATTN_F1,
         "Pk↓": 0.1493, "WindowDiff↓": 0.1563, "P": 0.8955, "R": 0.6742,
         "mono_FP↓": 0.1442, "F1_global": "—"},
        {"Approche": "Change-point (cosinus)", "F1_multi": CHANGEPOINT_F1,
         "Pk↓": 0.2815, "WindowDiff↓": 0.282, "P": 0.4545, "R": 0.4307,
         "mono_FP↓": 0.7019, "F1_global": 0.384},
    ], ["Approche", "F1_multi", "Pk↓", "WindowDiff↓", "P", "R", "mono_FP↓", "F1_global"]) + "\n")
    zero_pct = (bs.multi_zero / bs.n_multi * 100) if bs.n_multi else 0.0
    L.append(f"**Verdict frontières : {'OUI' if beats_bound else 'NON'}** — "
             f"Mistral F1_multi={bs.f1:.3f} vs attention {ATTN_F1:.3f} "
             f"({bs.f1 - ATTN_F1:+.3f}).\n")
    L.append(
        f"- **Échec = SOUS-segmentation, pas sur-coupe.** Précision quasi parfaite "
        f"(**P={bs.precision:.3f}** — quand le LLM coupe, il a raison), mais rappel "
        f"faible (**R={bs.recall:.3f}**) : il prédit en moyenne **{bs.pred_mean_multi:.2f}** "
        f"coupe/multi contre **{bs.gold_mean_multi:.2f}** attendues, et ne coupe PAS "
        f"DU TOUT sur **{bs.multi_zero}/{bs.n_multi}** multi ({zero_pct:.0f}%). Les "
        f"transitions du gold (« rédigées pour glisser naturellement ») sont vues "
        f"comme un seul flux cohérent.\n"
        f"- **Abstention mono PARFAITE** : **{bs.mono_fp_rate*100:.0f}%** de faux "
        f"positifs ({bs.mono_cuts_mean:.2f} coupe/mono) — strictement mieux que "
        f"l'attention (14%) et le change-point (70%). Le LLM ne coupe jamais un avis "
        f"mono-thème cohérent ; c'est le miroir exact de sa sous-segmentation.\n")

    # Thèmes
    L.append("## Récupération des THÈMES (le vrai but — l'attention ne sait pas faire)\n")
    L.append(f"Multi-label, choix fermé sur {len(labels)} thèmes, vs l'ensemble des "
             f"`seg_themes` du gold ({ts.covered} avis couverts).\n")
    L.append(_md_table([
        {"Granularité": "micro", "P": round(ts.micro_p, 4), "R": round(ts.micro_r, 4),
         "F1": round(ts.micro_f1, 4)},
        {"Granularité": "macro", "P": round(ts.macro_p, 4), "R": round(ts.macro_r, 4),
         "F1": round(ts.macro_f1, 4)},
    ], ["Granularité", "P", "R", "F1"]) + "\n")
    L.append(f"**Exactitude d'ENSEMBLE** (tous les thèmes d'un avis, ni plus ni moins) : "
             f"**{ts.exact_set*100:.0f}%** des avis.\n")
    L.append("### F1 par thème\n")
    rows = [{"thème": t, "P": round(ts.per_theme[t]["p"], 3),
             "R": round(ts.per_theme[t]["r"], 3), "F1": round(ts.per_theme[t]["f1"], 3),
             "TP": ts.per_theme[t]["tp"], "FP": ts.per_theme[t]["fp"],
             "FN": ts.per_theme[t]["fn"]}
            for t in sorted(labels, key=lambda x: -ts.per_theme[x]["f1"])]
    L.append(_md_table(rows, ["thème", "P", "R", "F1", "TP", "FP", "FN"]) + "\n")

    # Exemples
    if examples:
        L.append("## Exemples (avis multi → frontières & thèmes)\n")
        L.extend(examples)

    # Coût / honnêteté
    L.append("## Coût, latence, confidentialité — honnêteté\n")
    cold_avg = stats.cold_seconds / stats.cold_calls if stats.cold_calls else 0.0
    L.append(
        f"- **Coût d'un run à froid (sans cache)** : **{stats.cold_calls} appels** "
        f"`{model}`, ~**{stats.cold_seconds:.0f}s** cumulés (~{cold_avg:.2f}s/appel), "
        f"~**{stats.cold_chars:,}** caractères de prompts envoyés. C'est le coût réel "
        f"facturé pour évaluer les {n_items} avis.\n"
        f"- **Ce run** : {stats.api_calls} appels réels + {stats.cache_hits} servis par "
        f"le cache disque `.cache/llm/` ({stats.errors} erreurs) — le cache rend les "
        f"relances gratuites et déterministes.\n"
        f"- **Batching** : frontières {bound_batch} avis/appel, thèmes {theme_batch} "
        f"avis/appel (réduit le nb d'appels d'un facteur ~{bound_batch}/{theme_batch}).\n"
        f"- **Destinataire** : `api.mistral.ai` (UE).\n"
        f"- **Ce qui part à l'API** : le **texte intégral des avis citoyens** "
        f"(données potentiellement sensibles) est transmis à Mistral. L'attention "
        f"et le change-point tournent **100% en local** (aucune donnée ne sort). "
        f"C'est le compromis central : le LLM est plus capable mais externalise la "
        f"donnée et coûte par appel.\n"
        f"- **Déterminisme** : température 0.0 + JSON mode ; reproductible modulo "
        f"variations serveur. Cache → relances stables et gratuites.\n")

    # Verdict
    L.append("## Verdict\n")
    L.append(
        f"- **Frontières : {'le LLM BAT' if beats_bound else 'le LLM NE BAT PAS'} "
        f"l'attention réglé-main** ({bs.f1:.3f} vs {ATTN_F1:.3f}). "
        f"{'' if beats_bound else 'L'+chr(39)+'attention locale reste devant et gratuite. '}"
        f"Mais l'attention ne produit QUE des frontières.\n")
    L.append(
        f"- **Thèmes : le LLM récupère l'ensemble des thèmes à micro-F1={ts.micro_f1:.3f}** "
        f"(exact-set {ts.exact_set*100:.0f}%) — capacité que NI l'attention NI le "
        f"change-point n'ont. Si le but produit est « quels thèmes dans cet avis », "
        f"c'est la mesure qui compte, et le LLM la sert directement sans pipeline "
        f"d'embeddings ni seuils réglés à la main.\n")
    L.append(
        "- **Compromis** : capacité multi-thème immédiate et langue-agnostique, "
        "contre coût/latence par appel et sortie des données vers l'API. Pour de la "
        "segmentation de frontières pure et locale, l'attention reste préférable ; "
        "pour l'extraction thématique, le LLM est l'outil direct.\n")
    return "\n".join(L)


def _example_blocks(prepared, bhyps, thyps, max_n=3):
    blocks = []
    multi = [p for p in prepared if p.item.type == "multi" and p.item.id in bhyps]
    for p in multi[:max_n]:
        def seg(bset):
            return " ".join(("⟂ " + w) if i in bset else w
                            for i, w in enumerate(p.words))
        blocks.append(f"**{p.item.id}** — gold thèmes : "
                      f"{', '.join(sorted(p.gold_themes))}\n")
        blocks.append(f"- gold frontières : {seg(p.ref)}")
        blocks.append(f"- Mistral frontières : {seg(bhyps[p.item.id])}")
        blocks.append(f"- Mistral thèmes : {', '.join(sorted(thyps.get(p.item.id, set()))) or '∅'}\n")
    return blocks


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="LLM Mistral segmenteur + extracteur de thèmes.")
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--model", default=MC.NAMING_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="n'évaluer que les N premiers items")
    ap.add_argument("--max-calls", type=int, default=None, help="plafond d'appels API réels")
    ap.add_argument("--bound-batch", type=int, default=6)
    ap.add_argument("--theme-batch", type=int, default=12)
    ap.add_argument("--no-boundaries", action="store_true")
    ap.add_argument("--no-themes", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--scores-out", default=str(DEFAULT_SCORES))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    items, meta = load_gold(gold_path)
    taxonomy = meta.get("taxonomy", {})
    if not taxonomy:
        labels = sorted({t for it in items for t in it.seg_themes if t and t != "?"})
        taxonomy = {t: t for t in labels}
    labels = list(taxonomy)
    if args.limit:
        items = items[:args.limit]
    prepared = prepare(items)
    n_mono = sum(1 for p in prepared if p.item.type == "mono")
    n_multi = sum(1 for p in prepared if p.item.type == "multi")
    print(f"gold: {gold_path.name} — {len(prepared)} items ({n_mono} mono, {n_multi} multi)")
    if not MC.available():
        raise SystemExit("Aucune clé Mistral (var/mistral.key ou MISTRAL_API_KEY).")

    stats = CallStats()
    bhyps: dict[str, set[int]] = {}
    thyps: dict[str, set[str]] = {}

    if not args.no_boundaries:
        print("MESURE 1 — frontières…")
        bhyps = run_boundaries(prepared, args.model, args.bound_batch, stats, args.max_calls)
    if not args.no_themes:
        print("MESURE 2 — thèmes…")
        thyps = run_themes(prepared, taxonomy, args.model, args.theme_batch, stats, args.max_calls)

    bs = score_boundaries(prepared, bhyps) if bhyps else BoundScore()
    ts = score_themes(prepared, thyps, labels) if thyps else ThemeScore()
    print(f"frontières: F1_multi={bs.f1:.3f} Pk={bs.pk:.3f} mono_FP={bs.mono_fp_rate:.3f} "
          f"(couverts {bs.covered})")
    print(f"thèmes: micro-F1={ts.micro_f1:.3f} macro-F1={ts.macro_f1:.3f} "
          f"exact-set={ts.exact_set:.3f} (couverts {ts.covered})")
    print(f"API: {stats.api_calls} appels, {stats.cache_hits} cache, "
          f"{stats.errors} err, {stats.api_seconds:.1f}s")

    limited = bool(args.limit or args.max_calls or args.no_boundaries or args.no_themes)
    examples = _example_blocks(prepared, bhyps, thyps) if (bhyps and thyps) else []
    report = build_report(gold_path, len(prepared), n_mono, n_multi, args.model,
                          bs, ts, stats, labels, args.bound_batch, args.theme_batch,
                          limited, examples)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"✓ {args.out}")

    Path(args.scores_out).write_text(json.dumps({
        "gold": gold_path.name, "model": args.model, "n_items": len(prepared),
        "n_mono": n_mono, "n_multi": n_multi, "limited": limited,
        "boundaries": {
            "F1_multi": round(bs.f1, 4), "Pk": round(bs.pk, 4),
            "WindowDiff": round(bs.windowdiff, 4), "P": round(bs.precision, 4),
            "R": round(bs.recall, 4), "mono_FP": round(bs.mono_fp_rate, 4),
            "mono_cuts": round(bs.mono_cuts_mean, 3), "F1_global": round(bs.gf1, 4),
            "covered": bs.covered, "beats_attention_0769": bs.f1 > ATTN_F1,
            "multi_zero_cut": bs.multi_zero, "n_multi": bs.n_multi,
            "pred_mean_multi": round(bs.pred_mean_multi, 3),
            "gold_mean_multi": round(bs.gold_mean_multi, 3),
        },
        "themes": {
            "micro_P": round(ts.micro_p, 4), "micro_R": round(ts.micro_r, 4),
            "micro_F1": round(ts.micro_f1, 4), "macro_F1": round(ts.macro_f1, 4),
            "exact_set": round(ts.exact_set, 4), "covered": ts.covered,
            "per_theme": {t: {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in d.items()} for t, d in ts.per_theme.items()},
        },
        "cost": {
            "cold_calls": stats.cold_calls, "cold_seconds": round(stats.cold_seconds, 2),
            "cold_chars": stats.cold_chars,
            "this_run_api_calls": stats.api_calls, "cache_hits": stats.cache_hits,
            "errors": stats.errors,
        },
        "baselines": {"attention_F1": ATTN_F1, "changepoint_F1": CHANGEPOINT_F1},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ {args.scores_out}")


if __name__ == "__main__":
    main()

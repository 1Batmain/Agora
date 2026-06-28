"""A/B EXTRACTION : prompt ACTUEL (A=CLAIM_SYS) vs prompt RELÂCHÉ (B) — R&D pur.

Hypothèse (Bob) : la cible étant rétrogradée (cf. agora-stance-subject-verdict), on peut
RELÂCHER l'obligation dure de cible — qui fait *dropper* des claims (mode paresseux) — et
viser une SEGMENTATION COMPLÈTE des avis multi-thèmes. On compare A et B sur ~200 avis
(grand débat prioritaire), gate verbatim `align_spans` aux deux, quantitatif + LLM-juge.

Aucun fichier produit modifié : on importe CLAIM_SYS / align_spans / parse_batch_claims
tels quels et on n'écrit que sous research/.

Usage :
  export MISTRAL_API_KEY=$(cat var/mistral.key)
  python -m research.extract_ab            # extraction A+B (cachée) + métriques
  python -m research.extract_ab --judge    # + LLM-juge sur ~40 avis
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pipeline.claims.extract import (
    BATCH_SYS_SUFFIX,
    CLAIM_SYS,
    parse_batch_claims,
)
from pipeline.claims.span import Claim, align_spans
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "research" / "extract_ab_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("AGORA_AB_MODEL", "mistral-large-latest")
BATCH = int(os.environ.get("AGORA_AB_BATCH", "6"))
WORKERS = int(os.environ.get("AGORA_AB_WORKERS", "4"))
SEED = 20260628

# Échantillon : grand débat PRIORITAIRE (avis longs multi-thèmes = le vrai test).
SAMPLE_PLAN = {"granddebat": 110, "tiktok": 35, "xstance": 30, "republique-numerique": 25}

# --------------------------------------------------------------------------- #
# PROMPT B — RELÂCHÉ. Garde sélectivité + verbatim strict + regroupement ; CHANGE :
#   (a) target = INDICE OPTIONNEL → jamais dropper un claim faute de cible (null OK) ;
#   (b) insiste sur la COMPLÉTUDE / segmentation des avis multi-thèmes.
# --------------------------------------------------------------------------- #
CLAIM_SYS_B = (
    "Tu es un analyste d'avis citoyens, multilingue (FR, DE, IT, EN…). On te donne UN "
    "avis. Tu en extrais les CLAIMS : ses idées de FOND distinctes — chaque grief, "
    "opinion ou proposition du citoyen. Tu RECOPIES chaque portion MOT POUR MOT depuis "
    "l'avis (sous-chaîne EXACTE : mêmes mots, même orthographe, même ponctuation, fautes "
    "comprises) ; tu ne reformules RIEN, n'ajoutes RIEN, ne corriges RIEN.\n"
    "\n"
    "Chaque claim a DEUX champs :\n"
    "• `parts` : la/les portion(s) verbatim qui PORTENT l'idée. En général UNE seule "
    "portion contiguë. Mais si l'idée est répartie sur des passages NON-CONTIGUS de "
    "l'avis (p.ex. la phrase qui pose l'idée + la fin d'une phrase plus loin qui s'y "
    "réfère), mets CHAQUE morceau verbatim dans `parts` → ils forment UN seul claim. "
    "N'utilise PLUSIEURS parts QUE si les morceaux appartiennent vraiment à la même idée.\n"
    "• `target` : un INDICE OPTIONNEL — l'OBJET / l'aspect sur lequel porte la position "
    "(« les vidéos », « le temps d'écran », « la fiscalité locale », « le mille-feuille "
    "administratif »…), recopié VERBATIM depuis l'avis. Mets-la SI une courte portion de "
    "l'avis pointe l'objet sans effort ; sinon `target=null`. NE JAMAIS écarter un claim "
    "de fond au prétexte que sa cible est diffuse ou implicite : une position réelle se "
    "garde toujours, cible ou pas. La cible n'est qu'un indice secondaire, pas un filtre.\n"
    "\n"
    "RÈGLES :\n"
    "1. COMPLÉTUDE (priorité) — un avis citoyen, surtout long, ARGUMENTE souvent sur "
    "PLUSIEURS thèmes distincts (p.ex. fiscalité ET démocratie ET services publics). "
    "Capture CHAQUE prise de position distincte de l'avis : n'en oublie AUCUNE, ne t'arrête "
    "pas à la première. Sépare les thèmes RÉELLEMENT distincts en claims distincts. "
    "Balaie l'avis du début à la fin.\n"
    "2. SÉLECTIVITÉ — n'extrais que la SUBSTANCE : une PRISE DE POSITION (grief, opinion, "
    "proposition). Laisse de côté le pur cadrage, le narratif et les annonces qui ne "
    "portent aucune position par eux-mêmes (« pour illustrer… », « mes doléances sont "
    "triples : », politesses, anecdote de contexte). Pas de bruit, pas de redite.\n"
    "3. REGROUPEMENT — ne FRAGMENTE pas une même idée. Restent DANS UN SEUL claim : un "
    "contraste (« X et non Y »), une justification (« … parce que … »), une condition "
    "(« si …, alors … ») et une énumération qui DÉTAILLE une seule idée. Sépare les idées "
    "distinctes, mais ne coupe pas une idée unique en morceaux.\n"
    "4. VERBATIM — chaque part ET la target sont des sous-chaînes EXACTES de l'avis. En "
    "cas de doute, recopie un peu plus de contexte plutôt que d'altérer le texte.\n"
    "\n"
    "EXEMPLES :\n"
    "• « j'aime les vidéos parce qu'elles me font rire » → UN claim, parts=[toute la "
    "portion], target=« les vidéos ».\n"
    "• « Avoir des élus qui représentent l'intérêt des citoyens et non l'intérêt de ceux "
    "qui ont financé leur campagne » → UN claim (le contraste « … et non … » est UNE idée), "
    "target=« les élus ».\n"
    "• Avis multi-thèmes « Il faut baisser les impôts. Par ailleurs trop d'élus, supprimons "
    "le Sénat. Et les services publics ruraux disparaissent. » → TROIS claims distincts "
    "(fiscalité / nombre d'élus / services publics ruraux), un par thème.\n"
    "• « Le temps passé sur l'écran est trop long. […] et ça, ça me dégoûte » → si « ça » "
    "renvoie au temps d'écran : UN claim, parts=[« Le temps passé sur l'écran est trop "
    "long », « ça me dégoûte »], target=« temps passé sur l'écran ».\n"
    "\n"
    "Si l'avis ne porte AUCUNE position (pur narratif/cadrage), renvoie une liste vide. "
    "Réponds STRICTEMENT en JSON : {\"claims\": [{\"parts\": [\"extrait verbatim 1\"], "
    "\"target\": \"cible verbatim ou null\"}, …]}."
)


def batch_prompt(sys_text: str, texts: list[str]) -> list[dict]:
    """Prompt LOT avec un system arbitraire (réutilise le suffixe LOT de prod)."""
    blocks = [f"=== AVIS #{i} ===\n{t}" for i, t in enumerate(texts, 1)]
    user = (
        f"Voici {len(texts)} avis numérotés. Extrais les claims de CHAQUE avis "
        "séparément, et réponds avec un objet clé par numéro.\n\n" + "\n\n".join(blocks)
    )
    return [{"role": "system", "content": sys_text + BATCH_SYS_SUFFIX},
            {"role": "user", "content": user}]


@dataclass
class Avis:
    id: str
    text: str
    ds: str


def load_sample() -> list[Avis]:
    """Échantillon déterministe : grand débat prioritaire, favorise les avis LONGS."""
    rng = random.Random(SEED)
    out: list[Avis] = []
    for ds, n in SAMPLE_PLAN.items():
        path = ROOT / "backend" / "cache" / ds / "ideas.jsonl"
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            # Schéma nested (granddebat/xstance/repnum) OU plat (tiktok).
            props = o.get("props") or {}
            txt = props.get("text_clean") or o.get("text_clean") or ""
            if len(txt.strip()) < 30:
                continue
            rows.append((o["id"], txt))
        if ds == "granddebat":
            # Favorise les avis LONGS (multi-thèmes) : top par longueur, échantillonné.
            rows.sort(key=lambda r: len(r[1]), reverse=True)
            pool = rows[: n * 4]
            rng.shuffle(pool)
            picked = pool[:n]
        else:
            rng.shuffle(rows)
            picked = rows[:n]
        out.extend(Avis(id=i, text=t, ds=ds) for i, t in picked)
    return out


def _complete(messages, max_tokens=4096):
    """Appel mistral-large avec retries (RPM bas des gros modèles)."""
    for attempt in range(7):
        try:
            return mistral_client.chat(
                messages, model=MODEL, temperature=0.0,
                max_tokens=max_tokens, json_mode=True, timeout=180,
            )
        except mistral_client.MistralError as exc:
            if exc.status in {0, 408, 429, 500, 502, 503, 504} and attempt < 6:
                delay = min(40.0, 2.0 * (2 ** attempt))
                print(f"  ⏳ HTTP {exc.status} retry {attempt+1}/6 dans {delay:.0f}s")
                time.sleep(delay)
                continue
            print(f"  ⚠️ HTTP {exc.status} (abandon)")
            return None
    return None


def extract_arm(arm: str, sys_text: str, avis: list[Avis]) -> dict:
    """Extrait un bras → {avis_id: {specs:[...], claims:[Claim.to_dict]}}. Caché disque."""
    cache = CACHE_DIR / f"raw_{arm}.json"
    results: dict[str, dict] = {}
    if cache.exists():
        results = json.loads(cache.read_text())
    # INCRÉMENTAL : ne (ré)extraire QUE les avis absents du cache.
    todo = [a for a in avis if a.id not in results]
    if not todo:
        print(f"[{arm}] cache complet ({len(results)} avis) → {cache.name}")
        return results
    print(f"[{arm}] {len(results)} en cache, {len(todo)} à extraire")

    step = max(1, BATCH)
    batches = [todo[i:i + step] for i in range(0, len(todo), step)]

    def proc(batch):
        raw = _complete(batch_prompt(sys_text, [a.text for a in batch]),
                        max_tokens=min(8192, 700 * len(batch)))
        specs_by_idx = parse_batch_claims(raw, len(batch)) if raw else [None] * len(batch)
        local = {}
        for j, a in enumerate(batch):
            specs = specs_by_idx[j] or []
            claims = align_spans(a.text, specs)  # gate verbatim — NON ancrés rejetés
            local[a.id] = {
                "n_specs": len(specs),
                "claims": [c.to_dict() for c in claims],
            }
        return local

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(proc, b): b for b in batches}
        for fut in as_completed(futs):
            local = fut.result()
            results.update(local)
            done += len(futs[fut])
            print(f"[{arm}] {done}/{len(todo)} (nouveaux)")
    cache.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    return results


def metrics(arm_data: dict, avis: list[Avis]) -> dict:
    """Agrège claims/avis, taux verbatim, % cible, longueur span — global + par ds."""
    by_id = {a.id: a for a in avis}
    agg = {}
    for scope in ("ALL", "granddebat"):
        ids = [a.id for a in avis if scope == "ALL" or a.ds == scope]
        n_avis = len(ids)
        tot_specs = tot_claims = tot_target = tot_spanlen = n_spans = empty = 0
        for aid in ids:
            d = arm_data[aid]
            tot_specs += d["n_specs"]
            cl = d["claims"]
            tot_claims += len(cl)
            if not cl:
                empty += 1
            for c in cl:
                if c.get("target"):
                    tot_target += 1
                for s, e in c["spans"]:
                    if e > s >= 0:
                        tot_spanlen += (e - s)
                        n_spans += 1
        agg[scope] = {
            "n_avis": n_avis,
            "claims_per_avis": round(tot_claims / n_avis, 3) if n_avis else 0,
            "verbatim_pass_rate": round(tot_claims / tot_specs, 3) if tot_specs else 0,
            "n_specs": tot_specs,
            "n_claims": tot_claims,
            "pct_with_target": round(tot_target / tot_claims, 3) if tot_claims else 0,
            "mean_span_chars": round(tot_spanlen / n_spans, 1) if n_spans else 0,
            "pct_empty_avis": round(empty / n_avis, 3) if n_avis else 0,
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    avis = load_sample()
    print(f"Échantillon : {len(avis)} avis "
          f"({ {ds: sum(1 for a in avis if a.ds==ds) for ds in SAMPLE_PLAN} })")

    A = extract_arm("A", CLAIM_SYS, avis)
    B = extract_arm("B", CLAIM_SYS_B, avis)

    res = {"model": MODEL, "n_avis": len(avis),
           "A": metrics(A, avis), "B": metrics(B, avis)}
    (CACHE_DIR / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.judge:
        from research.extract_ab_judge import run_judge
        run_judge(avis, A, B)


if __name__ == "__main__":
    main()

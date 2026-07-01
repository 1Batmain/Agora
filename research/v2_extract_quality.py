"""VALIDATION QUALITÉ RENFORCÉE — extraction v1 vs v2, juge AVEUGLE, PANEL de 3.

Objectif (Bob) : prouver le GAIN DE QUALITÉ de l'extraction v2 (question globale +
regroupement renforcé) plus solidement que « 2 flags corrigés », AVANT de payer une
re-extraction complète. On durcit la preuve sur trois axes :

  1. ÉCHANTILLON LARGE et VARIÉ — ~40 avis granddebat tirés au hasard (seed), stratifiés
     par longueur (courts / moyens / longs / très-longs multi-thèmes) + les 2 avis flaggés.
  2. JUGE AVEUGLE — chaque avis présenté à un PANEL de 3 juges mistral-large, étiquettes
     anonymisées (lot 1 / lot 2), ORDRE RANDOMISÉ INDÉPENDAMMENT par juge (décorrèle le
     biais de position), température 0.5 (diversité de panel réelle, sinon clones à T=0).
  3. CRITÈRES SÉPARÉS — sur-segmentation · complétude (pénalise la SUR-FUSION qui perd un
     thème) · fidélité (verbatim, pas de reformulation). Majorité du panel par critère.

Les deux bras passent le GATE VERBATIM dur (`align_spans`) : tout claim non sous-chaîne
exacte de l'avis est rejeté. V1 = snapshot EXACT du prompt de prod d'avant le commit v2.

Sortie : research/v2_extract_cache/{raw_v1,raw_v2,panel}.json + résumé imprimé.
Le rapport final (research/v2_quality_note.md) est écrit séparément.

Lancement (racine du worktree) :
  export MISTRAL_API_KEY=$(cat var/mistral.key)
  uv run python -m research.v2_extract_quality
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from pipeline.claims.extract import claim_sys, parse_claims
from pipeline.claims.ollama import parse_json_object
from pipeline.claims.span import align_spans
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "v2_extract_cache"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("AGORA_V2_MODEL", "mistral-large-latest")
SEED = 42
N_SAMPLE = 40            # avis tirés (hors flaggés)
N_PANEL = 3              # juges par avis
JUDGE_TEMP = 0.5         # diversité du panel (à T=0 les 3 juges sont identiques)

GD_QUESTION = (
    "Que faudrait-il faire pour renouer le lien entre les citoyens et "
    "les élus qui les représentent ?"
)
FLAGGED = ["granddebat:1-11830", "granddebat:1-4924"]

# --------------------------------------------------------------------------- #
# V1 — snapshot EXACT du prompt de prod AVANT le commit v2 (pas de question,
# regroupement non renforcé). Identique à research/extract_v2.py (iso-périmètre).
# --------------------------------------------------------------------------- #
CLAIM_SYS_V1 = (
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

# --------------------------------------------------------------------------- #
# JUGE — panel aveugle, 3 critères SÉPARÉS (noms du brief).
# --------------------------------------------------------------------------- #
JUDGE_SYS = (
    "Tu es un évaluateur RIGOUREUX et NEUTRE de l'extraction d'opinions citoyennes. On te "
    "donne UN avis, puis DEUX lots de claims (« lot 1 » et « lot 2 ») extraits de cet avis "
    "par deux systèmes différents. Un bon lot capture les PRISES DE POSITION de l'avis "
    "(griefs, opinions, propositions) en les recopiant verbatim.\n\n"
    "Juge les deux lots sur TROIS dimensions INDÉPENDANTES :\n"
    "• SUR_SEGMENTATION : quel lot évite le MIEUX de COUPER une même idée en morceaux ? "
    "Un lot qui sépare le PROBLÈME de la SOLUTION qui y répond, ou qui éclate une énumération "
    "détaillant UNE idée, SUR-SEGMENTE (mauvais). « Gagner » ici = NE PAS sur-segmenter "
    "(garder une idée entière dans un seul claim).\n"
    "• COMPLETUDE : quel lot capture le PLUS de prises de position RÉELLES de l'avis, sans "
    "en OUBLIER ? Un lot qui FUSIONNE deux thèmes distincts en perd un : c'est une PERTE de "
    "complétude (mauvais). « Gagner » ici = ne rater AUCUN thème réel.\n"
    "• FIDELITE : quel lot est le plus VERBATIM — recopie les mots de l'avis sans "
    "reformuler, résumer ni corriger ? « Gagner » = coller au texte d'origine.\n\n"
    "Attention : SUR_SEGMENTATION et COMPLETUDE sont en TENSION (trop fusionner perd des "
    "thèmes ; trop découper coupe les idées) — juge-les SÉPARÉMENT et honnêtement.\n"
    "Pour chaque dimension réponds « 1 », « 2 » ou « tie ». Sois exigeant : ne déclare "
    "« tie » que si les lots sont vraiment équivalents sur CETTE dimension. Réponds "
    "STRICTEMENT en JSON : {\"sur_segmentation\": \"1|2|tie\", \"completude\": \"1|2|tie\", "
    "\"fidelite\": \"1|2|tie\", \"justif\": \"une phrase courte\"}."
)

DIMS = ("sur_segmentation", "completude", "fidelite")


@dataclass
class Avis:
    id: str
    text: str


def load_sample() -> list[Avis]:
    """~40 avis tirés au hasard, STRATIFIÉS par longueur, + 2 flaggés en tête.

    Bandes (sur la distribution observée : p25≈79, p50≈148, p75≈274, p90≈510, p95≈781) :
      court   30–120   (idée unique / simple)
      moyen   120–350  (le gros du corpus)
      long    350–800  (multi-thèmes émergents)
      tres_long 800–2200 (vrai test du tradeoff complétude vs sur-fusion ;
                          cap 2200 pour éviter les essais p99 qui tronquent le JSON)
    On tire 10 par bande (40), avec un seed fixe → reproductible.
    """
    path = ROOT / "backend" / "cache" / "granddebat" / "ideas.jsonl"
    pool: list[Avis] = []
    flagged: dict[str, Avis] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        txt = ((o.get("props") or {}).get("text_clean") or "").strip()
        if len(txt) < 30:
            continue
        a = Avis(id=o["id"], text=txt)
        if o["id"] in FLAGGED:
            flagged[o["id"]] = a
        pool.append(a)

    rng = random.Random(SEED)
    bands = [("court", 30, 120), ("moyen", 120, 350),
             ("long", 350, 800), ("tres_long", 800, 2200)]
    flagged_ids = set(FLAGGED)
    picked: list[Avis] = []
    seen: set[str] = set()
    for _, lo, hi in bands:
        cand = [a for a in pool if lo <= len(a.text) < hi and a.id not in flagged_ids]
        rng.shuffle(cand)
        for a in cand[:10]:
            if a.id not in seen:
                picked.append(a)
                seen.add(a.id)
    # Flaggés en tête (lecture facile), sans doublon.
    head = [flagged[i] for i in FLAGGED if i in flagged and i not in seen]
    return head + picked


def mono_prompt(sys_text: str, text: str) -> list[dict]:
    return [{"role": "system", "content": sys_text},
            {"role": "user", "content": "Avis :\n" + text}]


def _complete(messages, max_tokens=3072, temperature=0.0):
    for attempt in range(7):
        try:
            return mistral_client.chat(
                messages, model=MODEL, temperature=temperature,
                max_tokens=max_tokens, json_mode=True, timeout=180,
            )
        except mistral_client.MistralError as exc:
            if exc.status in {0, 408, 429, 500, 502, 503, 504} and attempt < 6:
                time.sleep(min(40.0, 2.0 * (2 ** attempt)))
                continue
            return None
    return None


def extract(arm: str, sys_text: str, avis: list[Avis]) -> dict:
    """Extrait un bras mono-avis, gate verbatim, caché disque."""
    cache = OUT / f"raw_{arm}.json"
    res: dict = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [a for a in avis if a.id not in res]
    if not todo:
        print(f"[{arm}] cache complet ({len(res)})")
        return res

    def one(a: Avis):
        raw = _complete(mono_prompt(sys_text, a.text))
        specs = parse_claims(raw)
        claims = align_spans(a.text, specs)
        return a.id, {"n_specs": len(specs), "claims": [c.to_dict() for c in claims]}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, a) for a in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            aid, d = fut.result()
            res[aid] = d
            print(f"[{arm}] {k}/{len(todo)}  {aid}: {d['n_specs']} specs → "
                  f"{len(d['claims'])} claims")
    cache.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def _fmt(claims: list[dict], avis_text: str) -> str:
    if not claims:
        return "(aucun claim)"
    out = []
    for i, c in enumerate(claims, 1):
        tgt = c.get("target")
        tag = ""
        if tgt:
            tag = f"  [cible: « {avis_text[tgt[0]:tgt[1]]} »]"
        out.append(f"{i}. « {c['text']} »{tag}")
    return "\n".join(out)


def run_panel(avis: list[Avis], V1: dict, V2: dict) -> dict:
    """Panel de N_PANEL juges aveugles par avis. Chaque juge : ordre A/B randomisé
    indépendamment (seed avis+juge) + température. Majorité par critère → gagnant."""
    cache = OUT / "panel.json"
    if cache.exists():
        print("[panel] cache existant")
        return json.loads(cache.read_text())

    def judge_call(a: Avis, j: int):
        # Ordre randomisé indépendant par (avis, juge).
        r = random.Random(f"{a.id}|{j}")
        v1_is_lot1 = r.random() < 0.5
        c1, c2 = V1[a.id]["claims"], V2[a.id]["claims"]
        lot1, lot2 = (c1, c2) if v1_is_lot1 else (c2, c1)
        user = (f"AVIS :\n{a.text}\n\n--- LOT 1 ---\n{_fmt(lot1, a.text)}\n\n"
                f"--- LOT 2 ---\n{_fmt(lot2, a.text)}")
        raw = _complete([{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content": user}],
                        max_tokens=400, temperature=JUDGE_TEMP)
        obj = parse_json_object(raw or "") or {}

        def deanon(v):
            if v not in ("1", "2"):
                return "tie"
            winner_is_lot1 = (v == "1")
            # lot1 == v1 ssi v1_is_lot1 ; gagnant=lot1 → v1 ssi v1_is_lot1
            return "v1" if (winner_is_lot1 == v1_is_lot1) else "v2"

        return {dim: deanon(obj.get(dim)) for dim in DIMS} | {
            "judge": j, "v1_is_lot1": v1_is_lot1, "justif": obj.get("justif", "")}

    # Toutes les (avis × juge) en parallèle.
    jobs = [(a, j) for a in avis for j in range(N_PANEL)]
    raw_votes: dict[str, list] = {a.id: [None] * N_PANEL for a in avis}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(judge_call, a, j): (a.id, j) for a, j in jobs}
        done = 0
        for fut in as_completed(futs):
            aid, j = futs[fut]
            raw_votes[aid][j] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[panel] {done}/{len(jobs)} votes")

    # Majorité par avis & critère.
    per_avis = []
    for a in avis:
        votes = raw_votes[a.id]
        row = {"id": a.id, "chars": len(a.text),
               "n_v1": len(V1[a.id]["claims"]), "n_v2": len(V2[a.id]["claims"]),
               "votes": votes}
        for dim in DIMS:
            tally = Counter(v[dim] for v in votes)
            # gagnant = strict majorité (>N/2) sinon tie
            top, cnt = tally.most_common(1)[0]
            row[dim] = top if cnt > N_PANEL / 2 else "tie"
            row[dim + "_tally"] = dict(tally)
        per_avis.append(row)

    # Agrégat global par critère.
    agg = {}
    for dim in DIMS:
        c = Counter(r[dim] for r in per_avis)
        n = len(per_avis)
        agg[dim] = {
            "v2_wins": c.get("v2", 0), "v1_wins": c.get("v1", 0),
            "ties": c.get("tie", 0), "n": n,
            "v2_win_rate": round(c.get("v2", 0) / n, 3),
            "v1_win_rate": round(c.get("v1", 0) / n, 3),
            # taux v2 parmi les avis DÉCIDÉS (hors tie) — gain net réel
            "v2_rate_decided": round(
                c.get("v2", 0) / max(1, c.get("v2", 0) + c.get("v1", 0)), 3),
        }

    out = {"model": MODEL, "n_avis": len(avis), "n_panel": N_PANEL,
           "judge_temp": JUDGE_TEMP, "dims": DIMS, "agg": agg, "per_avis": per_avis}
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main():
    if not mistral_client.available():
        kf = ROOT / "var" / "mistral.key"
        if kf.exists():
            os.environ["MISTRAL_API_KEY"] = kf.read_text().strip()
    avis = load_sample()
    by_len = sorted(len(a.text) for a in avis)
    print(f"{len(avis)} avis (2 flaggés + {len(avis)-2} tirés), "
          f"longueurs {by_len[0]}–{by_len[-1]} chars, médiane {by_len[len(by_len)//2]}")

    V1 = extract("v1", CLAIM_SYS_V1, avis)
    V2 = extract("v2", claim_sys(GD_QUESTION), avis)

    # Stats de base (sanity).
    tot1 = sum(len(V1[a.id]["claims"]) for a in avis)
    tot2 = sum(len(V2[a.id]["claims"]) for a in avis)
    sp1 = sum(V1[a.id]["n_specs"] for a in avis)
    sp2 = sum(V2[a.id]["n_specs"] for a in avis)
    print(f"\nclaims/avis  v1={tot1/len(avis):.2f}  v2={tot2/len(avis):.2f}")
    print(f"verbatim     v1={tot1/max(1,sp1):.3f}  v2={tot2/max(1,sp2):.3f}")

    panel = run_panel(avis, V1, V2)
    print("\n================ PANEL (majorité de 3) ================")
    print(json.dumps(panel["agg"], ensure_ascii=False, indent=2))

    # Cas où v2 PERD (par critère) — pour le rapport honnête.
    print("\n---- cas où v2 PERD ----")
    for dim in DIMS:
        losers = [r for r in panel["per_avis"] if r[dim] == "v1"]
        print(f"\n[{dim}] v2 perd sur {len(losers)} avis :")
        for r in losers:
            j = next((v["justif"] for v in r["votes"] if v[dim] == "v1"), "")
            print(f"  {r['id']} (n_v1={r['n_v1']}, n_v2={r['n_v2']}): {j[:120]}")


if __name__ == "__main__":
    main()

"""V1 vs V2 EXTRACTION : anti-sur-segmentation cadrée par la question globale — R&D pur.

Problème (Bob + FLAGS) = SUR-SEGMENTATION : la découpe sépare le PROBLÈME de la SOLUTION
proposée, et éclate plusieurs phrases d'un MÊME sujet en plusieurs claims. Cas flaggés
(granddebat) :
  - 1-11830 : « … le mandat unique est une réponse » → problème + solution = 1 claim.
  - 1-4924  : « devoir de présence… sanction financière en cas d'absentéisme… » → les
              sous-points sur l'absentéisme = 1 claim.

V2 = deux changements (cf. pipeline/claims/extract.py) :
  1. injecte la QUESTION GLOBALE de la consultation (cadre la granularité) ;
  2. renforce le REGROUPEMENT (problème+solution=1 ; même sujet=1).

Validation SANS re-extraction complète : on re-extrait SEULEMENT les avis flaggés + 8-10
longs avis (multi-thèmes, le vrai test du tradeoff), v1 vs v2, gate verbatim `align_spans`
aux deux. On regarde : #claims v1/v2, fidélité verbatim, et le découpage des cas flaggés.

V1 = snapshot du prompt de prod AVANT ce commit (figé ci-dessous). V2 = `claim_sys(q)`
courant. Mono-avis (pas de batch) pour isoler la granularité par avis.

Usage :
  export MISTRAL_API_KEY=$(cat var/mistral.key)
  uv run python -m research.extract_v2
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from pipeline.claims.extract import claim_sys
from pipeline.claims.span import align_spans
from pipeline.cluster import mistral_client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "extract_v2_cache"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("AGORA_V2_MODEL", "mistral-large-latest")

# Question globale du Grand Débat (descriptor granddebat.json `question`).
GD_QUESTION = (
    "Que faudrait-il faire pour renouer le lien entre les citoyens et "
    "les élus qui les représentent ?"
)

# --------------------------------------------------------------------------- #
# V1 — snapshot EXACT du prompt de prod AVANT le commit v2 (regroupement non
# renforcé, pas de question). Figé ici pour comparer à isopérimètre.
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


@dataclass
class Avis:
    id: str
    text: str


def mono_prompt(sys_text: str, text: str) -> list[dict]:
    return [{"role": "system", "content": sys_text},
            {"role": "user", "content": "Avis :\n" + text}]


def load_avis() -> list[Avis]:
    """Charge les avis flaggés + les 10 avis granddebat les PLUS LONGS (multi-thèmes)."""
    path = ROOT / "backend" / "cache" / "granddebat" / "ideas.jsonl"
    rows: list[Avis] = []
    flagged = {"granddebat:1-11830", "granddebat:1-4924"}
    flagged_avis: list[Avis] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        txt = (o.get("props") or {}).get("text_clean") or ""
        if len(txt.strip()) < 30:
            continue
        a = Avis(id=o["id"], text=txt)
        if o["id"] in flagged:
            flagged_avis.append(a)
        rows.append(a)
    # 10 avis LONGS mais TYPIQUES (bande 500–1600 chars, > p90 sans les essais
    # pathologiques p99 de 8k chars qui tronquent le JSON) = test du tradeoff complétude.
    band = [a for a in rows if 500 <= len(a.text) <= 1600 and a.id not in flagged]
    band.sort(key=lambda r: len(r.text), reverse=True)
    longs = band[:10]
    # Flaggés en tête (lecture facile dans le rapport).
    return flagged_avis + longs


def _complete(messages):
    for attempt in range(7):
        try:
            return mistral_client.chat(
                messages, model=MODEL, temperature=0.0,
                max_tokens=3072, json_mode=True, timeout=180,
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


def parse_specs(raw: str | None) -> list[dict]:
    """Réponse mono-avis {"claims":[{parts,target}]} → specs (tolérant, comme la prod)."""
    from pipeline.claims.extract import parse_claims
    return parse_claims(raw)


def extract(arm: str, sys_text: str, avis: list[Avis]) -> dict:
    """Extrait un bras mono-avis → {id: {n_specs, claims:[dict]}}. Caché disque."""
    cache = OUT / f"raw_{arm}.json"
    res: dict = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [a for a in avis if a.id not in res]
    if not todo:
        print(f"[{arm}] cache complet ({len(res)})")
        return res
    for k, a in enumerate(todo, 1):
        raw = _complete(mono_prompt(sys_text, a.text))
        specs = parse_specs(raw)
        claims = align_spans(a.text, specs)  # gate verbatim
        res[a.id] = {"n_specs": len(specs), "claims": [c.to_dict() for c in claims]}
        print(f"[{arm}] {k}/{len(todo)}  {a.id}: {len(specs)} specs → {len(claims)} claims")
    cache.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    avis = load_avis()
    by_id = {a.id: a for a in avis}
    print(f"{len(avis)} avis (2 flaggés + {len(avis)-2} longs)")

    V1 = extract("v1", CLAIM_SYS_V1, avis)
    V2 = extract("v2", claim_sys(GD_QUESTION), avis)

    # Tableau + métriques.
    lines = []
    tot1 = tot2 = vb1n = vb1d = vb2n = vb2d = 0
    for a in avis:
        d1, d2 = V1[a.id], V2[a.id]
        c1, c2 = len(d1["claims"]), len(d2["claims"])
        tot1 += c1
        tot2 += c2
        vb1n += c1
        vb1d += d1["n_specs"]
        vb2n += c2
        vb2d += d2["n_specs"]
        lines.append((a.id, len(a.text), c1, c2))

    summary = {
        "model": MODEL,
        "n_avis": len(avis),
        "claims_v1": tot1,
        "claims_v2": tot2,
        "claims_per_avis_v1": round(tot1 / len(avis), 2),
        "claims_per_avis_v2": round(tot2 / len(avis), 2),
        "verbatim_pass_v1": round(vb1n / vb1d, 3) if vb1d else 0,
        "verbatim_pass_v2": round(vb2n / vb2d, 3) if vb2d else 0,
        "rows": [{"id": i, "chars": n, "v1": a, "v2": b} for i, n, a, b in lines],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Dump lisible des claims pour les 2 flaggés (jugement humain vs flags).
    dump = OUT / "flagged_claims.txt"
    with dump.open("w", encoding="utf-8") as f:
        for fid in ("granddebat:1-11830", "granddebat:1-4924"):
            a = by_id[fid]
            f.write(f"\n{'='*70}\n{fid}  ({len(a.text)} chars)\n{a.text}\n")
            for arm, data in (("V1", V1), ("V2", V2)):
                f.write(f"\n--- {arm} : {len(data[fid]['claims'])} claims ---\n")
                for j, c in enumerate(data[fid]["claims"], 1):
                    tgt = c.get("target")
                    tgt_txt = a.text[tgt[0]:tgt[1]] if tgt else None
                    f.write(f"  [{j}] {c['text']!r}  (target={tgt_txt!r})\n")
    print(f"\n→ {dump}")


if __name__ == "__main__":
    main()

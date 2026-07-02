"""TÉMOIN re-coupe — façade macro du CORPUS COMPLET (22k) re-coupé sauce_magique vs officiel.

Reprend le protocole de `granddebat_witness_v2.py` À L'IDENTIQUE (même TÉMOIN : les 14
sous-thèmes officiels OpinionWay de l'axe « Démocratie & citoyenneté » restreints à la
question « renouer le lien citoyens ↔ élus » ; même juge mistral-large, même barème 0-5,
mapping expert avec LISTE de sous-thèmes officiels par macro), appliqué à la façade
issue de la RE-COUPE sauce_magique du corpus complet granddebat (22 174 avis) :
37 macros = 17 thèmes réels + 20 singletons (1 avis, bruit — IGNORÉS comme en v2).

Différence de structure vs v2 (3k re-extrait) : la re-coupe du 22k donne des macros
plus GROS et consolidés — n2 (« Implication citoyenne », 4590 avis) porte À LA FOIS
participation, débats/concertation ET tirage au sort (403 claims « tirage/tirés au
sort » dans ses citations) ; le vote blanc (n31) reste une facette du mode de scrutin
(même convention que v2/n346).

Usage :
    MISTRAL_API_KEY=$(cat var/mistral.key) uv run python research/granddebat_witness_recut.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cluster import mistral_client  # noqa: E402

# Cache DEV du corpus complet re-coupé (identique à ce qui est promu en prod
# sous l'id `granddebat-complet`). Surchageable : argv[1] = chemin du dossier analysis/.
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "~/agora-dev/backend/cache/granddebat/analysis"
)
OUT = Path(__file__).resolve().parent / "granddebat_witness_recut_results.json"
JUDGE_MODEL = "mistral-large-latest"

# --- TÉMOIN : 14 sous-thèmes officiels (IDENTIQUE à granddebat_witness{,_v2}.py) --- #
OFFICIAL = {
    "scrutin_proportionnelle": "Mode de scrutin : introduire une dose de proportionnelle "
        "aux législatives pour une Assemblée plus représentative ; reconnaissance du vote "
        "blanc ; débat sur le vote obligatoire.",
    "ric_referendum": "Référendum d'initiative citoyenne (RIC) et recours élargi au "
        "référendum : législatif, abrogatoire, révocatoire, constituant.",
    "cumul_limitation_mandats": "Limitation et non-cumul des mandats dans le temps pour "
        "renouveler la classe politique et éviter la professionnalisation.",
    "nombre_elus": "Réduction du nombre d'élus et de parlementaires ; rôle et fonctionnement "
        "du Parlement (Assemblée, Sénat, quorum).",
    "privileges_remunerations": "Privilèges, rémunérations, frais et retraites des élus jugés "
        "excessifs : suppression / plafonnement / contrôle (moralisation).",
    "transparence_depenses": "Transparence et contrôle de l'argent public : publication et "
        "justification des dépenses, lutte contre le gaspillage.",
    "reddition_comptes": "Responsabilité et reddition de comptes des élus : respect des "
        "promesses de campagne, bilans réguliers, sanctions en cas de manquement.",
    "casier_probite": "Probité et casier judiciaire : casier vierge exigé, inéligibilité, "
        "sanctions pénales des élus condamnés.",
    "participation_consultation": "Démocratie participative : associer les citoyens aux "
        "décisions entre les élections (consultations, plateformes, budgets participatifs).",
    "debats_concertation": "Réunions publiques, débats et concertation de proximité ; "
        "intermédiaires et instances de dialogue élu-citoyen.",
    "tirage_au_sort": "Tirage au sort de citoyens / assemblées citoyennes pour représenter et "
        "contrôler (CESE, conseils citoyens).",
    "lien_proximite_confiance": "Restaurer le lien, la confiance et l'écoute : élus proches du "
        "terrain, à l'écoute des réalités vécues, exemplarité.",
    "decentralisation_territoriale": "Organisation territoriale : clarifier les compétences "
        "État/régions/départements/communes, simplifier le millefeuille, proximité.",
    "renouvellement_classe_politique": "Renouvellement et diversification de la classe "
        "politique : profils issus de la société civile, statut de l'élu, lutte contre les "
        "lobbies et la « politique politicienne ».",
}
OFFICIAL_OUT_OF_SCOPE = {
    "immigration_laicite": "Immigration, intégration, laïcité (axe Démocratie mais autres "
        "colonnes de la source — hors question 'lien citoyens-élus').",
    "civisme_engagement": "Citoyenneté, civisme, service national/civique, devoirs (autres "
        "colonnes de la source).",
}

# --- MAPPING expert macro re-coupée → sous-thème(s) officiel(s) ([] = hors-axe) --- #
# 17 macros réelles ; les 20 singletons (1 avis) sont IGNORÉS (bruit, comme en v2).
MAPPING = {
    "n1":  ["lien_proximite_confiance"],
    # n2 consolide participation + réunions/débats + tirage au sort (kw « tirés »,
    # « tirage », « assemblée » ; 403 claims tirage-au-sort dans citations/n2.json).
    "n2":  ["participation_consultation", "debats_concertation", "tirage_au_sort"],
    "n3":  ["lien_proximite_confiance"],
    "n4":  ["transparence_depenses"],
    "n5":  ["privileges_remunerations"],
    "n23": ["casier_probite"],
    # n24 = exaspération envers les comportements / « politique politicienne ».
    "n24": ["renouvellement_classe_politique"],
    "n25": ["nombre_elus"],
    "n26": ["cumul_limitation_mandats"],
    "n27": ["ric_referendum"],
    "n28": ["scrutin_proportionnelle"],
    "n29": ["decentralisation_territoriale"],
    "n30": ["reddition_comptes"],
    "n31": ["scrutin_proportionnelle"],   # vote blanc = facette du mode de scrutin (cf. v2)
    "n32": [],   # éducation civique/civisme — colonnes hors question (attendu ABSENT du témoin)
    "n33": ["lien_proximite_confiance"],
    # n34 = lobbys / financement de la vie politique → « lutte contre les lobbies ».
    "n34": ["renouvellement_classe_politique"],
}

JUDGE_SYS = (
    "Tu es un évaluateur expert des consultations citoyennes françaises. On te donne "
    "(A) un thème ÉMERGENT produit par un système d'analyse automatique sur le Grand "
    "Débat National 2019, avec son titre et sa synthèse, et (B) la description d'un "
    "SOUS-THÈME OFFICIEL de la synthèse de référence (OpinionWay). Juge à quel point le "
    "thème émergent CORRESPOND fidèlement au sous-thème officiel.\n\n"
    "Réponds STRICTEMENT en JSON : {\"alignment\": <0-5>, \"verdict\": <\"faithful\"|"
    "\"finer_split\"|\"partial\"|\"mismatch\">, \"justification\": <1 phrase>}.\n"
    "- alignment 5 = le thème émergent capte exactement et entièrement le sous-thème "
    "officiel ; 0 = aucun rapport.\n"
    "- verdict 'faithful' = bonne correspondance directe ; 'finer_split' = le thème "
    "émergent est une facette PLUS FINE/précise du sous-thème officiel (légitime, plus "
    "granulaire) ; 'partial' = ne couvre qu'une partie ; 'mismatch' = mauvais "
    "appariement."
)


def load_macros() -> list[dict]:
    a = json.loads((CACHE / "analysis.json").read_text(encoding="utf-8"))
    macros = [t for t in a["themes"] if t["parent_id"] is None]
    macros.sort(key=lambda t: -t["n_avis"])
    out = []
    for m in macros:
        ins = CACHE / "insights" / f"{m['id']}.json"
        synth = ""
        if ins.exists():
            synth = json.loads(ins.read_text(encoding="utf-8")).get("markdown", "")
        out.append({
            "id": m["id"], "title": m["title"], "hook": m.get("hook", ""),
            "n_avis": m["n_avis"], "n_claims": m["n_claims"],
            "keywords": m.get("keywords", [])[:15], "synthesis": synth,
        })
    return out


def judge(macro: dict, official_key: str) -> dict:
    a_block = (
        f"TITRE: {macro['title']}\n"
        f"ACCROCHE: {macro['hook']}\n"
        f"MOTS-CLÉS: {', '.join(macro['keywords'])}\n"
        f"SYNTHÈSE:\n{macro['synthesis'][:1400]}"
    )
    b_block = OFFICIAL[official_key]
    user = (
        f"(A) THÈME ÉMERGENT (Agora)\n{a_block}\n\n"
        f"(B) SOUS-THÈME OFFICIEL [{official_key}]\n{b_block}\n\n"
        "JSON :"
    )
    raw = ""
    for attempt in range(6):
        try:
            raw = mistral_client.chat(
                [{"role": "system", "content": JUDGE_SYS},
                 {"role": "user", "content": user}],
                model=JUDGE_MODEL, temperature=0.0, max_tokens=300, json_mode=True,
                timeout=120.0,
            )
            break
        except mistral_client.MistralError as exc:
            wait = min(30.0, 2.0 * (2 ** attempt))
            print(f"      retry {attempt + 1}/6 ({exc}) → {wait:.0f}s")
            time.sleep(wait)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = {"alignment": None, "verdict": "parse_error", "justification": raw[:200]}
    return d


def main() -> None:
    macros = load_macros()
    print(f"{len(macros)} macros (corpus complet re-coupé) chargés depuis {CACHE}")
    results = []
    best_by_official: dict[str, dict] = {}
    for m in macros:
        if m["n_avis"] <= 1:                     # singleton = bruit, ignoré (cf. v2)
            print(f"  {m['id']:>4} ({m['n_avis']:>4} avis) → SINGLETON IGNORÉ : {m['title']}")
            results.append({**{k: m[k] for k in ('id', 'title', 'n_avis', 'n_claims')},
                            "official": None, "judges": [], "note": "singleton_ignored"})
            continue
        keys = MAPPING.get(m["id"])
        if keys is None:
            print(f"  {m['id']:>4} ({m['n_avis']:>4} avis) → NON MAPPÉ (?) : {m['title']}")
            results.append({**{k: m[k] for k in ('id', 'title', 'n_avis', 'n_claims')},
                            "official": None, "judges": [], "note": "unmapped"})
            continue
        if not keys:
            print(f"  {m['id']:>4} ({m['n_avis']:>4} avis) → AJOUT/hors-axe : {m['title']}")
            results.append({**{k: m[k] for k in ('id', 'title', 'n_avis', 'n_claims')},
                            "official": None, "judges": [], "note": "out_of_axis"})
            continue
        judges = []
        for ok in keys:
            j = judge(m, ok)
            j["official"] = ok
            judges.append(j)
            al = j.get("alignment")
            print(f"  {m['id']:>4} ({m['n_avis']:>4} avis) → {ok:<30} "
                  f"align={al} {j.get('verdict')}")
            if isinstance(al, (int, float)):
                prev = best_by_official.get(ok)
                if prev is None or al > prev["alignment"]:
                    best_by_official[ok] = {"alignment": al, "macro": m["id"],
                                            "verdict": j.get("verdict")}
        results.append({**{k: m[k] for k in ('id', 'title', 'n_avis', 'n_claims')},
                        "official": keys, "judges": judges})

    covered = set(best_by_official)
    missing = [k for k in OFFICIAL if k not in covered]
    additions = [r for r in results if r.get("note") == "out_of_axis"]
    singletons = [r for r in results if r.get("note") == "singleton_ignored"]
    all_pair_aligns = [j["alignment"] for r in results for j in r.get("judges", [])
                       if isinstance(j.get("alignment"), (int, float))]
    cover_aligns = [v["alignment"] for v in best_by_official.values()]
    all_judges = [j for r in results for j in r.get("judges", [])]
    summary = {
        "n_macros": len(macros),
        "n_macros_scored": sum(1 for r in results if r.get("judges")),
        "n_singletons_ignored": len(singletons),
        "official_subthemes_total": len(OFFICIAL),
        "official_subthemes_covered": len(covered),
        "coverage_pct": round(100 * len(covered) / len(OFFICIAL), 1),
        "covered": sorted(covered),
        "missing": missing,
        "out_of_scope_expected_absent": list(OFFICIAL_OUT_OF_SCOPE),
        "additions": [{"id": r["id"], "title": r["title"], "n_avis": r["n_avis"]} for r in additions],
        "singletons_ignored": [{"id": r["id"], "title": r["title"]} for r in singletons],
        "mean_alignment_all_pairs": round(sum(all_pair_aligns) / len(all_pair_aligns), 2) if all_pair_aligns else None,
        "mean_alignment_coverage": round(sum(cover_aligns) / len(cover_aligns), 2) if cover_aligns else None,
        "n_pairs_judged": len(all_pair_aligns),
        "faithful": sum(1 for j in all_judges if j.get("verdict") == "faithful"),
        "finer_split": sum(1 for j in all_judges if j.get("verdict") == "finer_split"),
        "partial": sum(1 for j in all_judges if j.get("verdict") == "partial"),
        "mismatch": sum(1 for j in all_judges if j.get("verdict") == "mismatch"),
        "best_by_official": best_by_official,
    }
    OUT.write_text(json.dumps({"summary": summary, "results": results,
                               "official": OFFICIAL, "mapping": MAPPING},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RÉSUMÉ re-coupe (corpus complet) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nÉcrit → {OUT}")


if __name__ == "__main__":
    main()

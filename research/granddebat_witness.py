"""TÉMOIN — synthèses Agora vs synthèse OFFICIELLE du Grand Débat National (2019).

But : se servir de la synthèse OFFICIELLE (OpinionWay / granddebat.fr) comme
témoin/cible pour juger la qualité des thèmes+synthèses qu'Agora fait émerger sur
le dataset `granddebat`.

Particularité du corpus : ce dataset = espace « Démocratie et citoyenneté »,
colonne 13 = question OUVERTE « Que faudrait-il faire pour renouer le lien entre
les citoyens et les élus qui les représentent ? ». Les 4 axes officiels du Grand
Débat (Fiscalité · Organisation de l'État · Transition écologique · Démocratie &
citoyenneté) NE sont donc PAS tous dans le corpus : un seul axe (Démocratie &
citoyenneté) est échantillonné. Le vrai test de couverture porte sur les
SOUS-THÈMES OFFICIELS de cet axe, tels que listés dans la synthèse OpinionWay.

Méthode :
  1) charge les macros Agora + leurs synthèses (cache analysis.json + insights/) ;
  2) MAPPE chaque macro Agora → sous-thème officiel le plus proche (table experte,
     vérité terrain qualitative — pas du code corpus-spécifique du pipeline) ;
  3) LLM-juge (Mistral) : alignement macro Agora ↔ sous-thème officiel (0-5 +
     verdict faithful / finer-split / addition / off-topic) ;
  4) couverture : combien des sous-thèmes officiels Agora retrouve-t-il, manques,
     ajouts.

Sortie : research/granddebat_witness_results.json (chiffré). La note de synthèse
manuelle est research/granddebat_witness_note.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cluster import mistral_client  # noqa: E402

# Cache produit par le pipeline mergé (build_analysis) sur le repo principal.
CACHE = Path(
    "/home/bat/projects/Analyse-des-consultations-citoyennes/backend/cache/granddebat/analysis"
)
OUT = Path(__file__).resolve().parent / "granddebat_witness_results.json"
JUDGE_MODEL = "mistral-large-latest"

# --------------------------------------------------------------------------- #
# VÉRITÉ TERRAIN — sous-thèmes officiels de l'axe « Démocratie & citoyenneté »
# (synthèse OpinionWay / granddebat.fr 2019), restreints à ce qui concerne la
# question ouverte « renouer le lien citoyens ↔ élus ». Source : cf. mémoire
# `agora-official-syntheses`. C'est le TÉMOIN, pas du code de clustering.
# --------------------------------------------------------------------------- #
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
    # Sous-thèmes officiels de l'axe HORS de cette question ouverte (autres colonnes
    # de la source : immigration, laïcité, civisme/service national). Attendus ABSENTS
    # de ce corpus — leur absence n'est PAS un manque d'Agora.
}
OFFICIAL_OUT_OF_SCOPE = {
    "immigration_laicite": "Immigration, intégration, laïcité (axe Démocratie mais autres "
        "colonnes de la source — hors question 'lien citoyens-élus').",
    "civisme_engagement": "Citoyenneté, civisme, service national/civique, devoirs (autres "
        "colonnes de la source).",
}

# --------------------------------------------------------------------------- #
# MAPPING expert macro Agora → sous-thème officiel (None = addition/hors-axe).
# --------------------------------------------------------------------------- #
MAPPING = {
    "n0":  "lien_proximite_confiance",
    "n1":  "lien_proximite_confiance",
    "n2":  "participation_consultation",
    "n3":  "nombre_elus",
    "n4":  "renouvellement_classe_politique",
    "n5":  "privileges_remunerations",
    "n27": "debats_concertation",
    "n19": "scrutin_proportionnelle",
    "n20": "decentralisation_territoriale",
    "n29": "transparence_depenses",
    "n30": "reddition_comptes",
    "n28": "casier_probite",
    "n31": "cumul_limitation_mandats",
    "n32": "ric_referendum",
    "n33": "tirage_au_sort",
    "n34": None,   # limitation 80km/h — débordement gilets jaunes, hors-axe
    "n35": None,   # PNL (1 avis) — bruit singleton
    "n36": None,   # parcours énarques (1 avis) — bruit singleton
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
    for attempt in range(6):                       # backoff borné (429/timeout/5xx)
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
    print(f"{len(macros)} macros Agora chargés depuis {CACHE}")
    results = []
    covered = set()
    for m in macros:
        ok = MAPPING.get(m["id"])
        if ok is None:
            print(f"  {m['id']:>4} ({m['n_avis']:>4} avis) → AJOUT/hors-axe : {m['title']}")
            results.append({**{k: m[k] for k in ('id', 'title', 'n_avis', 'n_claims')},
                            "official": None, "judge": None})
            continue
        j = judge(m, ok)
        covered.add(ok)
        print(f"  {m['id']:>4} ({m['n_avis']:>4} avis) → {ok:<28} "
              f"align={j.get('alignment')} {j.get('verdict')}")
        results.append({**{k: m[k] for k in ('id', 'title', 'n_avis', 'n_claims')},
                        "official": ok, "judge": j})

    missing = [k for k in OFFICIAL if k not in covered]
    additions = [r for r in results if r["official"] is None]
    aligns = [r["judge"]["alignment"] for r in results
              if r["judge"] and isinstance(r["judge"].get("alignment"), (int, float))]
    summary = {
        "n_macros": len(macros),
        "official_subthemes_total": len(OFFICIAL),
        "official_subthemes_covered": len(covered),
        "coverage_pct": round(100 * len(covered) / len(OFFICIAL), 1),
        "covered": sorted(covered),
        "missing": missing,
        "out_of_scope_expected_absent": list(OFFICIAL_OUT_OF_SCOPE),
        "additions": [{"id": r["id"], "title": r["title"], "n_avis": r["n_avis"]} for r in additions],
        "mean_alignment": round(sum(aligns) / len(aligns), 2) if aligns else None,
        "faithful": sum(1 for r in results if r["judge"] and r["judge"].get("verdict") == "faithful"),
        "finer_split": sum(1 for r in results if r["judge"] and r["judge"].get("verdict") == "finer_split"),
        "partial": sum(1 for r in results if r["judge"] and r["judge"].get("verdict") == "partial"),
        "mismatch": sum(1 for r in results if r["judge"] and r["judge"].get("verdict") == "mismatch"),
    }
    OUT.write_text(json.dumps({"summary": summary, "results": results,
                               "official": OFFICIAL, "mapping": MAPPING},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RÉSUMÉ ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nÉcrit → {OUT}")


if __name__ == "__main__":
    main()

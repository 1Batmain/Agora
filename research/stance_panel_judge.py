"""JUGE AVEUGLE — panel de 3 passes sur les 26 paires anonymisées de la cible de stance a/b/c.

Protocole STRICT (comme les panels extraction-v2) : pour chaque paire anonymisée
(deux cibles candidates X/Y pour la même feuille, ordre aléatoire, provenance masquée),
3 juges INDÉPENDANTS (mistral-large, température 0, prompts LÉGÈREMENT variés) répondent
« quelle cible capture le mieux le débat CENTRAL du thème, en restant polaire et
débattable ? » → X, Y ou nul. MAJORITÉ = décision de la paire ; les égalités sont comptées.

Le juge NE regarde PAS la clé pair_id→variante (dépouillé APRÈS, script séparé). Sortie :
`research/stance_target_ab_panel_votes.jsonl` (votes bruts par paire, SANS étiquette a/b/c).

Lancement :
    MISTRAL_API_KEY=... python research/stance_panel_judge.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
PANEL_PATH = RESEARCH_DIR / "stance_target_ab_panel.jsonl"
VOTES_PATH = RESEARCH_DIR / "stance_target_ab_panel_votes.jsonl"

MODEL = "mistral-large-latest"
API_URL = "https://api.mistral.ai/v1/chat/completions"

# Trois cadrages de juge LÉGÈREMENT variés — même tâche, formulation différente, pour que le
# panel ne soit pas trois copies du même biais de phrasé. Tous exigent central + polaire.
JUDGE_SYSTEMS = [
    # (1) neutre, proche de la question du panel
    "Tu es un juge impartial de la qualité d'objets de clivage citoyens. On te donne le TITRE "
    "d'un thème, quelques contributions, et DEUX propositions d'objet de clivage (X et Y). "
    "Choisis celle qui capture le mieux le sujet CENTRAL du thème — pas une facette secondaire "
    "ni le détail le plus bruyant — tout en restant une proposition POLAIRE claire sur laquelle "
    "on peut être POUR ou CONTRE. Si les deux se valent, réponds \"nul\". "
    "Réponds STRICTEMENT en JSON : {\"choix\": \"X\"|\"Y\"|\"nul\", \"raison\": \"...\"}.",
    # (2) accent sur central > saillant
    "Tu évalues deux résumés du clivage AU CŒUR d'un thème citoyen. Le meilleur résumé porte "
    "sur ce qui est CENTRAL et représentatif du thème (le débat que la majorité des "
    "contributions partagent), et NON sur un aspect saillant ou marginal qui ne concernerait "
    "qu'une minorité. Il doit aussi rester une affirmation débattable (POUR/CONTRE possible). "
    "Entre X et Y, lequel est le meilleur à ce titre ? \"nul\" si équivalents. "
    "Réponds STRICTEMENT en JSON : {\"choix\": \"X\"|\"Y\"|\"nul\", \"raison\": \"...\"}.",
    # (3) accent sur polaire/débattable + éviter le passe-partout
    "Tu es juré d'un panel qui note des objets de débat. Pour le thème donné, deux formulations "
    "(X, Y) proposent l'enjeu débattable central. Une bonne formulation est (i) fidèle au sujet "
    "CENTRAL du thème, (ii) nettement POLAIRE — on voit un camp POUR et un camp CONTRE —, et "
    "(iii) ni trop vague/passe-partout ni hors-sujet. Laquelle, X ou Y, remplit le mieux ces "
    "trois critères ? Réponds \"nul\" si vraiment équivalentes. "
    "Réponds STRICTEMENT en JSON : {\"choix\": \"X\"|\"Y\"|\"nul\", \"raison\": \"...\"}.",
]


def _user_message(item: dict) -> str:
    claims = "\n".join(f"- {c[:400]}" for c in item.get("context_claims", []))
    return (
        f"TITRE DU THÈME : {item['title']}\n\n"
        f"CONTRIBUTIONS (contexte) :\n{claims}\n\n"
        f"PROPOSITION X : {item['option_X']}\n"
        f"PROPOSITION Y : {item['option_Y']}\n\n"
        "Quelle proposition (X ou Y) capture le mieux le débat CENTRAL du thème tout en "
        "restant polaire et débattable ? \"nul\" si équivalentes."
    )


def _call(api_key: str, system: str, user: str, retries: int = 4) -> dict:
    body = json.dumps({
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            data = json.loads(content)
            choix = str(data.get("choix", "")).strip().upper()
            if choix not in ("X", "Y", "NUL"):
                choix = "NUL"
            return {"choix": "nul" if choix == "NUL" else choix,
                    "raison": str(data.get("raison", "")).strip()[:300]}
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
                KeyError, TimeoutError) as exc:
            last = exc
            code = getattr(exc, "code", None)
            # 429 / 5xx → backoff et on réessaie
            wait = 3 * (attempt + 1)
            if code and 400 <= code < 429 and code != 429:
                break  # erreur cliente non transitoire
            time.sleep(wait)
    return {"choix": "err", "raison": f"(échec: {type(last).__name__} {last})"}


def _decision(votes: list[str]) -> tuple[str, int]:
    """Majorité sur [X|Y|nul|err]. Retourne (décision, marge). Décision ∈ {X,Y,tie}.
    tie = pas de majorité stricte pour X ou Y (inclut 2 'nul', ou 1-1-1, ou X-Y-nul)."""
    nx = votes.count("X")
    ny = votes.count("Y")
    if nx > ny and nx >= 2:
        return "X", nx
    if ny > nx and ny >= 2:
        return "Y", ny
    return "tie", max(nx, ny)


def main() -> None:
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        sys.exit("MISTRAL_API_KEY manquant.")
    items = [json.loads(l) for l in PANEL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(items)} paires anonymisées, panel de {len(JUDGE_SYSTEMS)} juges (mistral-large, temp 0)\n",
          flush=True)

    out = []
    for i, item in enumerate(items, 1):
        user = _user_message(item)
        judges = [_call(api_key, sys_p, user) for sys_p in JUDGE_SYSTEMS]
        votes = [j["choix"] for j in judges]
        decision, margin = _decision([v for v in votes])
        rec = {
            "pair_id": item["pair_id"],
            "title": item["title"],
            "option_X": item["option_X"],
            "option_Y": item["option_Y"],
            "votes": votes,
            "decision": decision,
            "margin": margin,
            "judge_raisons": [j["raison"] for j in judges],
        }
        out.append(rec)
        print(f"[{i:>2}/{len(items)}] {item['pair_id']:<12} votes={votes} → {decision} ({margin})",
              flush=True)

    with VOTES_PATH.open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_err = sum(1 for r in out if "err" in r["votes"])
    print(f"\n✓ {len(out)} paires jugées → {VOTES_PATH}"
          + (f"  ⚠ {n_err} paires avec ≥1 erreur d'appel" if n_err else ""), flush=True)


if __name__ == "__main__":
    main()

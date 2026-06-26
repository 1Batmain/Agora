"""Seed des contributions de démo de la consultation OUVERTE « Améliorer Agora ».

Écrit `backend/cache/ameliorer-agora/submissions.seed.jsonl` (COMMITTÉ) : ~8 retours
variés (ergonomie, lenteur, fiabilité des analyses, idées de features…) embeddés
nomic-v2, pour que la corrélation soit NON VIDE dès la 1ʳᵉ vraie contribution.

Lancer (une fois) :
    uv run --extra embed-contender python -m backend.scripts.seed_ameliorer_agora
"""

from __future__ import annotations

import json

from backend.submissions import embed_text, seed_path

CONSULTATION_ID = "ameliorer-agora"

# Retours d'exemple VARIÉS (ergonomie · perf · qualité des analyses · accessibilité ·
# fonctionnalités souhaitées). Horodatages fictifs étalés (ordre d'arrivée plausible).
SEEDS: list[tuple[str, str]] = [
    ("L'interface est trop chargée, on se perd dans les menus. Il faudrait simplifier "
     "la navigation et mettre en avant l'essentiel.", "2026-06-10T09:12:00+00:00"),
    ("Le chargement des analyses est vraiment lent, j'attends plusieurs secondes à "
     "chaque clic. C'est décourageant.", "2026-06-11T14:03:00+00:00"),
    ("J'aimerais pouvoir exporter les résultats d'une consultation en PDF ou en CSV "
     "pour les partager et les retravailler.", "2026-06-12T08:47:00+00:00"),
    ("Les thèmes proposés par l'IA ne correspondent pas toujours à ce que les gens ont "
     "écrit : ça manque parfois de fidélité.", "2026-06-13T17:25:00+00:00"),
    ("Il faudrait une vraie version mobile : l'application est presque inutilisable "
     "depuis un téléphone.", "2026-06-15T21:40:00+00:00"),
    ("Ce serait bien d'avoir un moteur de recherche pour retrouver un avis précis "
     "dans une consultation volumineuse.", "2026-06-17T11:18:00+00:00"),
    ("Le contraste et la taille de certains textes rendent la lecture difficile : "
     "pensez à l'accessibilité des personnes malvoyantes.", "2026-06-19T10:05:00+00:00"),
    ("On devrait pouvoir réagir ou répondre aux contributions des autres citoyens, "
     "pour que ça devienne un vrai échange.", "2026-06-21T16:52:00+00:00"),
]


def main() -> None:
    path = seed_path(CONSULTATION_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for text, ts in SEEDS:
            vec = embed_text(text)
            row = {"text": text, "vec": [float(x) for x in vec.ravel()], "ts": ts}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"seedé {len(SEEDS)} contributions → {path}")


if __name__ == "__main__":
    main()

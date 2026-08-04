"""État d'une session LIVE — thèmes gelés, assignations, positions par orateur, dérive.

Le pipeline live n'est PAS le pipeline batch au ralenti. Il repose sur un choix explicite
(cf. `.agent/notes/LIVE_SCENARIO.md` §3) :

> On **assigne** chaque claim nouveau au thème existant le plus proche, au lieu de
> **repartitionner**. Les thèmes sont GELÉS après amorçage.

Ce choix contourne le problème dur — garder l'identité des thèmes à travers une repartition,
sans quoi la carte se remélange sous les yeux du spectateur — au lieu de le résoudre. Le prix
à payer est la **dérive** : à mesure que le débat avance, des claims tombent loin de tout
centroïde. On la MESURE ici pour savoir quand une restructuration s'impose ; on ne la subit pas.

## Ce que l'état porte, et ce qu'il ne porte pas

- Il porte les thèmes (centroïde + objet de clivage), les assignations, les positions
  agrégées par orateur, et les indicateurs de dérive.
- Il ne porte **aucune** position déduite de la géométrie : une position vient TOUJOURS d'un
  jugement LLM (`pipeline.stance`), jamais d'une proximité d'embedding — mesuré, les
  embeddings captent le sujet et pas la position (NMI ≈ 0,04–0,06).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Percentile des similarités intra-thème de l'amorçage qui définit « couvert ».
# DÉRIVÉ des données, pas choisi : on prend le bas de la distribution observée au bootstrap,
# de sorte qu'un claim « aussi bien assigné que les 10 % les moins bien assignés de
# l'amorçage » compte encore comme couvert. Un littéral (0.4, 0.5…) serait un magic number
# calé sur un embedder — exactement l'erreur qui a rendu la corrélation des contributions
# inopérante lors de la bascule nomic→arctic.
COVERAGE_PERCENTILE = 10.0

# Plancher de sécurité : sous ce cosinus, on refuse de dire « couvert » même si la
# distribution d'amorçage était très étalée. Borne de bon sens, pas un calibrage.
COVERAGE_FLOOR = 0.15


@dataclass
class LiveTheme:
    """Un thème GELÉ : son centroïde sert d'aimant, son objet de clivage de cible de stance."""

    id: str
    label: str
    keywords: list[str]
    centroid: np.ndarray                 # L2-normalisé (cosinus = produit scalaire)
    cleavage: str = ""                   # proposition polaire débattable
    cleavage_justif: str = ""
    n_bootstrap: int = 0                 # claims ayant servi à le constituer

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "keywords": self.keywords[:8],
                "cleavage": self.cleavage, "cleavage_justif": self.cleavage_justif,
                "n_bootstrap": self.n_bootstrap}


@dataclass
class Assignment:
    """Un claim rattaché (ou non) à un thème, avec sa position si elle a pu être jugée."""

    claim_id: str
    turn_id: str
    seq: int
    stime: float | None
    speaker: str
    speaker_id: str | None
    text: str
    theme_id: str | None                 # None = NON COUVERT (nourrit la dérive)
    similarity: float
    stance: str | None = None            # favorable | defavorable | nuance
    confidence: str | None = None
    justif: str = ""

    @property
    def covered(self) -> bool:
        return self.theme_id is not None

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id, "turn_id": self.turn_id, "seq": self.seq,
            "stime": self.stime, "speaker": self.speaker, "speaker_id": self.speaker_id,
            "text": self.text, "theme_id": self.theme_id,
            "similarity": round(self.similarity, 4),
            "stance": self.stance, "confidence": self.confidence, "justif": self.justif,
        }


class LiveState:
    """État mutable d'une session rejouée. Sérialisable en instantané pour l'affichage."""

    def __init__(self, themes: list[LiveTheme], *, coverage_threshold: float,
                 session: str = "", topic: str = "") -> None:
        self.themes = {t.id: t for t in themes}
        self.coverage_threshold = float(coverage_threshold)
        self.session = session
        self.topic = topic
        self.assignments: list[Assignment] = []
        self.turns_seen = 0
        self.claims_seen = 0
        self._matrix = (np.stack([t.centroid for t in themes]).astype(np.float32)
                        if themes else np.zeros((0, 1), dtype=np.float32))
        self._ids = [t.id for t in themes]

    # -- assignation ---------------------------------------------------------------- #

    def nearest(self, vec: np.ndarray) -> tuple[str | None, float]:
        """Thème le plus proche + similarité. `(None, sim)` si sous le seuil de couverture.

        Renvoie la similarité MÊME quand rien n'est couvert : c'est elle qui alimente
        l'indicateur de dérive. Un « non couvert » silencieux se lirait comme « rien à
        signaler » alors que c'est le signal qui déclenche une restructuration.
        """
        if self._matrix.shape[0] == 0:
            return None, 0.0
        v = np.asarray(vec, dtype=np.float32).ravel()
        if v.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"dimension {v.shape[0]} ≠ centroïdes {self._matrix.shape[1]} — "
                "embedder différent de celui de l'amorçage."
            )
        sims = self._matrix @ v
        i = int(np.argmax(sims))
        best = float(sims[i])
        return (self._ids[i] if best >= self.coverage_threshold else None), best

    def add(self, assignment: Assignment) -> None:
        self.assignments.append(assignment)
        self.claims_seen += 1

    # -- agrégats ------------------------------------------------------------------- #

    def theme_counts(self) -> dict[str, int]:
        c = Counter(a.theme_id for a in self.assignments if a.covered)
        return {tid: c.get(tid, 0) for tid in self.themes}

    def drift(self, window: int = 40) -> dict:
        """Part de claims NON COUVERTS — globale, récente, et RELATIVE au bruit de fond.

        La fenêtre compte plus que le total : c'est elle qui dit si le débat part ailleurs,
        alors que la moyenne globale est amortie par le passé.

        ⚠️ **Un taux brut ne s'interprète pas.** Observé sur la séance retraite : 40 % de
        claims non couverts, dont l'essentiel n'était pas un sujet NOUVEAU mais du théâtre
        parlementaire — « il est fort de café de nous intenter un procès », « le
        sous-amendement no 42160 ». Lu brut, ce 40 % réclamerait une restructuration là où
        il n'y a rien à restructurer.

        D'où `baseline` : par construction du seuil (percentile bas des similarités
        d'amorçage), une part connue des claims tombe sous le seuil même quand RIEN n'a
        changé. `excess` = ce qui dépasse ce bruit de fond, et c'est le seul chiffre qui
        porte un signal de dérive THÉMATIQUE.
        """
        baseline = COVERAGE_PERCENTILE / 100.0
        if not self.assignments:
            return {"global": 0.0, "recent": 0.0, "window": window, "n_uncovered": 0,
                    "baseline": baseline, "excess": 0.0}
        uncovered = [a for a in self.assignments if not a.covered]
        recent = self.assignments[-window:]
        recent_rate = sum(1 for a in recent if not a.covered) / len(recent)
        return {
            "global": round(len(uncovered) / len(self.assignments), 4),
            "recent": round(recent_rate, 4),
            "window": window,
            "n_uncovered": len(uncovered),
            "baseline": baseline,
            "excess": round(max(0.0, recent_rate - baseline), 4),
        }

    def speaker_positions(self, *, min_claims: int = 1,
                          high_confidence_only: bool = False) -> list[dict]:
        """Position agrégée par orateur ET par thème.

        ⚠️ N'agrège QUE des stances jugées par LLM. `high_confidence_only` permet de ne
        retenir que les jugements `high` — utile pour afficher, sachant que le registre
        parlementaire expose au discours rapporté (un orateur cite la thèse adverse pour la
        réfuter), risque identifié et NON encore corrigé par un banc.
        """
        buckets: dict[tuple[str, str], Counter] = defaultdict(Counter)
        labels: dict[str, str] = {}
        for a in self.assignments:
            if not a.covered or not a.stance:
                continue
            if high_confidence_only and a.confidence != "high":
                continue
            key = (a.speaker_id or a.speaker, a.theme_id)
            buckets[key][a.stance] += 1
            labels[a.speaker_id or a.speaker] = a.speaker

        out: list[dict] = []
        for (sid, tid), counts in buckets.items():
            n = sum(counts.values())
            if n < min_claims:
                continue
            fav, opp = counts.get("favorable", 0), counts.get("defavorable", 0)
            decided = fav + opp
            # `position` n'est affichée que si une majorité NETTE se dégage ; sinon on dit
            # « partagé » plutôt que de trancher sur un écart d'une voix.
            if decided == 0:
                position = "sans position"
            elif fav > opp:
                position = "favorable"
            elif opp > fav:
                position = "défavorable"
            else:
                position = "partagé"
            out.append({
                "speaker_id": sid, "speaker": labels.get(sid, sid), "theme_id": tid,
                "n_claims": n, "favorable": fav, "defavorable": opp,
                "nuance": counts.get("nuance", 0), "position": position,
            })
        out.sort(key=lambda r: (-r["n_claims"], r["speaker"]))
        return out

    def theme_opinion(self) -> list[dict]:
        """Répartition d'opinion par thème (favorable / défavorable / nuance)."""
        buckets: dict[str, Counter] = defaultdict(Counter)
        for a in self.assignments:
            if a.covered and a.stance:
                buckets[a.theme_id][a.stance] += 1
        out = []
        for tid, theme in self.themes.items():
            c = buckets.get(tid, Counter())
            n = sum(c.values())
            out.append({
                "theme_id": tid, "label": theme.label, "cleavage": theme.cleavage,
                "n_claims": n,
                "favorable": c.get("favorable", 0),
                "defavorable": c.get("defavorable", 0),
                "nuance": c.get("nuance", 0),
                "part_favorable": round(c.get("favorable", 0) / n, 3) if n else None,
            })
        out.sort(key=lambda r: -r["n_claims"])
        return out

    # -- instantané ------------------------------------------------------------------ #

    def snapshot(self, *, last_turns: int = 12) -> dict:
        """Vue sérialisable pour l'affichage live."""
        recent = self.assignments[-last_turns * 3:]
        return {
            "session": self.session,
            "topic": self.topic,
            "turns_seen": self.turns_seen,
            "claims_seen": self.claims_seen,
            "coverage_threshold": round(self.coverage_threshold, 4),
            "themes": [t.to_dict() for t in self.themes.values()],
            "theme_counts": self.theme_counts(),
            "theme_opinion": self.theme_opinion(),
            "drift": self.drift(),
            "speakers": self.speaker_positions(min_claims=2)[:40],
            "recent_claims": [a.to_dict() for a in recent][-30:],
        }

    def write_snapshot(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)                       # atomique : le lecteur ne voit jamais un demi-état


def derive_coverage_threshold(vecs: np.ndarray, centroids: np.ndarray,
                              membership: np.ndarray) -> float:
    """Seuil de couverture DÉRIVÉ de l'amorçage, pas choisi.

    Pour chaque claim d'amorçage on prend le cosinus à SON centroïde ; le seuil est le
    percentile bas de cette distribution. Traduction : « est couvert un claim au moins aussi
    bien rattaché que les 10 % les moins bien rattachés de l'amorçage ».

    Pourquoi dériver plutôt que fixer : l'échelle des cosinus dépend de l'embedder. Un
    littéral se périme silencieusement à la première bascule de modèle — c'est exactement ce
    qui est arrivé au seuil de corrélation des contributions (0,68 nomic laissé en place
    sous arctic, fonction inopérante une semaine).
    """
    if len(vecs) == 0 or centroids.shape[0] == 0:
        return COVERAGE_FLOOR
    sims = np.einsum("ij,ij->i", vecs.astype(np.float32),
                     centroids[membership].astype(np.float32))
    return float(max(COVERAGE_FLOOR, np.percentile(sims, COVERAGE_PERCENTILE)))


__all__ = [
    "LiveTheme", "Assignment", "LiveState", "derive_coverage_threshold",
    "COVERAGE_PERCENTILE", "COVERAGE_FLOOR",
]

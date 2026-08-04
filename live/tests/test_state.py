"""État live : assignation, dérive, agrégation par orateur.

Aucun LLM, aucun réseau : ces règles sont géométriques ou comptables, donc testables
directement. Ce qui relève du jugement (la stance) est INJECTÉ dans les fixtures — le
module ne doit jamais déduire une position d'une proximité d'embedding.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from live.state import (
    COVERAGE_FLOOR,
    Assignment,
    LiveState,
    LiveTheme,
    derive_coverage_threshold,
)


def _theme(tid: str, vec: list[float], label: str = "thème", cleavage: str = "faire X") -> LiveTheme:
    v = np.array(vec, dtype=np.float32)
    return LiveTheme(id=tid, label=label, keywords=["a", "b"],
                     centroid=v / np.linalg.norm(v), cleavage=cleavage)


def _state(threshold: float = 0.5) -> LiveState:
    return LiveState([_theme("t0", [1, 0, 0]), _theme("t1", [0, 1, 0])],
                     coverage_threshold=threshold, session="S", topic="Sujet")


def _assign(state: LiveState, *, speaker="M. A", speaker_id="PA1", theme_id="t0",
            stance=None, confidence="high", sim=0.9, seq=1) -> Assignment:
    a = Assignment(claim_id=f"c{seq}", turn_id=f"T{seq}", seq=seq, stime=float(seq),
                   speaker=speaker, speaker_id=speaker_id, text="un claim",
                   theme_id=theme_id, similarity=sim, stance=stance, confidence=confidence)
    state.add(a)
    return a


# --------------------------------------------------------------------------------- #
# 1. Assignation au plus proche centroïde
# --------------------------------------------------------------------------------- #

def test_assignation_au_plus_proche():
    s = _state()
    assert s.nearest(np.array([1, 0, 0], dtype=np.float32))[0] == "t0"
    assert s.nearest(np.array([0, 1, 0], dtype=np.float32))[0] == "t1"


def test_sous_le_seuil_rien_nest_couvert():
    """Un claim loin de tout thème est NON COUVERT — c'est le signal de dérive."""
    s = _state(threshold=0.9)
    theme_id, sim = s.nearest(np.array([0.7, 0.7, 0.0], dtype=np.float32))
    assert theme_id is None
    assert sim == pytest.approx(0.7, abs=1e-5)


def test_la_similarite_est_rendue_meme_sans_couverture():
    """Un « non couvert » muet se lirait comme « rien à signaler ». Il faut le chiffre."""
    s = _state(threshold=0.99)
    _, sim = s.nearest(np.array([1, 0, 0], dtype=np.float32))
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_dimension_incompatible_leve():
    """Changer d'embedder entre amorçage et live doit LEVER, pas produire un cosinus faux."""
    s = _state()
    with pytest.raises(ValueError, match="dimension"):
        s.nearest(np.zeros(8, dtype=np.float32))


def test_aucun_theme_ne_plante_pas():
    vide = LiveState([], coverage_threshold=0.5)
    assert vide.nearest(np.array([1.0], dtype=np.float32)) == (None, 0.0)


# --------------------------------------------------------------------------------- #
# 2. Seuil de couverture DÉRIVÉ (jamais un littéral)
# --------------------------------------------------------------------------------- #

def test_le_seuil_est_derive_de_lamorcage():
    """Le seuil suit l'échelle de cosinus de l'embedder utilisé, au lieu d'être figé.

    Un littéral se périme silencieusement à la première bascule de modèle — vécu avec le
    seuil de corrélation des contributions (0,68 nomic laissé en place sous arctic).
    """
    rng = np.random.RandomState(0)
    centroids = np.eye(3, dtype=np.float32)
    vecs = np.zeros((60, 3), dtype=np.float32)
    membership = np.zeros(60, dtype=int)
    for i in range(60):
        c = i % 3
        membership[i] = c
        v = centroids[c] + 0.25 * rng.randn(3).astype(np.float32)
        vecs[i] = v / np.linalg.norm(v)
    seuil = derive_coverage_threshold(vecs, centroids, membership)
    sims = np.einsum("ij,ij->i", vecs, centroids[membership])
    assert COVERAGE_FLOOR <= seuil <= float(sims.max())
    # ~10 % des claims d'amorçage tombent sous le seuil, par construction du percentile.
    assert 0.02 <= float((sims < seuil).mean()) <= 0.2


def test_le_seuil_a_un_plancher():
    """Une distribution très étalée ne doit pas produire un seuil qui accepte n'importe quoi."""
    centroids = np.eye(2, dtype=np.float32)
    vecs = np.array([[1, 0], [0, 1]], dtype=np.float32)
    seuil = derive_coverage_threshold(vecs * 0.01, centroids, np.array([0, 1]))
    assert seuil >= COVERAGE_FLOOR


def test_amorcage_vide_donne_le_plancher():
    assert derive_coverage_threshold(np.zeros((0, 3), dtype=np.float32),
                                     np.zeros((0, 3), dtype=np.float32),
                                     np.zeros(0, dtype=int)) == COVERAGE_FLOOR


# --------------------------------------------------------------------------------- #
# 3. Dérive
# --------------------------------------------------------------------------------- #

def test_la_derive_recente_prime_sur_la_globale():
    """C'est la fenêtre récente qui dit si le débat part ailleurs ; la globale est amortie."""
    s = _state()
    for i in range(50):
        _assign(s, theme_id="t0", seq=i)
    for i in range(50, 60):
        _assign(s, theme_id=None, seq=i)
    d = s.drift(window=10)
    assert d["recent"] == 1.0
    assert d["global"] < 0.2
    assert d["n_uncovered"] == 10


def test_derive_sans_assignation():
    assert _state().drift()["global"] == 0.0


# --------------------------------------------------------------------------------- #
# 4. Positions par orateur — uniquement depuis des stances JUGÉES
# --------------------------------------------------------------------------------- #

def test_position_majoritaire_par_orateur_et_theme():
    s = _state()
    for i in range(3):
        _assign(s, stance="favorable", seq=i)
    _assign(s, stance="defavorable", seq=9)
    ligne = s.speaker_positions()[0]
    assert ligne["position"] == "favorable"
    assert (ligne["favorable"], ligne["defavorable"]) == (3, 1)


def test_egalite_dit_partage_et_ne_tranche_pas():
    """Trancher sur un écart nul fabriquerait une opinion qui n'existe pas."""
    s = _state()
    _assign(s, stance="favorable", seq=1)
    _assign(s, stance="defavorable", seq=2)
    assert s.speaker_positions()[0]["position"] == "partagé"


def test_que_des_nuances_donne_sans_position():
    s = _state()
    _assign(s, stance="nuance", seq=1)
    _assign(s, stance="nuance", seq=2)
    assert s.speaker_positions()[0]["position"] == "sans position"


def test_un_claim_sans_stance_nentre_pas_dans_lagregat():
    """Aucune position ne doit être déduite de la seule proximité géométrique."""
    s = _state()
    _assign(s, stance=None, seq=1)
    assert s.speaker_positions() == []


def test_un_claim_non_couvert_nentre_pas_dans_lagregat():
    s = _state()
    _assign(s, theme_id=None, stance="favorable", seq=1)
    assert s.speaker_positions() == []


def test_filtre_haute_confiance():
    """Le registre parlementaire expose au discours rapporté : pouvoir n'afficher que le net."""
    s = _state()
    _assign(s, stance="favorable", confidence="low", seq=1)
    _assign(s, stance="favorable", confidence="high", seq=2)
    assert s.speaker_positions(high_confidence_only=True)[0]["n_claims"] == 1
    assert s.speaker_positions()[0]["n_claims"] == 2


def test_orateurs_distincts_par_id_pas_par_libelle():
    s = _state()
    _assign(s, speaker="M. Martin", speaker_id="PA1", stance="favorable", seq=1)
    _assign(s, speaker="M. Martin", speaker_id="PA2", stance="defavorable", seq=2)
    assert len(s.speaker_positions()) == 2


def test_le_meme_orateur_est_separe_par_theme():
    s = _state()
    _assign(s, theme_id="t0", stance="favorable", seq=1)
    _assign(s, theme_id="t1", stance="defavorable", seq=2)
    lignes = {r["theme_id"]: r["position"] for r in s.speaker_positions()}
    assert lignes == {"t0": "favorable", "t1": "défavorable"}


# --------------------------------------------------------------------------------- #
# 5. Opinion par thème & instantané
# --------------------------------------------------------------------------------- #

def test_opinion_par_theme():
    s = _state()
    _assign(s, theme_id="t0", stance="favorable", seq=1)
    _assign(s, theme_id="t0", stance="favorable", seq=2)
    _assign(s, theme_id="t0", stance="defavorable", seq=3)
    t0 = [r for r in s.theme_opinion() if r["theme_id"] == "t0"][0]
    assert (t0["favorable"], t0["defavorable"]) == (2, 1)
    assert t0["part_favorable"] == pytest.approx(2 / 3, abs=1e-3)


def test_theme_sans_claim_a_une_part_nulle_pas_zero():
    """`None` ≠ 0 % : un dénominateur inventé mentirait sur un thème vide."""
    t1 = [r for r in _state().theme_opinion() if r["theme_id"] == "t1"][0]
    assert t1["n_claims"] == 0 and t1["part_favorable"] is None


def test_instantane_serialisable(tmp_path):
    s = _state()
    _assign(s, stance="favorable", seq=1)
    _assign(s, stance="favorable", seq=2)
    p = tmp_path / "snap.json"
    s.write_snapshot(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["topic"] == "Sujet"
    assert len(data["themes"]) == 2
    assert data["claims_seen"] == 2
    assert data["drift"]["global"] == 0.0
    assert data["speakers"][0]["position"] == "favorable"


def test_ecriture_dinstantane_atomique(tmp_path):
    """Le lecteur ne doit jamais voir un demi-état : aucun `.tmp` ne subsiste."""
    s = _state()
    p = tmp_path / "sub" / "snap.json"
    s.write_snapshot(p)
    s.write_snapshot(p)
    assert p.exists()
    assert list(p.parent.glob("*.tmp")) == []
    json.loads(p.read_text(encoding="utf-8"))

"""M2 — hero représentatif : score composite (pureté × couverture × représentativité ×
lisibilité) au build, exposé en `hero_avis_id` dans `theme_dict`. Générique.
"""
from types import SimpleNamespace

import numpy as np

from backend.analysis import ThemeNode, theme_dict, _hero_avis, _lisibilite


class _FakeAvis:
    def __init__(self, id_: str, text: str):
        self.id = id_
        self.text = text


def _node(members):
    return ThemeNode(
        id="n0", parent_id=None, depth=0, members=list(members), centroid=np.zeros(3),
        dispersion=0.1, consensus=0.8, weight=1.0, n_claims=len(members), n_avis=2,
    )


def test_lisibilite_window():
    assert _lisibilite("x" * 400) == 1.0            # dans la fenêtre 200–1200
    assert _lisibilite("x" * 100) == 100 / 200      # trop court → pénalisé
    assert _lisibilite("x" * 2400) == 1200 / 2400   # trop long → pénalisé
    assert _lisibilite("") == 0.0


def test_theme_dict_exposes_hero_avis_id():
    """Contrat : theme_dict expose `hero_avis_id` (None par défaut)."""
    assert theme_dict(_node([0]))["hero_avis_id"] is None
    n = _node([0])
    n.hero_avis_id = "a42"
    assert theme_dict(n)["hero_avis_id"] == "a42"


def test_hero_prefers_pure_covered_representative_avis():
    """Avis A (3 claims tous dans le thème, centraux) l'emporte sur B (1 claim, périphérique)."""
    # claims 0,1,2 -> avis A (idx 0) ; claims 3,4,5 -> avis B (idx 1)
    claim_owner = [0, 0, 0, 1, 1, 1]
    vecs = np.array([[1, 0, 0]] * 3 + [[0, 1, 0]] * 3, dtype=float)
    avis = [_FakeAvis("a", "x" * 400), _FakeAvis("b", "y" * 400)]
    prepared = SimpleNamespace(claim_vecs=vecs, claim_owner=claim_owner, avis=avis)
    avis_total = np.bincount(np.asarray(claim_owner), minlength=2)  # [3, 3]
    # thème = les 3 claims de A + 1 claim de B
    hero = _hero_avis(_node([0, 1, 2, 3]), prepared, avis_total)
    assert hero == "a"


def test_hero_purity_penalises_scattered_avis():
    """À couverture égale, l'avis PUR (tous ses claims dans le thème) l'emporte."""
    # A (idx0) : 2 claims, tous dans le thème (pur). B (idx1) : 2 claims dans le thème
    # mais 6 au total ailleurs (impur). Vecteurs identiques → seule la pureté départage.
    claim_owner = [0, 0, 1, 1, 1, 1, 1, 1]
    vecs = np.array([[1, 0, 0]] * 8, dtype=float)
    avis = [_FakeAvis("a", "x" * 400), _FakeAvis("b", "y" * 400)]
    prepared = SimpleNamespace(claim_vecs=vecs, claim_owner=claim_owner, avis=avis)
    avis_total = np.bincount(np.asarray(claim_owner), minlength=2)  # [2, 6]
    # thème = claims 0,1 (A) + 2,3 (B) → couverture 2 chacun ; A pur (2/2) vs B impur (2/6)
    hero = _hero_avis(_node([0, 1, 2, 3]), prepared, avis_total)
    assert hero == "a"


def test_hero_none_on_empty_theme():
    prepared = SimpleNamespace(claim_vecs=np.zeros((0, 3)), claim_owner=[], avis=[])
    assert _hero_avis(_node([]), prepared, np.zeros(0, dtype=int)) is None

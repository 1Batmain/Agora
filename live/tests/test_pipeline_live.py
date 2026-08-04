"""Pipeline live — découpage pour extraction et composition des couches.

Ce qui touche au LLM n'est pas testé ici (c'est le rôle du rejeu réel) ; ce fichier
verrouille les règles DÉTERMINISTES sur lesquelles tout le reste repose.
"""

from __future__ import annotations

import pytest

from live.pipeline import MAX_CHUNK_CHARS, LiveConfig, split_for_extraction


# --------------------------------------------------------------------------------- #
# Découpage pour extraction — trouvé nécessaire sur données réelles
# --------------------------------------------------------------------------------- #

def test_un_tour_court_nest_pas_decoupe():
    texte = "Une intervention brève. Deux phrases seulement."
    assert split_for_extraction(texte) == [texte]


def test_un_tour_long_est_decoupe_sous_la_borne():
    """Mesuré : le plus long tour de la séance retraite fait 20 655 caractères — une
    intervention ministérielle interrompue 40 fois puis recollée. Sans découpage, un lot
    d'amorçage dépassait 100 000 caractères d'entrée pour 3 200 tokens de sortie."""
    texte = "Voici une phrase de longueur raisonnable sur les retraites. " * 200
    chunks = split_for_extraction(texte)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


def test_le_decoupage_ne_perd_pas_de_texte():
    texte = "Phrase une. Phrase deux ! Phrase trois ? Phrase quatre. " * 80
    chunks = split_for_extraction(texte)
    recolle = " ".join(chunks)
    # Mêmes mots, dans le même ordre : le découpage ne fait que déplacer des espaces.
    assert recolle.split() == texte.split()


def test_le_decoupage_respecte_les_phrases():
    """Couper au milieu d'une proposition garderait l'ancrage verbatim mais détruirait
    le sens — ce qui est pire, parce que ça ne se voit pas."""
    phrases = [f"Ceci est la phrase numéro {i} de cette intervention parlementaire." for i in range(90)]
    chunks = split_for_extraction(" ".join(phrases))
    for c in chunks:
        assert c.strip().endswith((".", "!", "?", "…")), f"fragment coupé en plein milieu : {c[-60:]!r}"


def test_une_phrase_plus_longue_que_la_borne_nest_pas_coupee():
    """On préfère un fragment trop grand à un claim tronqué."""
    monstre = "mot " * (MAX_CHUNK_CHARS // 2)
    chunks = split_for_extraction(monstre)
    assert len(chunks) == 1 and len(chunks[0]) > MAX_CHUNK_CHARS


@pytest.mark.parametrize("vide", ["", "   ", "\n"])
def test_texte_vide(vide):
    assert split_for_extraction(vide) == []


# --------------------------------------------------------------------------------- #
# Configuration — les modèles viennent du PROFIL, jamais du module live
# --------------------------------------------------------------------------------- #

def test_la_config_lit_le_profil():
    cfg = LiveConfig(profile_name="live")
    assert cfg.profile.name == "live"
    # Le profil live est explicitement NON VALIDÉ : seul `extract` repose sur une mesure.
    assert cfg.profile.validated is False


def test_le_cache_live_est_isole(tmp_path):
    """Le pipeline live ne doit écrire nulle part dans les caches servis."""
    cfg = LiveConfig(cache_dir=tmp_path / "live")
    assert cfg.emb_path.parent == tmp_path / "live"
    assert "backend/cache" not in str(cfg.emb_path)


def test_le_module_nimporte_rien_du_backend():
    """Isolation stricte : `live/` compose des couches de `pipeline/`, jamais de `backend/`."""
    import pathlib
    racine = pathlib.Path(__file__).resolve().parents[1]
    for module in ("pipeline.py", "state.py", "replay.py", "transcript.py"):
        contenu = (racine / module).read_text(encoding="utf-8")
        assert "import backend" not in contenu, module
        assert "from backend" not in contenu, module

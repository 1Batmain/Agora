"""Transcripts AN → tours de parole : les règles sont mesurées, donc testables.

Chaque test correspond à un constat chiffré de `.agent/notes/LIVE_DATA_RECON.md` ou à un
défaut observé sur données réelles (`LIVE_PROBE_CLAIMS.md`). Fixtures XML en ligne : le
corpus réel (311 Mo) n'a pas à être présent pour que la suite tourne.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from live.transcript import (
    MIN_CHARS,
    Turn,
    is_chair,
    parse_session,
    read_session_meta,
    split_reactions,
    stitch,
    write_jsonl,
)

LONG = "Il faut abroger cette réforme injuste et rétablir un âge de départ soutenable. " * 4


def _xml(paragraphes: str) -> str:
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <compteRendu>
          <uid>CRTEST001</uid>
          <metadonnees>
            <dateSeance>20241128090000000</dateSeance>
            <dateSeanceJour>jeudi 28 novembre 2024</dateSeanceJour>
            <numSeance>55</numSeance>
            <session>Session ordinaire 2024-2025</session>
            <presidentSeance>Présidence de Mme la présidente</presidentSeance>
          </metadonnees>
          <contenu>
            <point>
              <intitule>Abrogation de la retraite à 64 ans</intitule>
              {paragraphes}
            </point>
          </contenu>
        </compteRendu>
    """)


def _p(seq, code, texte, *, orateur="M. Jean Dupont", acteur="PA111111",
       role=None, style="NORMAL", stime=None) -> str:
    attrs = (f'ordre_absolu_seance="{seq}" code_grammaire="{code}" code_style="{style}" '
             f'id_syceron="s{seq}"')
    if acteur:
        attrs += f' id_acteur="{acteur}"'
    if role:
        attrs += f' roledebat="{role}"'
    st = f' stime="{stime}"' if stime is not None else ""
    return (f'<paragraphe {attrs}><orateurs>{orateur} 111111</orateurs>'
            f'<texte{st}>{texte}</texte></paragraphe>')


# --------------------------------------------------------------------------------- #
# 1. Réactions vs SIGLES — la règle la plus délicate
# --------------------------------------------------------------------------------- #

def test_les_reactions_sont_extraites_du_verbatim():
    txt = "Nous devons agir. (Applaudissements sur les bancs du groupe LFI-NFP.) C'est urgent."
    clean, reactions = split_reactions(txt)
    assert "Applaudissements" not in clean
    assert clean == "Nous devons agir. C'est urgent."
    assert reactions == ["Applaudissements sur les bancs du groupe LFI-NFP."]


@pytest.mark.parametrize("sigle", ["PLFSS", "CSG", "ZAN", "Drees", "TPE et PME", "CEDH"])
def test_les_sigles_prononces_sont_preserves(sigle):
    """~19 % des parenthèses du corpus sont des sigles DITS par l'orateur.

    Les retirer casserait l'ancrage verbatim — un claim ne se retrouverait plus dans le
    texte source. C'est pourquoi l'inconnu est CONSERVÉ, jamais supprimé par défaut.
    """
    txt = f"Le projet de loi de financement de la sécurité sociale ({sigle}) prévoit ceci."
    clean, reactions = split_reactions(txt)
    assert f"({sigle})" in clean
    assert reactions == []


def test_interjection_citee_traitee_comme_reaction():
    clean, reactions = split_reactions("Vous mentez ! (« Ah ! » sur les bancs du groupe RN.)")
    assert "Ah !" not in clean
    assert len(reactions) == 1


def test_reaction_et_sigle_dans_la_meme_phrase():
    """Le cas mixte : on retire l'une SANS toucher à l'autre."""
    clean, reactions = split_reactions(
        "La contribution sociale généralisée (CSG) doit baisser. (Exclamations.)")
    assert "(CSG)" in clean
    assert "Exclamations" not in clean
    assert reactions == ["Exclamations."]


# --------------------------------------------------------------------------------- #
# 2. Filtres — procédure et perchoir
# --------------------------------------------------------------------------------- #

def test_procedure_ecartee_fond_conserve(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", LONG)
        + _p(2, "SCRUT_PUBLIC", "Je mets aux voix les amendements identiques. " * 10)
        + _p(3, "SUSP_SEANCE_2_1", "La séance est suspendue. " * 20)
        + _p(4, "INTERRUPTION", "Très bien !")
    ), encoding="utf-8")
    turns = parse_session(src)
    assert len(turns) == 1
    assert turns[0].seq == 1


def test_perchoir_filtre_meme_sans_lattribut_roledebat(tmp_path):
    """1 629 tours (1,7 %) portent « Mme la présidente » SANS `roledebat` — la fuite mesurée.

    Deux des trois replis d'ancrage observés à la sonde venaient de là.
    """
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", "Pensez à l'image que nous donnons à ceux qui nous regardent. " * 6,
           orateur="Mme la présidente", acteur="PA721908")          # PAS de roledebat
        + _p(2, "PAROLE_GENERALE", LONG, orateur="M. Autre", acteur="PA222222")
    ), encoding="utf-8")
    turns = parse_session(src)
    assert [t.speaker for t in turns] == ["M. Autre"]


def test_perchoir_filtre_par_lattribut(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", "L'ordre du jour appelle la discussion. " * 10,
           orateur="M. Président de séance", acteur="PA333333", role="president")
    ), encoding="utf-8")
    assert parse_session(src) == []


@pytest.mark.parametrize("label,role,attendu", [
    ("Mme la présidente", "", True),
    ("M. le président", "", True),
    ("M. Jean Dupont", "president", True),
    ("Mme Mathilde Panot", "", False),
    ("M. Ugo Bernalicis", "rapporteur", False),
])
def test_detection_du_perchoir(label, role, attendu):
    assert is_chair(label, role) is attendu


def test_didascalie_en_italique_ecartee(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", "(La séance est reprise à quinze heures.) " * 10,
           style="Info Italiques", orateur="", acteur="")
        + _p(2, "PAROLE_GENERALE", LONG)
    ), encoding="utf-8")
    assert len(parse_session(src)) == 1


# --------------------------------------------------------------------------------- #
# 3. Recollage — 36,9 % du corpus
# --------------------------------------------------------------------------------- #

def test_paragraphes_consecutifs_du_meme_orateur_recolles(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", "Première partie de mon intervention. " * 5, stime="100.5")
        + _p(2, "INTERRUPTION", "Très bien !", orateur="M. Gêneur", acteur="PA999999")
        + _p(3, "PAROLE_GENERALE", "…seconde partie de mon intervention. " * 5, stime="140.0")
    ), encoding="utf-8")
    turns = parse_session(src)
    assert len(turns) == 1, "les deux moitiés du même orateur doivent former UN tour"
    assert "Première partie" in turns[0].text and "seconde partie" in turns[0].text
    assert turns[0].paragraph_ids == ["s1", "s3"]
    assert turns[0].stime == 100.5, "le tour porte le timecode de son DÉBUT"


def test_la_marque_de_reprise_est_retiree():
    """Les 6,9 % de tours commençant par « … » sont des reprises typographiques."""
    turns = stitch([
        {"session": "S", "seq": 1, "speaker": "M. A", "speaker_id": "PA1", "role": "",
         "stime": None, "text": "…refusent de dire cette vérité simple. " * 8, "pid": "s1"},
    ])
    assert not turns[0].text.startswith("…")


def test_orateurs_differents_ne_sont_pas_fusionnes(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", LONG, orateur="M. Un", acteur="PA1")
        + _p(2, "PAROLE_GENERALE", LONG, orateur="Mme Deux", acteur="PA2")
    ), encoding="utf-8")
    turns = parse_session(src)
    assert len(turns) == 2
    assert [t.speaker_id for t in turns] == ["PA1", "PA2"]


def test_homonymes_distingues_par_id_acteur():
    """Le libellé peut se répéter ; `id_acteur` (90,1 % du corpus) fait foi."""
    turns = stitch([
        {"session": "S", "seq": 1, "speaker": "M. Martin", "speaker_id": "PA1", "role": "",
         "stime": None, "text": LONG, "pid": "s1"},
        {"session": "S", "seq": 2, "speaker": "M. Martin", "speaker_id": "PA2", "role": "",
         "stime": None, "text": LONG, "pid": "s2"},
    ])
    assert len(turns) == 2


def test_le_recollage_precede_le_filtre_de_longueur():
    """L'ORDRE est la règle : filtrer avant recoller jetterait des fragments qui, réunis,
    dépassent le seuil. C'est le mode de panne que ce test verrouille."""
    moitie = "a" * (MIN_CHARS // 2 + 10)
    turns = stitch([
        {"session": "S", "seq": 1, "speaker": "M. A", "speaker_id": "PA1", "role": "",
         "stime": 1.0, "text": moitie, "pid": "s1"},
        {"session": "S", "seq": 2, "speaker": "M. A", "speaker_id": "PA1", "role": "",
         "stime": 2.0, "text": moitie, "pid": "s2"},
    ])
    assert len(turns) == 1 and turns[0].n_chars >= MIN_CHARS


def test_un_paragraphe_purement_reactionnel_ne_coupe_pas_le_tour():
    """Une salve d'applaudissements n'interrompt pas l'intervention : le tour reste UN."""
    turns = stitch([
        {"session": "S", "seq": 1, "speaker": "M. A", "speaker_id": "PA1", "role": "",
         "stime": 1.0, "text": "Première moitié de mon propos. " * 5, "pid": "s1"},
        {"session": "S", "seq": 2, "speaker": "M. A", "speaker_id": "PA1", "role": "",
         "stime": 2.0, "text": "(Applaudissements sur les bancs du groupe SOC.)", "pid": "s2"},
        {"session": "S", "seq": 3, "speaker": "M. A", "speaker_id": "PA1", "role": "",
         "stime": 3.0, "text": "Seconde moitié de mon propos. " * 5, "pid": "s3"},
    ])
    assert len(turns) == 1
    assert any("Applaudissements" in r for r in turns[0].reactions)
    assert "Applaudissements" not in turns[0].text


# --------------------------------------------------------------------------------- #
# 3 bis. Frontières — la règle qui évite de fusionner DEUX interventions distinctes
# --------------------------------------------------------------------------------- #

def test_le_perchoir_rompt_lintervention(tmp_path):
    """« La parole est à M. X » termine une intervention et en ouvre une autre.

    Sans cette règle, filtrer le perchoir AVANT de recoller rendait adjacents deux passages
    du même orateur séparés par tout un débat. Le perchoir n'est pas émis, mais il SÉPARE.
    """
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", LONG, orateur="M. Dupont", acteur="PA1")
        + _p(2, "PAROLE_GENERALE", "La parole est à M. Dupont, pour deux minutes. " * 6,
             orateur="Mme la présidente", acteur="PA721908")
        + _p(3, "PAROLE_GENERALE", LONG, orateur="M. Dupont", acteur="PA1")
    ), encoding="utf-8")
    turns = parse_session(src)
    assert len(turns) == 2, "le perchoir doit séparer deux interventions du même orateur"
    assert [t.seq for t in turns] == [1, 3]


@pytest.mark.parametrize("code", ["SCRUT_PUBLIC", "SUSP_SEANCE_2_1", "ADOP_ART", "VOTE_1"])
def test_un_evenement_de_procedure_rompt_lintervention(tmp_path, code):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", LONG, orateur="M. Dupont", acteur="PA1")
        + _p(2, code, "Je mets aux voix. " * 20, orateur="M. Dupont", acteur="PA1")
        + _p(3, "PAROLE_GENERALE", LONG, orateur="M. Dupont", acteur="PA1")
    ), encoding="utf-8")
    assert len(parse_session(src)) == 2


def test_une_interruption_ne_rompt_PAS_lintervention(tmp_path):
    """La distinction qui compte : un chahut n'interrompt pas le raisonnement, un scrutin si.

    Mesuré : un ministre est interrompu 40 fois d'affilée pendant UNE seule intervention.
    """
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", LONG, orateur="M. Ministre", acteur="PA1")
        + _p(2, "INTERRUPTION", "Ça suffit !", orateur="M. Gêneur", acteur="PA2")
        + _p(3, "PAROLE_GENERALE", LONG, orateur="M. Ministre", acteur="PA1")
    ), encoding="utf-8")
    assert len(parse_session(src)) == 1


def test_aucun_tour_nenjambe_une_frontiere(tmp_path):
    """INVARIANT vérifié sur données réelles (5 862 tours, 60 séances : 0 violation).

    Le reproduire ici en fixture le protège des régressions sans exiger le corpus (311 Mo).
    """
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(1, "PAROLE_GENERALE", LONG, orateur="M. A", acteur="PA1")
        + _p(2, "INTERRUPTION", "Oh !", orateur="M. B", acteur="PA2")
        + _p(3, "PAROLE_GENERALE", LONG, orateur="M. A", acteur="PA1")
        + _p(4, "SCRUT_PUBLIC", "Il est procédé au scrutin. " * 12, orateur="Mme la présidente")
        + _p(5, "PAROLE_GENERALE", LONG, orateur="M. A", acteur="PA1")
    ), encoding="utf-8")
    turns = parse_session(src)
    frontieres = {4}
    for t in turns:
        seqs = [int(p.lstrip("s")) for p in t.paragraph_ids]
        assert not any(min(seqs) < b < max(seqs) for b in frontieres), (
            f"le tour {t.id} enjambe une frontière : {t.paragraph_ids}")
    assert [t.seq for t in turns] == [1, 5]


def test_tours_trop_courts_ecartes_apres_recollage(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(_p(1, "PAROLE_GENERALE", "Trop court.")), encoding="utf-8")
    assert parse_session(src) == []


# --------------------------------------------------------------------------------- #
# 4. Métadonnées, ordre, sérialisation
# --------------------------------------------------------------------------------- #

def test_metadonnees_de_seance(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(_p(1, "PAROLE_GENERALE", LONG)), encoding="utf-8")
    meta = read_session_meta(src)
    assert meta.uid == "CRTEST001"
    assert meta.date_label == "jeudi 28 novembre 2024"
    assert meta.num_seance == "55"
    assert meta.topics == ["Abrogation de la retraite à 64 ans"]


def test_les_tours_sortent_dans_lordre_de_seance(tmp_path):
    """L'ordre de séance EST la ligne de temps du rejeu : il ne dépend pas de l'ordre XML."""
    src = tmp_path / "cr.xml"
    src.write_text(_xml(
        _p(9, "PAROLE_GENERALE", LONG, orateur="M. Neuf", acteur="PA9")
        + _p(3, "PAROLE_GENERALE", LONG, orateur="M. Trois", acteur="PA3")
    ), encoding="utf-8")
    assert [t.seq for t in parse_session(src)] == [3, 9]


def test_le_timecode_est_conserve(tmp_path):
    src = tmp_path / "cr.xml"
    src.write_text(_xml(_p(1, "PAROLE_GENERALE", LONG, stime="6070.84")), encoding="utf-8")
    assert parse_session(src)[0].stime == 6070.84


def test_ecriture_jsonl(tmp_path):
    turns = [Turn(id="S:1", session="S", seq=1, speaker="M. A", speaker_id="PA1",
                  role="", stime=1.0, text="un texte", reactions=["Applaudissements."],
                  paragraph_ids=["s1"])]
    out = tmp_path / "sub" / "turns.jsonl"
    assert write_jsonl(turns, out) == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["speaker_id"] == "PA1" and row["reactions"] == ["Applaudissements."]


def test_le_module_nimporte_rien_du_backend():
    """Isolation : le pipeline live ne doit pas pouvoir casser l'Agora servi."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "transcript.py"
    contenu = src.read_text(encoding="utf-8")
    assert "import backend" not in contenu and "from backend" not in contenu


# --------------------------------------------------------------------------------- #
# 5. Réactions LONGUES et en cascade (défaut trouvé sur données réelles)
# --------------------------------------------------------------------------------- #

def test_reaction_en_cascade_longue_est_extraite():
    """Les réactions s'enchaînent dans une seule parenthèse et dépassent 200 caractères.

    Une borne trop basse les laissait polluer le verbatim — observé sur un tour réel de la
    séance retraite. Mesuré : 0,3 % des parenthèses dépassent 200 c (max 446), et 99 %
    d'entre elles sont des réactions.
    """
    longue = ("Applaudissements sur plusieurs bancs des groupes Dem, EPR et HOR. – "
              "Exclamations sur plusieurs bancs des groupes LFI-NFP et EcoS. – "
              "M. Manuel Bompard agite le revers de sa veste en signe de protestation "
              "tandis que plusieurs députés se lèvent et quittent l'hémicycle.")
    assert len(longue) > 200
    clean, reactions = split_reactions(f"Le cirque, c'est vous ! ({longue}) Je poursuis.")
    assert clean == "Le cirque, c'est vous ! Je poursuis."
    assert len(reactions) == 1


def test_une_parenthese_demesuree_est_laissee_intacte():
    """Garde-fou : au-delà de la borne, on NE TOUCHE À RIEN plutôt que de couper au hasard."""
    enorme = "Applaudissements " * 60          # > 500 caractères
    texte = f"Un propos. ({enorme}) La suite."
    clean, reactions = split_reactions(texte)
    assert reactions == [] and clean == " ".join(texte.split())

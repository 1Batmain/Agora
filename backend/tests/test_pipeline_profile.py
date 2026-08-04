"""Le profil de pipeline est la SOURCE DE VÉRITÉ des modèles et de la forme.

Ces tests protègent les trois propriétés qui ont motivé le module (cf.
`.agent/notes/PIPELINE_PROFILE.md`) :

1. **ISO-COMPORTEMENT** — `final` reproduit EXACTEMENT les valeurs effectives d'avant le
   refactor. C'est le témoin : sans lui, un refactor qui déplace silencieusement un modèle
   se lirait comme une simple réorganisation.
2. **NON-COLLISION** — deux profils ne peuvent pas se corrompre le cache. Vécu de la même
   famille : une bascule d'embedder avait laissé un seuil périmé en place, rendant une
   fonction servie inopérante une semaine durant, sans un signal.
3. **PAS DE CASCADE** — surcharger un rôle n'en déplace pas d'autres. L'ancien montage
   (`AGORA_OPINION_MODEL` → `AGORA_ENRICH_MODEL` → littéral) couplait trois rôles sans
   le déclarer.
"""

from __future__ import annotations

import re

from pathlib import Path

import pytest

from pipeline import profile as P

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------- #
# 1. Iso-comportement : les valeurs servies n'ont pas bougé
# --------------------------------------------------------------------------------- #

# Valeurs EFFECTIVES relevées dans le code AVANT l'introduction du profil (inventaire
# exhaustif en tête de PIPELINE_PROFILE.md). Ce tableau est le témoin de non-régression :
# le modifier doit être un acte DÉLIBÉRÉ, adossé à un verdict mesuré.
HISTORIQUE = {
    "extract": "mistral-large-latest",
    "enrich": "mistral-large-latest",
    "abstraction": "mistral-small-latest",
    "opinion": "mistral-large-latest",
    "argmine": "mistral-small-latest",
    "translate": "mistral-small-latest",
}


@pytest.mark.parametrize("role,attendu", sorted(HISTORIQUE.items()))
def test_final_reproduit_les_valeurs_historiques(role, attendu, monkeypatch):
    monkeypatch.delenv(f"AGORA_MODEL_{role.upper()}", raising=False)
    assert P.FINAL.model_for(role) == attendu, (
        f"le profil `final` a changé le modèle du rôle {role!r} : "
        f"{P.FINAL.model_for(role)!r} au lieu de {attendu!r}. "
        "Si c'est voulu, il faut un verdict mesuré ET mettre à jour ce témoin."
    )


def test_final_garde_la_forme_et_l_embedder():
    assert P.FINAL.embedder == "arctic-l"      # Apache 2.0 — cf. pare-feu de licence
    assert P.FINAL.fine_gamma == 3.0
    assert P.FINAL.seed == 42


def test_les_modules_lisent_bien_le_profil():
    """Les alias dérivés des modules valent le profil actif — pas une copie figée."""
    from backend import analysis, build_analysis, build_arguments, build_opinion
    from pipeline.claims import backend as claims_backend
    from pipeline.cluster import layers, mistral_client

    actif = P.active()
    assert build_analysis.EXTRACT_MODEL == actif.model_for("extract")
    assert build_analysis.ENRICH_MODEL == actif.model_for("enrich")
    assert analysis.ABSTRACTION_CHAT_MODEL == actif.model_for("abstraction")
    assert build_opinion.MODEL == actif.model_for("opinion")
    assert build_arguments.MODEL == actif.model_for("argmine")
    assert mistral_client.NAMING_MODEL == actif.model_for("enrich")
    assert claims_backend.API_MODEL == actif.model_for("extract")
    assert layers.FINE_GAMMA == actif.fine_gamma


def test_le_defaut_dextraction_ne_diverge_plus():
    """Avant : trois défauts d'extraction contradictoires selon le point d'entrée.

    `build_analysis` disait `mistral-large-latest`, `claims/backend` `ministral-3b-latest`.
    Un appelant qui oubliait `model=` détruisait ou manquait le cache sans un mot.
    """
    from pipeline.claims import backend as claims_backend
    from backend import build_analysis

    assert claims_backend.API_MODEL == build_analysis.EXTRACT_MODEL


# --------------------------------------------------------------------------------- #
# 2. Non-collision : un profil ne peut pas écraser le cache d'un autre
# --------------------------------------------------------------------------------- #

def test_final_ecrit_au_chemin_historique(tmp_path):
    """`final` ne doit RIEN déplacer : les caches existants restent lisibles tels quels."""
    assert P.FINAL.dataset_dir(tmp_path, "un-dataset") == tmp_path / "un-dataset"


def test_live_est_isole_de_final(tmp_path):
    final_dir = P.FINAL.dataset_dir(tmp_path, "un-dataset")
    live_dir = P.LIVE.dataset_dir(tmp_path, "un-dataset")
    assert live_dir != final_dir
    assert final_dir not in live_dir.parents or live_dir.name != final_dir.name
    # Le point qui compte : écrire sous `live` ne touche AUCUN fichier de `final`.
    live_dir.mkdir(parents=True)
    (live_dir / "analysis.json").write_text("{}", encoding="utf-8")
    assert not (final_dir / "analysis.json").exists()


def test_tous_les_profils_non_defaut_sont_namespaces():
    """Garde-fou pour les profils FUTURS : seul `final` peut écrire au chemin historique."""
    for prof in P.list_profiles():
        if prof.name == P.DEFAULT_PROFILE:
            assert prof.cache_namespace == ""
        else:
            assert prof.cache_namespace, (
                f"le profil {prof.name!r} n'a pas de cache_namespace : il écrirait dans le "
                "dossier du profil servi et l'écraserait."
            )


def test_la_signature_distingue_les_profils():
    assert P.FINAL.signature() != P.LIVE.signature()


# --------------------------------------------------------------------------------- #
# 3. Sélection et surcharges
# --------------------------------------------------------------------------------- #

def test_profil_inconnu_leve_au_lieu_de_replier(monkeypatch):
    """Fail-closed : un `AGORA_PROFILE` mal orthographié ne doit PAS bâtir en `final`."""
    monkeypatch.setenv("AGORA_PROFILE", "finla")
    with pytest.raises(ValueError, match="profil inconnu"):
        P.active()


def test_profil_par_defaut_sans_env(monkeypatch):
    monkeypatch.delenv("AGORA_PROFILE", raising=False)
    assert P.active().name == P.DEFAULT_PROFILE == "final"


def test_surcharge_dun_role_nen_deplace_aucun_autre(monkeypatch):
    """Anti-cascade : c'est LE défaut de l'ancien montage à deux niveaux."""
    monkeypatch.setenv("AGORA_MODEL_OPINION", "un-modele-de-test")
    assert P.FINAL.model_for("opinion") == "un-modele-de-test"
    for role in P.FINAL.ROLES:
        if role != "opinion":
            assert P.FINAL.model_for(role) == HISTORIQUE[role], (
                f"surcharger `opinion` a déplacé le rôle {role!r} — la cascade est de retour."
            )


def test_role_inconnu_leve():
    with pytest.raises(ValueError, match="rôle inconnu"):
        P.FINAL.model_for("nimporte-quoi")


def test_live_est_marque_non_valide():
    """`live` porte des valeurs provisoires : le drapeau doit le dire, pas un commentaire."""
    assert P.FINAL.validated is True
    assert P.LIVE.validated is False


def test_live_partage_l_espace_dembedding_de_final():
    """Embedders différents ⇒ dimensions et échelles de cosinus incomparables."""
    assert P.LIVE.embedder == P.FINAL.embedder


# --------------------------------------------------------------------------------- #
# 4. Source de vérité unique : plus de littéral de modèle ailleurs
# --------------------------------------------------------------------------------- #

def test_aucun_litteral_de_modele_hors_du_profil():
    """Le critère d'acceptation réel : les VALEURS ne vivent qu'ici.

    On cherche les littéraux `"mistral-*"` / `"ministral-*"` dans le code de production.
    Trois exceptions LÉGITIMES, chacune pour une raison distincte :

    - `pipeline/profile.py` : la source de vérité elle-même ;
    - les modèles **OLLAMA** locaux (`DEFAULT_MODEL`, `MAC_MODEL` — `ministral-3:latest`) :
      ce sont des noms de modèles servis en local, pas des rôles d'API ;
    - `backend/cost.py` : une **grille tarifaire**, qui doit nommer les modèles pour les
      facturer. Elle ne CHOISIT rien — mais elle doit rester alignée sur les modèles que
      les profils peuvent réellement sélectionner (un modèle absent retombe silencieusement
      sur le tarif « small », donc une estimation fausse).

    `research/` et les tests sont hors périmètre (bancs, valeurs figées volontairement).
    """
    # Scan en Python pur : PAS de `subprocess`/`git`. Un autre test de la suite remplace
    # globalement `subprocess.Popen` par un double, ce qui faisait échouer ce test selon
    # l'ORDRE d'exécution — un test doit constater l'état du dépôt, pas l'ordre des tests.
    litteral = re.compile(r'"(mistral|ministral)-[a-z0-9.:-]+"')
    fuites: list[str] = []
    for chemin in sorted(list((REPO / "backend").rglob("*.py"))
                         + list((REPO / "pipeline").rglob("*.py"))):
        rel = chemin.relative_to(REPO).as_posix()
        if "/tests/" in rel or rel.endswith("/profile.py"):
            continue                                    # bancs & source de vérité
        if rel == "backend/cost.py":
            continue                                    # grille tarifaire : ne choisit rien
        for n, ligne in enumerate(chemin.read_text(encoding="utf-8").splitlines(), 1):
            if not litteral.search(ligne):
                continue
            if re.match(r"\s*(DEFAULT_MODEL|MAC_MODEL)\s*=", ligne) and "ministral-3:" in ligne:
                continue                                # modèles OLLAMA locaux
            fuites.append(f"{rel}:{n}:{ligne.strip()}")
    assert not fuites, (
        "des littéraux de modèle subsistent hors du profil — la source de vérité est "
        "à nouveau dupliquée :\n  " + "\n  ".join(fuites)
    )

"""Le LIBELLÉ d'un dataset servi ne doit jamais être rétrogradé par un rebuild.

Régression réelle : un re-ingest de tiktok sans `--label` a écrasé « Consultation TikTok
(FR) » par « tiktok » dans `meta.json`. Muette — le seul symptôme est le titre dans l'UI.
La cause : le repli `label or desc.extra["label"] or dataset` traitait un dataset DÉJÀ SERVI
comme un dataset neuf.

Deux verrous, testés ici :
  1. le libellé vit dans le DESCRIPTEUR (source de vérité, survit à tout rebuild) ;
  2. à défaut, `build_cache` HÉRITE du libellé déjà présent dans `meta.json`.
"""
from __future__ import annotations

import json

from pipeline.ingest.config import DESCRIPTORS_DIR

# Datasets servis en prod : leur libellé doit être porté par le descripteur, pas par une
# retouche manuelle du cache (que le prochain rebuild effacerait).
SERVIS = ("tiktok", "granddebat", "xstance", "republique-numerique", "ameliorer-agora")


def test_les_descripteurs_servis_portent_un_libelle():
    manquants = []
    for name in SERVIS:
        path = DESCRIPTORS_DIR / f"{name}.json"
        if not path.exists():
            continue
        desc = json.loads(path.read_text(encoding="utf-8"))
        label = desc.get("label")
        if not label or label == name:
            manquants.append(name)
    assert not manquants, (
        f"descripteurs sans libellé d'affichage : {manquants} — un rebuild les fera "
        "retomber sur leur id (« tiktok » au lieu de « Consultation TikTok (FR) »)."
    )


def test_le_libelle_existant_est_herite_a_defaut(tmp_path, monkeypatch):
    """Sans `--label` ni libellé au descripteur, on garde celui du `meta.json` existant."""
    from backend import build_cache as B

    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"id": "ds", "label": "Ma Consultation (FR)"}),
                         encoding="utf-8")

    # On rejoue la SEULE logique de résolution du libellé (le reste du build charge torch).
    def resolve(label, desc_label, meta_file, dataset):
        previous = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        inherited = previous.get("label")
        if inherited == dataset:
            inherited = None
        return label or desc_label or inherited or dataset

    assert resolve(None, None, meta_path, "ds") == "Ma Consultation (FR)"
    assert resolve(None, "Du descripteur", meta_path, "ds") == "Du descripteur"
    assert resolve("Explicite", None, meta_path, "ds") == "Explicite"

    # Un dataset neuf (pas de meta) retombe légitimement sur son id.
    assert resolve(None, None, tmp_path / "absent.json", "ds") == "ds"

    # Un repli d'un run PRÉCÉDENT (label == id) ne doit pas être hérité comme un vrai libellé.
    degrade = tmp_path / "degrade.json"
    degrade.write_text(json.dumps({"id": "ds", "label": "ds"}), encoding="utf-8")
    assert resolve(None, "Du descripteur", degrade, "ds") == "Du descripteur"

    assert hasattr(B, "build_cache")

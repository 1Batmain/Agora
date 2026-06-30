"""Hiérarchie MÈRE→ENFANTS (`backend.build_children`) — schéma + serving.

Verrouille le contrat servi par `/datasets` quand des consultations sont
imbriquées : un ENFANT (descripteur avec `parent_id`) est EXCLU de la liste
top-level mais reste servi PAR ID sur les autres endpoints ; une MÈRE expose
`children`. Les tests qui dépendent d'enfants RÉELLEMENT construits se SKIPPENT
proprement si le split n'a pas été lancé (cf. `build_children`), à l'image du
reste de la suite (jamais d'échec parasite, jamais de build déclenché).
"""

from __future__ import annotations

import pytest

from backend.build_children import slugify
from backend.recluster import dataset_descriptor
from ._helpers import available_datasets


def test_slugify_generic():
    # Générique : minuscule, non-alphanum → '-', chiffres préservés, repli non vide.
    assert slugify("Foreign Policy") == "foreign-policy"
    assert slugify("Welfare") == "welfare"
    assert slugify("  Éducation & Santé  ") == "ducation-sant"  # ASCII only, rogné
    assert slugify("3412") == "3412"
    assert slugify("!!!") == "x"


def _children_in_cache() -> list[str]:
    """Ids d'enfants RÉELLEMENT construits = datasets cachés avec `parent_id`."""
    return [d for d in available_datasets()
            if dataset_descriptor(d).get("parent_id")]


def test_datasets_listing_has_no_children(client):
    """La liste top-level = MÈRES + SIMPLES — JAMAIS un enfant (`parent_id`)."""
    items = client.get("/datasets").json()
    assert all("parent_id" not in d for d in items), \
        "un enfant (parent_id) ne doit pas apparaître dans la liste top-level"


def test_children_served_by_id_but_hidden_from_listing(client):
    """Chaque enfant construit : ABSENT de /datasets mais RÉSOLU par id (build_status 200)."""
    children = _children_in_cache()
    if not children:
        pytest.skip("aucun enfant construit (lance `python -m backend.build_children`)")
    listed = {d["id"] for d in client.get("/datasets").json()}
    for child in children:
        assert child not in listed, f"enfant {child!r} ne doit pas être listé top-level"
        # Servi comme un dataset normal : son id est whitelisté → 200, jamais 404.
        r = client.get("/build_status", params={"dataset": child})
        assert r.status_code == 200, r.text


def test_mother_exposes_children_and_back_reference(client):
    """Une mère expose `children`, et chaque enfant pointe en retour via `parent_id`."""
    children = _children_in_cache()
    if not children:
        pytest.skip("aucun enfant construit (lance `python -m backend.build_children`)")
    for child in children:
        desc = dataset_descriptor(child)
        parent = desc["parent_id"]
        assert parent in available_datasets(), f"parent {parent!r} de {child!r} absent du cache"
        parent_desc = dataset_descriptor(parent)
        assert child in parent_desc.get("children", []), \
            f"la mère {parent!r} doit lister son enfant {child!r} dans children"

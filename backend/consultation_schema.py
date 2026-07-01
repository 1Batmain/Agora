"""Schéma UNIQUE d'une consultation — source de vérité back↔front.

Une `Consultation` est l'objet servi tel quel par `GET /datasets` (un par dataset
clôturé ET un par consultation ouverte). Il est construit en UN SEUL endroit côté
backend — `recluster.dataset_descriptor` (clôturées, avec cache d'analyse) et
`recluster.open_consultation_descriptor` (ouvertes, sans cache) — qui retournent
tous deux CE type. Le front (`frontend/src/types.ts`) le mirroir à l'identique.

Aucune valeur en dur, aucun champ fantôme : tout est dérivé des données
(meta.json / ideas cachés / soumissions reçues).
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class Consultation(TypedDict):
    """Métadonnées d'une consultation, servies par `GET /datasets`.

    - `n_sample` : échantillon RÉELLEMENT analysé (= ancien `n_nodes`).
    - `n_contributions` : total RÉEL de contributions reçues (avant cap d'échantillonnage).
    - `n_nodes` : rétro-compat — alias historique de `n_sample` (lu par la suite
      pytest /datasets et par d'anciens consommateurs front). Toujours == `n_sample`.
    - `question`/`context` : présents pour les consultations OUVERTES (sujet affiché
      dans la vue Participer) ; optionnels pour les clôturées.
    - `parent_id`/`children` : hiérarchie MÈRE→ENFANTS (cf. `backend.build_children`).
      Une MÈRE porte `children=[ids d'enfants]` ; un ENFANT porte `parent_id=<mère>`.
      Une consultation SIMPLE (ni mère ni enfant) n'a ni l'un ni l'autre. Les enfants
      sont servis comme des datasets normaux (par id) mais EXCLUS de la liste top-level.
    """

    id: str
    label: str
    status: Literal["open", "closed"]
    n_sample: int
    n_contributions: int
    n_nodes: int  # rétro-compat : == n_sample
    languages: list[str]
    lang_counts: dict[str, int]
    source: str
    question: NotRequired[str]
    context: NotRequired[str]
    parent_id: NotRequired[str]
    children: NotRequired[list[str]]

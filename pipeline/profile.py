"""Profil de pipeline — SOURCE DE VÉRITÉ des modèles et de la forme du traitement.

Avant ce module, « quel modèle pour quoi » vivait dans **14 constantes réparties sur
7 fichiers**, avec trois défauts d'extraction contradictoires, un `TRANSLATE_MODEL`
dupliqué et des cascades d'env à deux niveaux dont les replis divergeaient. Ici, un
profil déclare **un modèle par RÔLE** plus les paramètres de forme ; les modules le
lisent au lieu de porter leur propre constante.

Même patron que `pipeline/embed/registry.py` (dataclass gelée + registre + résolveur),
étendu des embedders aux rôles LLM. Ajouter un profil = ajouter une entrée à `PROFILES`.

**UN CHAMP PAR RÔLE, jamais un champ `model` unique.** `abstraction` et `argmine` tournent
délibérément sur un modèle plus petit qu'`extract` — ce sont des choix mesurés (V-SELECT au
banc pour argmine), pas des oublis. Aplatir détruirait des verdicts.

Sélection : `AGORA_PROFILE` (défaut `final`). Surcharge fine par rôle via `AGORA_MODEL_<RÔLE>`
— UN seul niveau, pas de cascade (poser une variable ne doit jamais déplacer trois rôles à
l'insu de l'appelant).

Verdict et critères d'acceptation : `.agent/notes/PIPELINE_PROFILE.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineProfile:
    """Déclaration complète d'un pipeline : quels modèles, quelle forme, quel cache.

    `cache_namespace` — ISOLE les artefacts d'un profil. Vide pour `final`, qui écrit donc
    EXACTEMENT où le pipeline écrivait avant ce module (exigence d'iso-comportement : le
    refactor ne doit déplacer aucun octet). Tout autre profil écrit sous un sous-dossier
    dédié : deux profils qui partagent un dossier se corrompent mutuellement en silence —
    un profil « live » sur petit modèle écraserait les titres du profil de production.
    """

    name: str

    # --- rôles LLM (un champ par rôle) ---------------------------------------------- #
    extract: str          # extraction de claims verbatim (le plus coûteux, caché par avis)
    enrich: str           # titres, accroches, descriptions, insights
    abstraction: str      # profils de macro-thèmes (nommage/regroupement léger)
    opinion: str          # objet de clivage + stance des claims
    argmine: str          # sélection d'arguments (V-SELECT)
    translate: str        # traduction FR des avis/mots-clés non francophones

    # --- embedder (résolu par pipeline.embed.registry) ------------------------------- #
    embedder: str

    # --- forme du traitement --------------------------------------------------------- #
    fine_gamma: float     # résolution γ de la couche feuille servie
    resolution: float     # résolution Leiden par défaut (chaînes R&D)
    seed: int

    cache_namespace: str = ""
    validated: bool = False   # False ⇒ valeurs PROVISOIRES, non mesurées (cf. LIVE)
    note: str = ""

    # Rôles LLM, pour l'itération générique (surcharges d'env, diagnostics, tests).
    ROLES = ("extract", "enrich", "abstraction", "opinion", "argmine", "translate")

    def model_for(self, role: str) -> str:
        """Modèle d'un rôle, surcharge d'env `AGORA_MODEL_<RÔLE>` appliquée.

        Un SEUL niveau de repli (env → profil). Les anciennes cascades à deux niveaux
        (`AGORA_OPINION_MODEL` → `AGORA_ENRICH_MODEL` → littéral) couplaient des rôles
        sans le déclarer : poser une variable en déplaçait trois.
        """
        if role not in self.ROLES:
            raise ValueError(f"rôle inconnu : {role!r} (attendus : {', '.join(self.ROLES)})")
        return os.environ.get(f"AGORA_MODEL_{role.upper()}") or getattr(self, role)

    def dataset_dir(self, cache_dir: Path | str, dataset: str) -> Path:
        """Dossier de cache d'un dataset POUR CE PROFIL.

        `final` (namespace vide) → `<cache>/<dataset>` : chemin historique, inchangé.
        Sinon → `<cache>/<dataset>/_profiles/<namespace>`, isolé. C'est ce qui garantit
        qu'un build « live » ne peut pas écraser l'analyse servie.
        """
        base = Path(cache_dir) / dataset
        return base if not self.cache_namespace else base / "_profiles" / self.cache_namespace

    def signature(self) -> str:
        """Empreinte lisible du profil, à joindre aux clés de cache dérivées.

        Contient TOUT ce qui change une sortie (rôles + embedder + forme). Une signature
        incomplète est la panne la plus vicieuse du projet : elle sert un résultat périmé
        sans rien signaler.
        """
        parts = [self.name] + [f"{r}={self.model_for(r)}" for r in self.ROLES]
        parts += [f"embedder={self.embedder}", f"gamma={self.fine_gamma}",
                  f"resolution={self.resolution}", f"seed={self.seed}"]
        return "|".join(parts)


# --------------------------------------------------------------------------------- #
# Registre des profils
# --------------------------------------------------------------------------------- #

# PRODUCTION — reproduit EXACTEMENT les valeurs effectives d'avant ce module. Toute
# modification ici change la sortie servie : à faire par verdict mesuré, pas par confort.
FINAL = PipelineProfile(
    name="final",
    extract="mistral-large-latest",
    enrich="mistral-large-latest",
    abstraction="mistral-small-latest",   # nommage/regroupement léger (≠ extraction)
    opinion="mistral-large-latest",
    argmine="mistral-small-latest",       # V-SELECT : sélection, pas rédaction
    translate="mistral-small-latest",
    embedder="arctic-l",                  # Snowflake arctic-l v2.0, Apache 2.0
    fine_gamma=3.0,
    resolution=1.0,
    seed=42,
    cache_namespace="",                   # chemins historiques — NE PAS namespacer
    validated=True,
    note="profil servi ; valeurs figées par les bancs successifs.",
)

# LIVE — synthèse au fil de l'eau (débat rejoué). Couches lourdes retirées, modèles cheap.
#
# ⚠️ NON VALIDÉ (`validated=False`). Seul `extract` repose sur une mesure : ministral-3b ≈
# mistral-large sur 191 avis appariés (Δ +0,02) pour ~150× moins cher, cf.
# `research/llm_regression/`. Les autres rôles sont des choix PROVISOIRES par analogie de
# taille — aucun banc ne les couvre. Ne rien conclure d'une sortie « live » avant de les
# avoir mesurés ; le drapeau `validated` existe pour que ce soit lisible en un coup d'œil.
LIVE = PipelineProfile(
    name="live",
    extract="ministral-3b-latest",
    enrich="mistral-small-latest",
    abstraction="mistral-small-latest",
    opinion="mistral-small-latest",
    argmine="mistral-small-latest",
    translate="mistral-small-latest",
    embedder="arctic-l",                  # MÊME espace que final : sinon cosinus incomparables
    fine_gamma=3.0,
    resolution=1.0,
    seed=42,
    cache_namespace="live",               # isolé — ne peut pas écraser l'analyse servie
    validated=False,
    note="PROVISOIRE : seul `extract` est mesuré ; les autres rôles restent à bencher.",
)

PROFILES: dict[str, PipelineProfile] = {p.name: p for p in (FINAL, LIVE)}

DEFAULT_PROFILE = FINAL.name


def get_profile(name: str | None = None) -> PipelineProfile:
    """Profil par nom. `None` → `AGORA_PROFILE`, sinon le défaut (`final`).

    Fail-closed : un nom inconnu LÈVE plutôt que de retomber silencieusement sur `final` —
    un `AGORA_PROFILE=liv` mal orthographié qui bâtirait en production sans un mot serait
    exactement le genre de panne muette qu'on refuse.
    """
    if name is None:
        name = os.environ.get("AGORA_PROFILE", DEFAULT_PROFILE).strip() or DEFAULT_PROFILE
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(
            f"profil inconnu : {name!r} (connus : {', '.join(sorted(PROFILES))})"
        )
    return profile


def active() -> PipelineProfile:
    """Profil actif du processus (lit `AGORA_PROFILE` à CHAQUE appel).

    Volontairement NON mémoïsé : les tests et les CLI changent l'environnement en cours
    de vie ; un cache de module figerait le premier profil vu.
    """
    return get_profile(None)


def list_profiles() -> list[PipelineProfile]:
    return list(PROFILES.values())


def describe(profile: PipelineProfile | None = None) -> dict:
    """Vue sérialisable d'un profil (diagnostic `/health`, logs de build)."""
    p = profile or active()
    return {
        "name": p.name,
        "validated": p.validated,
        "models": {r: p.model_for(r) for r in p.ROLES},
        "embedder": p.embedder,
        "shape": {"fine_gamma": p.fine_gamma, "resolution": p.resolution, "seed": p.seed},
        "cache_namespace": p.cache_namespace,
    }


__all__ = [
    "PipelineProfile", "FINAL", "LIVE", "PROFILES", "DEFAULT_PROFILE",
    "get_profile", "active", "list_profiles", "describe",
]

"""Échantillon synthétique FR réaliste (thème : mobilité urbaine).

Sert à deux usages :
  1. Repli si une source réelle est indisponible (le pipeline aval tourne quand même).
  2. Génération du fixture committé `fixtures/ideas.sample.jsonl` (anonyme).

Le corpus contient volontairement des DUPLICATS exacts et des PARAPHRASES pour
exercer la déduplication (T-D3) et le clustering aval. 100 % synthétique : aucune
donnée personnelle réelle.
"""
from __future__ import annotations

import random

# Familles d'opinions (intention partagée, formulations variées = paraphrases).
_FAMILIES = [
    [
        "Il faut développer les pistes cyclables sécurisées dans toute la ville.",
        "On a besoin de plus de pistes cyclables protégées pour rouler sans danger.",
        "Davantage de voies vélo séparées de la route rendraient le vélo plus sûr.",
        "Sécurisons les déplacements à vélo avec un vrai réseau de pistes cyclables.",
    ],
    [
        "Les transports en commun devraient être gratuits pour les jeunes.",
        "La gratuité des bus et tramways pour les moins de 25 ans serait une bonne mesure.",
        "Rendre les transports publics gratuits pour les étudiants encouragerait leur usage.",
    ],
    [
        "Trop de voitures en centre-ville, il faut piétonniser les rues commerçantes.",
        "Le centre est saturé de voitures, piétonnisons davantage le coeur de ville.",
        "Réduire la place de la voiture au centre rendrait la ville plus agréable.",
        "Moins de circulation automobile au centre, plus d'espaces pour les piétons.",
    ],
    [
        "Il manque des bornes de recharge pour les véhicules électriques.",
        "On devrait installer beaucoup plus de bornes de recharge électrique.",
        "Le déploiement des bornes pour voitures électriques est trop lent.",
    ],
    [
        "Les bus ne passent pas assez souvent en banlieue.",
        "En périphérie, la fréquence des bus est largement insuffisante.",
        "Il faut renforcer les lignes de bus dans les quartiers excentrés.",
    ],
    [
        "Le covoiturage devrait être encouragé par des voies réservées.",
        "Des voies dédiées au covoiturage réduiraient les embouteillages.",
    ],
    [
        "Les trottinettes en libre-service encombrent les trottoirs.",
        "Il faut mieux réglementer le stationnement des trottinettes électriques.",
        "Les trottinettes laissées n'importe où gênent les piétons.",
    ],
    [
        "Un RER métropolitain relierait mieux les communes entre elles.",
        "Il faudrait un train régional cadencé pour connecter les villes voisines.",
    ],
    [
        "Le prix du carburant pénalise ceux qui n'ont pas d'alternative à la voiture.",
        "Sans transport en commun, beaucoup subissent la hausse du carburant.",
    ],
    [
        "Plus de parkings relais aux entrées de ville faciliteraient l'intermodalité.",
        "Des parkings relais connectés au tramway réduiraient l'usage de la voiture en centre.",
    ],
    [
        "Les horaires de tramway en soirée sont trop limités.",
        "Le tramway s'arrête trop tôt le soir, il faut prolonger le service.",
    ],
    [
        "La signalétique pour les vélos est insuffisante et dangereuse.",
        "Il manque un balisage clair des itinéraires cyclables.",
    ],
]

# Avis isolés (pas de paraphrase) pour la diversité.
_SINGLETONS = [
    "Je trouve que la ville devrait limiter la vitesse à 30 km/h partout.",
    "Les personnes à mobilité réduite n'ont pas assez d'accès aux quais.",
    "Pourquoi ne pas créer des navettes fluviales sur le fleuve ?",
    "Le stationnement payant est devenu beaucoup trop cher.",
    "Il faudrait végétaliser les abords des stations de transport.",
    "Les feux de circulation ne sont pas synchronisés, ça crée des bouchons.",
    "On devrait pouvoir emporter son vélo dans le métro plus facilement.",
    "Les pistes cyclables s'arrêtent souvent au milieu de nulle part.",
    "Un système de vélos en libre-service bien entretenu manque cruellement.",
    "Les bus polluent encore trop, passons à l'électrique ou l'hydrogène.",
    "Le télétravail devrait être encouragé pour réduire les trajets.",
    "Il faut des abris sécurisés pour garer les vélos près des gares.",
]

# Bruit : avis vides / quasi-vides pour vérifier le filtrage T-D2.
_NOISE = ["", "   ", "...", "rien", "ok", "??", "  -  "]


def generate(n: int = 300, seed: int = 42):
    """Retourne ~n enregistrements bruts {text, author, source, ts}.

    Déterministe (seed) -> régénération reproductible.
    """
    rng = random.Random(seed)
    base_ts = "2026-06-20T09:00:00"
    records = []

    def emit(text: str):
        idx = len(records)
        records.append(
            {
                "raw_id": f"syn-{idx:04d}",
                "text": text,
                # Auteurs partagés volontairement (poids social / dédup auteur).
                "author": f"citoyen-{rng.randint(1, max(2, n // 3))}",
                "source": "synthetic",
                "ts": base_ts,
            }
        )

    while len(records) < n:
        roll = rng.random()
        if roll < 0.55:  # paraphrase issue d'une famille
            fam = rng.choice(_FAMILIES)
            emit(rng.choice(fam))
        elif roll < 0.75:  # duplicat exact (1re formulation de la famille)
            fam = rng.choice(_FAMILIES)
            emit(fam[0])
        elif roll < 0.92:  # avis isolé
            emit(rng.choice(_SINGLETONS))
        else:  # bruit (sera filtré)
            emit(rng.choice(_NOISE))

    return records[:n]

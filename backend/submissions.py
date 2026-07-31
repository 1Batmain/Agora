"""Store + corrélation des contributions citoyennes (consultations OUVERTES).

PAS de LLM : chaque contribution est embeddée avec l'embedder LOCAL par défaut
(`embedder.DEFAULT_MODEL_ID`, aujourd'hui arctic-l) et corrélée au cosinus aux
contributions déjà reçues. Aucune clé, aucun appel réseau. Changer l'embedder par
défaut impose de REMESURER `SIMILARITY_THRESHOLD` (cf. ci-dessous).

Store append-only par consultation, deux fichiers sous `backend/cache/<id>/` :
  - `submissions.seed.jsonl` : seed de démo COMMITTÉ (corrélation non vide dès la
    1ʳᵉ vraie contribution) ; immuable.
  - `submissions.jsonl`       : contributions réelles, append-only, GITIGNORÉ.
Une ligne = `{text, vec, ts}` (`vec` = embedding L2-normalisé de l'embedder par défaut).

Le module est volontairement INDÉPENDANT de `recluster` (pas d'import croisé) :
il recalcule `CACHE_DIR` lui-même pour rester importable sans torch tant qu'on
n'embedde rien.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Même emplacement que les caches d'analyse (cf. recluster.CACHE_DIR), recalculé
# ici pour éviter un import croisé recluster ↔ submissions.
CACHE_DIR = Path(__file__).resolve().parent / "cache"

SEED_NAME = "submissions.seed.jsonl"
LIVE_NAME = "submissions.jsonl"

# Seuil de similarité cosinus pour compter une contribution « proche ».
# CALIBRÉ SUR L'EMBEDDER PAR DÉFAUT — donc à REMESURER si `DEFAULT_MODEL_ID` change :
# c'est une calibration d'embedder, pas un littéral de corpus. (Vécu : la bascule
# nomic-v2 → arctic-l a laissé le seuil nomic de 0.68 en place, ce qui rendait la
# corrélation quasi inopérante — les cosinus arctic tournent BEAUCOUP plus bas.)
#
# Mesure arctic-l (2026-07-31, même protocole que la calibration nomic d'origine) :
#   paires NON reliées (« plus d'arbres en ville » vs un retour produit) : 0.22–0.39
#   paraphrases reliées (perf / export / mobile / clarté)                : 0.41–0.79
#   → fenêtre de séparation ]0.385 ; 0.410[ ; retours réels : médiane 0.32, p95 0.49.
# La fenêtre est ÉTROITE (contre ~0.16 sous nomic) : la séparation est moins nette
# qu'avant. Si cette fonction devient importante, la dériver des données (comme
# `adaptive.derive_threshold`) plutôt que de la figer vaudrait mieux qu'un élargissement
# du jeu de sondes.
SIMILARITY_THRESHOLD = 0.40


def seed_path(consultation_id: str) -> Path:
    return CACHE_DIR / consultation_id / SEED_NAME


def live_path(consultation_id: str) -> Path:
    return CACHE_DIR / consultation_id / LIVE_NAME


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_submissions(consultation_id: str) -> list[dict]:
    """Toutes les contributions d'une consultation = seed COMMITTÉ + live append-only."""
    return _read_jsonl(seed_path(consultation_id)) + _read_jsonl(live_path(consultation_id))


def count_submissions(consultation_id: str) -> int:
    """Nombre de contributions reçues (seed + live), sans charger les vecteurs lourds."""
    return len(load_submissions(consultation_id))


def append_submission(consultation_id: str, text: str, vec, ts: str) -> None:
    """Ajoute une contribution au store LIVE (jamais au seed).

    `vec` peut être `None` : COLLECTE DIFFÉRÉE (prod publique serve-only, sans torch) — le
    texte est stocké seul, le vecteur sera calculé plus tard au build de l'analyse (en dev).
    `correlate` ignore déjà les lignes sans `vec`.
    """
    path = live_path(consultation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict = {"text": text, "ts": ts}
    if vec is not None:
        row["vec"] = [float(x) for x in np.asarray(vec).ravel()]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


_EMBEDDER = None


def embed_text(text: str) -> np.ndarray:
    """Embedde un texte avec l'embedder LOCAL par défaut (lazy-loadé, singleton).

    Le modèle torch n'est chargé qu'au 1ᵉʳ appel (1ʳᵉ contribution reçue) — le
    serveur démarre sans torch. Vecteur L2-normalisé (cosine = produit scalaire).
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from pipeline.embed.embedder import Embedder  # lazy : pas de torch au boot

        _EMBEDDER = Embedder()
    return _EMBEDDER.embed(text)


def correlate(
    vec: np.ndarray,
    existing: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """Corrèle un vecteur aux contributions existantes (cosinus).

    Vecteurs supposés L2-normalisés (cosine = produit scalaire). Renvoie un AGRÉGAT
    non-PII : le nombre de contributions au-dessus du seuil (`n_similar`) et la
    similarité la plus haute (`nearest_cos`). Vie privée (audit privacy #1) : NE
    renvoie JAMAIS le verbatim d'une autre contribution. Sans existant → zéro / None.
    """
    q = np.asarray(vec, dtype=np.float32).ravel()
    # On ne compare QUE des vecteurs du MÊME espace : un store peut contenir des vecteurs
    # d'un embedder précédent (dimension différente — nomic 768 vs arctic 1024). Les
    # ignorer est la seule sémantique correcte (un cosinus inter-espaces n'a aucun sens) et
    # évite un crash de forme sur `mat @ q`. Les lignes périmées redeviennent comparables
    # quand elles sont ré-embeddées au build.
    rows = [e["vec"] for e in existing if e.get("vec") and len(e["vec"]) == q.shape[0]]
    mat = np.array(rows, dtype=np.float32) if rows else np.empty((0, q.shape[0]), np.float32)
    if mat.size == 0:
        return {"n_similar": 0, "nearest_cos": None}
    sims = mat @ q
    n_similar = int((sims >= threshold).sum())
    return {
        "n_similar": n_similar,
        "nearest_cos": round(float(np.max(sims)), 4),
    }

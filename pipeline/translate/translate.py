"""Traduction d'avis citoyens vers le FRANÇAIS — cœur PUR, batché, langue-agnostique.

Brique réutilisable : on donne une liste de textes (dans n'importe quelle langue) et
on récupère leur traduction française, par **lots** (un seul appel LLM pour N textes →
coût/latence divisés). Aucune valeur de corpus en dur ; le modèle déduit la langue
source de chaque texte. Cheap par défaut (`mistral-small-latest`), surchargeable.

Robuste : repli GRACIEUX si pas de clé / erreur API / réponse mal formée → la fonction
renvoie `None` pour les textes non traduits (l'appelant garde l'original, ne cache rien
de faux). Ne lève jamais sur un échec réseau.

L'orchestration (détection de langue, cache disque idempotent aligné aux avis, phase de
build) vit côté `backend.translate` : ici, juste la traduction pure.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from pipeline.cluster import mistral_client

FR = "fr"
# Modèle CHEAP par défaut (traduction = tâche simple, batchée). Surchargeable par env.
DEFAULT_TRANSLATE_MODEL = "mistral-small-latest"
BATCH_SIZE = 20            # textes par appel LLM (compromis coût/latence/robustesse)
TRANSLATE_TEMPERATURE = 0.1  # fidèle, déterministe, pas créatif
# Bornage de génération : ~3× le texte source (marge pour langues verbeuses) + plancher.
TOKENS_PER_TEXT_FACTOR = 3
MIN_BATCH_TOKENS = 512
MAX_BATCH_TOKENS = 8000


def is_french(lang: str | None) -> bool:
    """Vrai si la langue est du français (pas de traduction nécessaire)."""
    return bool(lang) and str(lang).strip().lower().startswith("fr")


def _batch_messages(texts: list[str]) -> list[dict]:
    """Prompt JSON numéroté : traduire chaque texte en français, clés `"1".."N"`."""
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    system = (
        "Tu es un traducteur professionnel. On te donne des contributions citoyennes "
        "numérotées, écrites dans diverses langues. Traduis CHACUNE en français clair, "
        "fidèle et neutre — sans rien ajouter, retirer ni commenter, en préservant le "
        "sens et le registre. Ne traduis pas les textes déjà en français : recopie-les. "
        "Réponds UNIQUEMENT par un objet JSON dont les clés sont les numéros (\"1\", "
        "\"2\", …) et les valeurs les traductions françaises correspondantes."
    )
    user = (
        f"Traduis en français les {len(texts)} contributions suivantes. Renvoie un JSON "
        f'{{"1": "…", …, "{len(texts)}": "…"}} avec EXACTEMENT {len(texts)} entrées.\n\n'
        f"{numbered}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_batch(content: str, n: int) -> list[str] | None:
    """Parse la réponse JSON `{"1": …}` → liste alignée de longueur `n` (ou None)."""
    raw = (content or "").strip()
    if not raw:
        return None
    # Tolère un préfixe/suffixe hors-JSON : isole le premier objet `{…}`.
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: list[str] = []
    for i in range(1, n + 1):
        val = data.get(str(i))
        if not isinstance(val, str) or not val.strip():
            return None  # entrée manquante/vide → lot invalide (on ne cache rien de faux)
        out.append(val.strip())
    return out


def _max_tokens_for(texts: list[str]) -> int:
    chars = sum(len(t) for t in texts)
    est = (chars // 3) * TOKENS_PER_TEXT_FACTOR  # ~3 chars/token, marge ×3
    return max(MIN_BATCH_TOKENS, min(MAX_BATCH_TOKENS, est))


def translate_batch(
    texts: list[str],
    *,
    model: str = DEFAULT_TRANSLATE_MODEL,
    chat: Callable[..., str] | None = None,
) -> list[str | None]:
    """Traduit `texts` en français par lots → liste alignée (`None` pour les échecs).

    Un appel LLM par lot de `BATCH_SIZE`. Repli gracieux : sans clé Mistral ou sur
    erreur API / réponse mal formée, les éléments du lot concerné valent `None` (l'appelant
    garde l'original). `chat` permet d'injecter un client (tests) ; défaut = Mistral.
    """
    if not texts:
        return []
    call = chat or mistral_client.chat
    if chat is None and not mistral_client.available():
        return [None] * len(texts)  # pas de clé → rien traduit (repli original)

    out: list[str | None] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        try:
            content = call(
                _batch_messages(chunk), model=model,
                temperature=TRANSLATE_TEMPERATURE, max_tokens=_max_tokens_for(chunk),
                json_mode=True,
            )
            parsed = _parse_batch(content, len(chunk))
        except mistral_client.MistralError:
            parsed = None
        out.extend(parsed if parsed is not None else [None] * len(chunk))
    return out

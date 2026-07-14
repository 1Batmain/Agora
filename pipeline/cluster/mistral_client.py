"""Client API **Mistral** — partagé par le nommage `llm` et la synthèse.

Souverain EU (`api.mistral.ai`), langue-agnostique, zéro hardcoding de domaine.
La clé est lue depuis l'environnement (`MISTRAL_API_KEY`) ou un fichier secret
gitignoré ; elle n'est **jamais** codée en dur, **jamais** loggée, **jamais**
renvoyée dans une erreur.

Ordre de résolution de la clé (premier trouvé gagne) :
  1. variable d'env `MISTRAL_API_KEY` ;
  2. `backend/.env` (paire `MISTRAL_API_KEY=...`, fichier gitignoré) ;
  3. `var/mistral.key` à la racine du repo (fichier brut, gitignoré).

Tout est surchargeable par env (URL, modèles, timeout) — aucune valeur de corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

# Endpoint chat-completions Mistral (EU). Surcharge possible (proxy/tests).
API_URL = os.environ.get("AGORA_MISTRAL_URL", "https://api.mistral.ai/v1/chat/completions").rstrip("/")
# Modèle par défaut pour le nommage (titres courts batchés).
NAMING_MODEL = os.environ.get("AGORA_MISTRAL_MODEL", "mistral-large-latest")
# Modèle pour la synthèse (rapport) — par défaut le même, surchargeable (p.ex.
# `mistral-large-latest` pour un rapport plus fin).
SYNTHESIS_MODEL = os.environ.get("AGORA_MISTRAL_SYNTH_MODEL", NAMING_MODEL)
# Timeout réseau par appel (s). La synthèse peut être plus lente qu'un naming.
TIMEOUT = float(os.environ.get("AGORA_MISTRAL_TIMEOUT", "60"))

# ── Suivi d'usage (tokens) ──────────────────────────────────────────────────
# Accumulateur PROCESS-level : TOUT passe par `chat()` (extraction via ApiBackend,
# nommage, enrichissement, insights, opinion) → une seule instrumentation couvre le
# coût Mistral d'un build. `reset_usage()` au début d'un build, `get_usage()` à la fin.
import threading as _threading

_USAGE_LOCK = _threading.Lock()
_USAGE: dict = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}


# Appels perdus APRÈS épuisement des réessais : l'appelant va retomber sur son repli.
# Compté ici pour qu'un build puisse CONSTATER sa propre dégradation au lieu de la subir.
_EXHAUSTED: dict = {"count": 0, "by_status": {}}


def reset_usage() -> None:
    """Remet à zéro l'accumulateur de tokens (à appeler au début d'un build)."""
    with _USAGE_LOCK:
        _USAGE.update(calls=0, prompt_tokens=0, completion_tokens=0)
        _USAGE["by_model"] = {}
        _EXHAUSTED.update(count=0)
        _EXHAUSTED["by_status"] = {}


def get_exhausted() -> dict:
    """Appels définitivement perdus : `{count, by_status}` (429 = quota, 0 = réseau)."""
    import copy
    with _USAGE_LOCK:
        return copy.deepcopy(_EXHAUSTED)


def _note_exhausted(err: "MistralError") -> None:
    with _USAGE_LOCK:
        _EXHAUSTED["count"] += 1
        k = str(err.status)
        _EXHAUSTED["by_status"][k] = _EXHAUSTED["by_status"].get(k, 0) + 1


def get_usage() -> dict:
    """Instantané de l'usage accumulé : `{calls, prompt_tokens, completion_tokens, by_model}`."""
    import copy
    with _USAGE_LOCK:
        return copy.deepcopy(_USAGE)


def _record_usage(model: str, usage: dict) -> None:
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    with _USAGE_LOCK:
        _USAGE["calls"] += 1
        _USAGE["prompt_tokens"] += pt
        _USAGE["completion_tokens"] += ct
        m = _USAGE["by_model"].setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        m["calls"] += 1
        m["prompt_tokens"] += pt
        m["completion_tokens"] += ct

_KEY_ENV = "MISTRAL_API_KEY"
# Nom du fichier secret racine (cf. mémoire projet : `var/mistral.key`).
_KEY_FILE_REL = ("var", "mistral.key")
_DOTENV_REL = ("backend", ".env")


def _repo_root() -> Path:
    """Racine du repo (ce module vit dans `pipeline/cluster/`)."""
    return Path(__file__).resolve().parents[2]


def _read_dotenv_key(path: Path, name: str) -> str | None:
    """Lit `name=...` dans un .env minimal (sans dépendance python-dotenv)."""
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() == name:
                # retire export/quotes éventuels
                return val.strip().strip("\"'") or None
    except OSError:
        return None
    return None


def load_api_key() -> str | None:
    """Renvoie la clé Mistral (env > backend/.env > var/mistral.key) ou None.

    Ne lève jamais, ne logge jamais la valeur.
    """
    env = os.environ.get(_KEY_ENV)
    if env and env.strip():
        return env.strip()

    root = _repo_root()
    dotenv = root.joinpath(*_DOTENV_REL)
    if dotenv.exists():
        k = _read_dotenv_key(dotenv, _KEY_ENV)
        if k:
            return k

    key_file = root.joinpath(*_KEY_FILE_REL)
    if key_file.exists():
        try:
            content = key_file.read_text(encoding="utf-8").strip()
            if content:
                return content
        except OSError:
            pass
    return None


def available() -> bool:
    """Une clé est-elle disponible ? (ne valide PAS qu'elle soit correcte)."""
    return bool(load_api_key())


class MistralError(Exception):
    """Erreur d'appel Mistral — porte un `status` HTTP (0 = local/réseau).

    Le message est volontairement court et ne contient JAMAIS la clé.
    """

    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason
        super().__init__(f"mistral[{status}]: {reason}")


def _safe_reason(resp) -> str:
    """Extrait un motif d'erreur LISIBLE et SANS secret de la réponse Mistral."""
    try:
        data = resp.json()
        msg = data.get("message") or data.get("error") or data
        return str(msg)[:200]
    except Exception:
        return (resp.text or "")[:200] if hasattr(resp, "text") else "erreur inconnue"


# Un 429 (quota/rate-limit) ou un 5xx sont TRANSITOIRES : sans réessai, chaque appelant
# retombe sur son repli — titre en mots-clés, synthèse absente — et le build se déclare
# `ready` en ayant silencieusement perdu son enrichissement. C'est arrivé deux fois
# (« bug des 257 titres-labels », puis 27 % de titres + 37 % de synthèses perdus quand le
# quota Mistral a été atteint en cours de build). Le réessai vit ICI, une seule fois, pour
# tous les appelants — pas dans chaque module.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = int(os.environ.get("AGORA_MISTRAL_RETRIES", "4"))
BACKOFF_BASE = float(os.environ.get("AGORA_MISTRAL_BACKOFF", "2.0"))  # 2s, 4s, 8s, 16s
BACKOFF_CAP = 30.0


def _retry_after(resp) -> float | None:
    """Délai demandé par le serveur (`Retry-After`, en secondes), s'il est exploitable."""
    raw = resp.headers.get("Retry-After") if resp is not None else None
    try:
        return max(0.0, min(BACKOFF_CAP, float(raw))) if raw else None
    except (TypeError, ValueError):
        return None


def chat(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    json_mode: bool = False,
    timeout: float | None = None,
) -> str:
    """Un appel chat-completions Mistral. Renvoie le `content` du message assistant.

    RÉESSAIE les erreurs transitoires (429 quota/débit, 5xx, timeout, réseau) avec un
    backoff exponentiel plein-jitter, en respectant `Retry-After`. Ne réessaie JAMAIS
    ce qui ne peut pas guérir : absence de clé, 401/403 (clé invalide), 4xx de requête.

    Lève `MistralError` une fois les réessais épuisés (status = code HTTP, 0 =
    local/réseau). L'appelant décide du repli. La clé n'est jamais loggée.
    """
    key = load_api_key()
    if not key:
        raise MistralError(0, "no_api_key")

    import random
    import time

    import httpx

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    resp = None
    for attempt in range(MAX_RETRIES + 1):
        transient: MistralError | None = None
        try:
            resp = httpx.post(API_URL, json=payload, headers=headers,
                              timeout=timeout or TIMEOUT)
        except httpx.TimeoutException:
            transient = MistralError(0, "timeout")
        except httpx.HTTPError as exc:
            # Ne pas inclure d'éventuels headers/URL avec secret : type d'erreur seul.
            transient = MistralError(0, f"network_error:{type(exc).__name__}")
        else:
            if resp.status_code == 200:
                break
            err = MistralError(resp.status_code, _safe_reason(resp))
            if resp.status_code not in RETRY_STATUSES:
                raise err            # 401/403/400… : réessayer ne guérira rien
            transient = err

        if attempt == MAX_RETRIES:
            _note_exhausted(transient)
            raise transient
        delay = _retry_after(resp if transient.status in RETRY_STATUSES else None)
        if delay is None:            # plein-jitter : évite que N workers repartent ensemble
            delay = random.uniform(0.0, min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt)))
        time.sleep(delay)
        resp = None

    try:
        data = resp.json()
        _record_usage(model, data.get("usage") or {})
        return data["choices"][0]["message"]["content"] or ""
    except MistralError:
        raise
    except Exception:
        raise MistralError(0, "malformed_response")

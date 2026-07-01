"""Adaptateur MULTI-BACKEND pour l'extraction des claims (API par défaut, Mac opt-in).

Le Mac de Bob surchauffe → l'app ne dépend PAS de lui par défaut. L'extraction des
claims passe par un backend interchangeable, sélectionné par `AGORA_CLAIMS_BACKEND` :

  - ``api``  (DÉFAUT) — API **Mistral** ``ministral-3b-latest`` (souverain EU,
    via `pipeline.cluster.mistral_client` → clé `var/mistral.key`, JAMAIS loggée).
    La donnée citoyenne part chez Mistral (UE).
  - ``mac``  — Ollama **Mac** ``ministral-3`` (souverain LOCAL, opt-in) : la donnée
    ne sort pas du réseau privé (Tailscale, `AGORA_OLLAMA_URL`).
  - ``auto`` — sonde le Mac (timeout court) puis **repli API** s'il est injoignable.

Les DEUX backends partagent le même prompt et le même parsing (`extract.py`),
température 0, JSON mode → des claims au FORMAT identique quel que soit le chemin.
Les compteurs de coût/latence vont dans un `OllamaStats` partagé (réutilisé tel quel).
"""

from __future__ import annotations

import os
import time

from pipeline.claims.ollama import OllamaClient, OllamaStats, _redact
from pipeline.cluster import mistral_client

# Modèles par défaut, surchargeables par env (aucune valeur de corpus codée en dur).
API_MODEL = os.environ.get("AGORA_CLAIMS_API_MODEL", "ministral-3b-latest")
MAC_MODEL = os.environ.get("AGORA_CLAIMS_MAC_MODEL", "ministral-3:latest")

# Backend par défaut : API (le Mac est opt-in). Surchargeable par env.
DEFAULT_BACKEND = (os.environ.get("AGORA_CLAIMS_BACKEND") or "api").strip().lower()

# Sonde `auto` : timeout court (le Mac répond vite ou pas du tout).
MAC_PROBE_TIMEOUT = float(os.environ.get("AGORA_CLAIMS_MAC_TIMEOUT", "4"))

# Budget JSON par avis (les claims d'un avis tiennent largement dedans).
MAX_TOKENS = int(os.environ.get("AGORA_CLAIMS_MAX_TOKENS", "1024"))

# Retries sur erreurs TRANSITOIRES de l'API (429 rate-limit, 5xx, réseau). Sans cela,
# un 429 ferait tomber l'avis sur le repli "avis entier" → découpe DÉGRADÉE. Les gros
# modèles (mistral-large) ont des RPM bas : on retente avec backoff exponentiel borné.
API_MAX_RETRIES = int(os.environ.get("AGORA_CLAIMS_MAX_RETRIES", "6"))
API_BACKOFF_BASE = float(os.environ.get("AGORA_CLAIMS_BACKOFF_BASE", "2.0"))
API_BACKOFF_CAP = float(os.environ.get("AGORA_CLAIMS_BACKOFF_CAP", "30.0"))
_RETRIABLE_STATUS = frozenset({0, 408, 409, 429, 500, 502, 503, 504})


class BackendUnavailable(RuntimeError):
    """Le backend choisi est inutilisable (clé absente, Mac injoignable…).

    Message court et SANS secret — l'appelant le remonte tel quel (503 côté API).
    """


class ClaimBackend:
    """Interface commune : transforme des `messages` (prompt claims) en texte JSON.

    `name` est exposé dans `/claims` (transparence coût/données pour l'UI).
    `sovereign` = la donnée reste-t-elle dans le réseau privé (Mac) ? `note` est une
    phrase honnête sur où part la donnée.
    """

    name = "?"
    model = "?"
    sovereign = False
    note = ""

    def complete(self, messages: list[dict], *, stats: OllamaStats,
                 max_tokens: int | None = None) -> str | None:
        raise NotImplementedError

    def preflight(self) -> None:
        """Vérifie la disponibilité (clé/endpoint) SANS appel LLM, AVANT toute extraction.

        Lève `BackendUnavailable` si le backend est inutilisable → l'appelant échoue
        IMMÉDIATEMENT et proprement (pas de boucle sur des milliers de lots qui échouent).
        Défaut : no-op (backend supposé prêt)."""
        return None


class ApiBackend(ClaimBackend):
    """API Mistral (EU) via `mistral_client` — backend PRIMAIRE par défaut."""

    name = "api"
    sovereign = False
    note = "Données envoyées à l'API Mistral (UE) pour extraction."

    def __init__(self, model: str | None = None) -> None:
        # Pas d'I/O ni de validation ici : construire un backend ne doit pas échouer
        # quand les claims sont déjà en cache (la clé n'est requise qu'à l'extraction).
        self.model = model or API_MODEL

    def preflight(self) -> None:
        if not mistral_client.available():
            raise BackendUnavailable(
                "clé API Mistral absente — fournir MISTRAL_API_KEY ou var/mistral.key "
                "(ou passer AGORA_CLAIMS_BACKEND=mac pour l'extraction locale)."
            )

    def complete(self, messages: list[dict], *, stats: OllamaStats,
                 max_tokens: int | None = None) -> str | None:
        if not mistral_client.available():
            raise BackendUnavailable(
                "clé API Mistral absente — fournir MISTRAL_API_KEY ou var/mistral.key."
            )
        t0 = time.monotonic()
        for attempt in range(API_MAX_RETRIES + 1):
            try:
                content = mistral_client.chat(
                    messages, model=self.model, temperature=0.0,
                    max_tokens=max_tokens or MAX_TOKENS, json_mode=True,
                )
            except mistral_client.MistralError as exc:
                # `exc.status` seul : ne JAMAIS logger la clé (ni le message de l'API).
                if exc.status in _RETRIABLE_STATUS and attempt < API_MAX_RETRIES:
                    delay = min(API_BACKOFF_CAP, API_BACKOFF_BASE * (2 ** attempt))
                    print(f"  ⏳ mistral[{self.model}]: HTTP {exc.status} — retry "
                          f"{attempt + 1}/{API_MAX_RETRIES} dans {delay:.0f}s")
                    time.sleep(delay)
                    continue
                stats.errors += 1
                print(f"  ⚠️ mistral[{self.model}]: HTTP {exc.status} (abandon)")
                return None
            stats.calls += 1
            stats.cold_seconds += time.monotonic() - t0
            return content
        return None  # boucle épuisée sans succès (toutes les tentatives 429/5xx)


class MacBackend(ClaimBackend):
    """Ollama Mac (`ministral-3`) via `OllamaClient` — souverain local, opt-in."""

    name = "mac"
    sovereign = True
    note = "Extraction 100% locale (Mac) — les données ne sortent pas du réseau privé."

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.client = OllamaClient(base_url)
        self.model = model or MAC_MODEL
        self._think: bool | None = None
        self._warm = False

    def _ensure_warm(self, *, timeout: float = 600.0) -> bool:
        """Charge le modèle une fois (sort du cold-start). Renvoie le succès."""
        if self._warm:
            return True
        ok, think = self.client.warmup(self.model, timeout=timeout)
        if ok:
            self._think = think
            self._warm = True
        return ok

    def probe(self, *, timeout: float = MAC_PROBE_TIMEOUT) -> bool:
        """Le Mac est-il joignable ET le modèle chargeable, vite ? (pour `auto`)."""
        if not self._tags_reachable(timeout):
            return False
        return self._ensure_warm(timeout=timeout)

    def _tags_reachable(self, timeout: float) -> bool:
        """GET /api/tags — test de connectivité rapide (connexion refusée → vite faux)."""
        import httpx

        try:
            r = httpx.get(self.client.base_url + "/api/tags", timeout=timeout)
            r.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001 — on rapporte, on ne masque pas
            print(f"  ℹ️ Mac @ {_redact(self.client.base_url)} injoignable: {type(exc).__name__}")
            return False

    def complete(self, messages: list[dict], *, stats: OllamaStats,
                 max_tokens: int | None = None) -> str | None:
        if not self._ensure_warm():
            raise BackendUnavailable(
                f"LLM local {self.model!r} injoignable — exporter AGORA_OLLAMA_URL "
                "depuis var/MAC_LOCAL_OLLAMA et vérifier que le Mac est allumé."
            )
        return self.client.chat(messages, model=self.model, think=self._think,
                                stats=stats, max_tokens=max_tokens)


def resolve_backend(
    name: str | None = None,
    *,
    ollama_url: str | None = None,
    model: str | None = None,
) -> ClaimBackend:
    """Construit le backend demandé (`name` ou `AGORA_CLAIMS_BACKEND`, défaut ``api``).

    ``auto`` sonde le Mac (timeout court) et **bascule sur l'API** s'il est injoignable.
    Lève `BackendUnavailable` si le backend choisi est inutilisable, `ValueError` si
    le nom est inconnu. `model` surcharge le modèle par défaut du backend.
    """
    name = (name or DEFAULT_BACKEND).strip().lower()

    if name == "api":
        return ApiBackend(model=model)
    if name == "mac":
        return MacBackend(ollama_url, model=model)
    if name == "auto":
        mac = MacBackend(ollama_url, model=model)
        if mac.probe():
            return mac
        print("  ↪️ Mac indisponible → repli sur l'API Mistral.")
        return ApiBackend()  # repli : modèle API par défaut
    raise ValueError(
        f"AGORA_CLAIMS_BACKEND inconnu: {name!r} (attendu: api | mac | auto)."
    )

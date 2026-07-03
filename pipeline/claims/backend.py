"""Adaptateur MULTI-BACKEND pour l'extraction des claims (API par défaut, Ollama opt-in).

Le Ollama de Bob surchauffe → l'app ne dépend PAS de lui par défaut. L'extraction des
claims passe par un backend interchangeable, sélectionné par `AGORA_CLAIMS_BACKEND` :

  - ``api``  (DÉFAUT) — API **Mistral** ``ministral-3b-latest`` (souverain EU,
    via `pipeline.cluster.mistral_client` → clé `var/mistral.key`, JAMAIS loggée).
    La donnée citoyenne part chez Mistral (UE).
  - ``ollama``  — Ollama **Ollama** ``ministral-3`` (souverain LOCAL, opt-in) : la donnée
    ne sort pas du réseau privé (Tailscale, `AGORA_OLLAMA_URL`).
  - ``auto`` — sonde le Ollama (timeout court) puis **repli API** s'il est injoignable.

Les DEUX backends partagent le même prompt et le même parsing (`extract.py`),
température 0, JSON mode → des claims au FORMAT identique quel que soit le chemin.
Les compteurs de coût/latence vont dans un `OllamaStats` partagé (réutilisé tel quel).
"""

from __future__ import annotations

import os
from pipeline.claims.ollama import OllamaClient, OllamaStats, _redact
from pipeline.cluster import mistral_client
import time

# Modèles par défaut, surchargeables par env (aucune valeur de corpus codée en dur).
API_MODEL = os.environ.get("AGORA_CLAIMS_API_MODEL", "ministral-3b-latest")
OLLAMA_MODEL = os.environ.get("AGORA_CLAIMS_OLLAMA_MODEL", "ministral-3:latest")

# Backend par défaut : API (le Ollama est opt-in). Surchargeable par env.
DEFAULT_BACKEND = (os.environ.get("AGORA_CLAIMS_BACKEND") or "api").strip().lower()

# Sonde `auto` : timeout court (le Ollama répond vite ou pas du tout).
OLLAMA_PROBE_TIMEOUT = float(os.environ.get("AGORA_CLAIMS_OLLAMA_TIMEOUT", "4"))

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
    """Le backend choisi est inutilisable (clé absente, Ollama injoignable…).

    Message court et SANS secret — l'appelant le remonte tel quel (503 côté API).
    """


class ClaimBackend:
    """Interface commune : transforme des `messages` (prompt claims) en texte JSON.

    `name` est exposé dans `/claims` (transparence coût/données pour l'UI).
    `sovereign` = la donnée reste-t-elle dans le réseau privé (Ollama) ? `note` est une
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
                "(ou passer AGORA_CLAIMS_BACKEND=ollama pour l'extraction locale)."
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


class OllamaBackend(ClaimBackend):
    """Ollama Ollama (`ministral-3`) via `OllamaClient` — souverain local, opt-in."""

    name = "ollama"
    sovereign = True
    note = "Extraction 100% locale (Ollama) — les données ne sortent pas du réseau privé."

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.client = OllamaClient(base_url)
        self.model = model or OLLAMA_MODEL
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

    def probe(self, *, timeout: float = OLLAMA_PROBE_TIMEOUT) -> bool:
        """Le Ollama est-il joignable ET le modèle chargeable, vite ? (pour `auto`)."""
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
            print(f"  ℹ️ Ollama @ {_redact(self.client.base_url)} injoignable: {type(exc).__name__}")
            return False

    def complete(self, messages: list[dict], *, stats: OllamaStats,
                 max_tokens: int | None = None) -> str | None:
        if not self._ensure_warm():
            raise BackendUnavailable(
                f"LLM local {self.model!r} injoignable — exporter AGORA_OLLAMA_URL "
                "depuis var/OLLAMA_LOCAL_URL et vérifier que le Ollama est allumé."
            )
        return self.client.chat(messages, model=self.model, think=self._think,
                                stats=stats, max_tokens=max_tokens)


class LangchainBackend(ClaimBackend):
    """Backend générique Langchain pour les API compatibles OpenAI (LM Studio, NIM)."""

    def __init__(self, llm, name: str, sovereign: bool, note: str, supports_system_role: bool = True):
        self.llm = llm
        self.model = llm.model_name
        self.name = name
        self.sovereign = sovereign
        self.note = note
        self.supports_system_role = supports_system_role

    def complete(self, messages: list[dict], *, stats: OllamaStats,
                 max_tokens: int | None = None) -> str | None:
        from langchain_core.messages import SystemMessage, HumanMessage
        t0 = time.monotonic()
        
        system_prompt = ""
        if not self.supports_system_role:
            for m in messages:
                if m["role"] == "system":
                    system_prompt += m["content"] + "\n\n"
                    
        lc_msgs = []
        for m in messages:
            if m["role"] == "system":
                if self.supports_system_role:
                    lc_msgs.append(SystemMessage(content=m["content"]))
            elif m["role"] == "assistant":
                # Fallback for assistant role if needed
                from langchain_core.messages import AIMessage
                lc_msgs.append(AIMessage(content=m["content"]))
            else:
                content = m["content"]
                if not self.supports_system_role and system_prompt and not any(isinstance(x, HumanMessage) for x in lc_msgs):
                    content = system_prompt + content
                lc_msgs.append(HumanMessage(content=content))

        try:
            llm_to_use = self.llm
            if max_tokens:
                llm_to_use = self.llm.bind(max_tokens=max_tokens)
            
            response = llm_to_use.invoke(lc_msgs)
            
            usage_metadata = response.response_metadata.get("token_usage", {})
            stats.calls += 1
            stats.cold_seconds += time.monotonic() - t0
            stats.eval_tokens += usage_metadata.get("completion_tokens", 0)
            
            return response.content
            
        except Exception as exc:
            stats.errors += 1
            print(f"  [!] langchain[{self.model}]: {type(exc).__name__} - {exc}")
            return None


class LMStudioBackend(LangchainBackend):
    """LM Studio local (compatible OpenAI) via Langchain."""

    def __init__(self, model: str | None = None) -> None:
        from langchain_openai import ChatOpenAI
        model_name = model or os.environ.get("AGORA_LMSTUDIO_MODEL", "mistral-7b-instruct-v0.3")
        base_url = os.environ.get("AGORA_LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
        llm = ChatOpenAI(
            model=model_name,
            api_key="lm-studio",
            base_url=base_url,
            temperature=0.0
        )
        super().__init__(
            llm=llm,
            name="lmstudio",
            sovereign=True,
            note="Extraction 100% locale (LM Studio) — les données ne sortent pas du réseau.",
            supports_system_role=False
        )


class NimBackend(LangchainBackend):
    """Nvidia NIM (compatible OpenAI) via Langchain."""

    def __init__(self, model: str | None = None) -> None:
        from langchain_openai import ChatOpenAI
        model_name = model or os.environ.get("AGORA_NIM_MODEL", "mistralai/mistral-small-4-119b-2603")
        base_url = os.environ.get("AGORA_NIM_URL", "https://integrate.api.nvidia.com/v1")
        api_key = os.environ.get("NIM_API_KEY", "")
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key or "dummy",
            base_url=base_url,
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        super().__init__(
            llm=llm,
            name="nim",
            sovereign=False,
            note="Données envoyées à Nvidia NIM pour extraction."
        )

    def preflight(self) -> None:
        if not os.environ.get("NIM_API_KEY"):
            raise BackendUnavailable("clé API NIM absente — fournir NIM_API_KEY.")


def resolve_backend(
    name: str | None = None,
    *,
    ollama_url: str | None = None,
    model: str | None = None,
) -> ClaimBackend:
    """Construit le backend demandé (`name` ou `AGORA_CLAIMS_BACKEND`, défaut ``api``).

    ``auto`` sonde le Ollama (timeout court) et **bascule sur l'API** s'il est injoignable.
    Lève `BackendUnavailable` si le backend choisi est inutilisable, `ValueError` si
    le nom est inconnu. `model` surcharge le modèle par défaut du backend.
    """
    name = (name or DEFAULT_BACKEND).strip().lower()

    if name == "api":
        return ApiBackend(model=model)
    if name == "ollama":
        return OllamaBackend(ollama_url, model=model)
    if name == "auto":
        ollama = OllamaBackend(ollama_url, model=model)
        if ollama.probe():
            return ollama
        print("  ↪️ Ollama indisponible → repli sur l'API Mistral.")
        return ApiBackend()  # repli : modèle API par défaut
    if name == "lmstudio":
        return LMStudioBackend(model=model)
    if name == "nim":
        return NimBackend(model=model)
    raise ValueError(
        f"AGORA_CLAIMS_BACKEND inconnu: {name!r} (attendu: api | ollama | auto | lmstudio | nim)."
    )

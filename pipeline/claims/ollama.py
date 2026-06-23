"""Client Ollama souverain (LLM LOCAL via `AGORA_OLLAMA_URL`).

Promu depuis `research/segmentation/small_models.py` pour devenir un composant
DURABLE de `pipeline.claims` — utilisable par le backend sans dépendre du code
d'évaluation. Cible : le Mac de Bob (Apple Silicon via Tailscale), où tourne
`ministral-3` ; la donnée citoyenne ne sort jamais du réseau privé.

Caractéristiques :
  - endpoint configurable (`base_url` ou env `AGORA_OLLAMA_URL`, défaut localhost) ;
  - JSON mode + température 0 (extraction déterministe) ;
  - pensée des raisonneurs coupée (`think:false`) sinon champ omis (non-raisonneur) ;
  - cache disque clé par (endpoint, modèle, messages) → relances gratuites ;
  - erreurs RAPPORTÉES (jamais masquées), aucun secret loggé.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "ollama"

# Endpoint par défaut : Mac via Tailscale si AGORA_OLLAMA_URL est exporté, sinon
# Ollama local. On NE logge jamais l'URL complète (réseau privé).
DEFAULT_BASE = os.environ.get("AGORA_OLLAMA_URL", "http://localhost:11434").rstrip("/")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class OllamaStats:
    """Compteurs d'un run d'extraction (coût / latence / cache)."""
    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    cold_seconds: float = 0.0       # latence cumulée (miss + 1er coût mémorisé)
    eval_tokens: int = 0


def _redact(url: str) -> str:
    """Masque le host (réseau privé) dans un message d'erreur."""
    m = re.match(r"^(https?://)([^/:]+)", url or "")
    return f"{m.group(1)}<host>" if m else "<endpoint>"


class OllamaClient:
    """Client chat Ollama (JSON mode, cache disque). Un client = un endpoint."""

    def __init__(self, base_url: str | None = None, *, use_cache: bool = True,
                 cache_dir: Path | None = None) -> None:
        self.base_url = (base_url or DEFAULT_BASE).rstrip("/")
        self.url = self.base_url + "/api/chat"
        self.use_cache = use_cache
        self.cache_dir = cache_dir or CACHE_DIR

    # -- bas niveau --------------------------------------------------------- #
    def _post(self, messages: list[dict], *, model: str, think: bool | None,
              timeout: float) -> dict:
        """POST /api/chat. `think=None` → champ omis (modèle non-raisonneur)."""
        import httpx

        payload = {
            "model": model, "messages": messages, "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_ctx": 4096},
        }
        if think is not None:
            payload["think"] = think
        r = httpx.post(self.url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _key(self, model: str, messages: list[dict]) -> Path:
        blob = json.dumps([self.base_url, model, messages],
                          ensure_ascii=False, sort_keys=True)
        h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    # -- haut niveau -------------------------------------------------------- #
    def warmup(self, model: str, *, timeout: float = 600.0) -> tuple[bool, bool | None]:
        """Charge le modèle (sort le cold-start). Renvoie (ok, think).

        `think` = réglage de pensée retenu : `False` si le modèle accepte
        `think:false` (raisonneur), `None` s'il ne le supporte pas. Lève une
        exception silencieuse → (False, None) si l'endpoint est injoignable.
        """
        msg = [{"role": "user", "content": 'Réponds en JSON: {"ok": true}'}]
        for think in (False, None):
            try:
                self._post(msg, model=model, think=think, timeout=timeout)
                return True, think
            except Exception as exc:  # noqa: BLE001
                if think is False:
                    continue  # 400 « does not support thinking » → retombe sur None
                print(f"  ⚠️ warmup {model} @ {_redact(self.base_url)}: {type(exc).__name__}")
                return False, None
        return False, None

    def chat(self, messages: list[dict], *, model: str, think: bool | None,
             stats: OllamaStats, timeout: float = 600.0) -> str | None:
        """Chat (JSON mode, temp 0) avec cache disque. Renvoie le contenu ou None."""
        cpath = self._key(model, messages) if self.use_cache else None
        if cpath is not None and cpath.exists():
            rec = json.loads(cpath.read_text(encoding="utf-8"))
            stats.cache_hits += 1
            stats.cold_seconds += float(rec.get("seconds", 0.0))
            stats.eval_tokens += int(rec.get("eval_count", 0))
            return rec["content"]

        t0 = time.monotonic()
        try:
            data = self._post(messages, model=model, think=think, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — on rapporte, on ne masque pas
            stats.errors += 1
            print(f"  ⚠️ ollama[{model}] @ {_redact(self.base_url)}: {type(exc).__name__}")
            return None
        elapsed = time.monotonic() - t0
        content = (data.get("message") or {}).get("content") or ""
        content = _THINK_RE.sub("", content).strip()
        eval_count = int(data.get("eval_count", 0))
        stats.calls += 1
        stats.cold_seconds += elapsed
        stats.eval_tokens += eval_count
        if cpath is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(
                {"content": content, "seconds": round(elapsed, 3),
                 "eval_count": eval_count}, ensure_ascii=False), encoding="utf-8")
        return content


def parse_json_object(raw: str) -> dict | None:
    """Parse tolérant : extrait le 1er objet JSON d'une réponse LLM (bruit toléré).

    Les petits modèles ajoutent parfois du texte autour du JSON. On tente le parse
    direct, puis on isole la 1re accolade équilibrée. Renvoie None si rien d'exploitable.
    """
    if not raw:
        return None
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # Repli : 1re sous-chaîne {...} équilibrée.
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(raw)):
        c = raw[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(raw[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except (json.JSONDecodeError, ValueError):
                    return None
    return None

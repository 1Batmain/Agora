"""Client LLM **local** (vLLM / endpoint OpenAI-compatible) — redéclaration de
`mistral_client.chat` pour faire tourner les étapes LLM du build sur une box GPU
(H100) SANS qu'aucune donnée citoyenne ne sorte de la machine.

Même contrat que `mistral_client.chat` (mêmes exceptions `MistralError`, même
accumulateur de tokens), mais :
  * endpoint local (`AGORA_LOCAL_LLM_URL`, défaut vLLM :8000), AUCUNE clé requise ;
  * modèle par défaut local (`AGORA_LOCAL_LLM_MODEL`, ex. Gemma 12B) ;
  * repli « single-turn » optionnel (`AGORA_LOCAL_LLM_FOLD_SYSTEM=1`) pour les
    chat-templates qui refusent le rôle `system` (certains variants Gemma).

En pratique, les builds existants n'importent PAS ce module : ils passent par
`mistral_client`, dont l'URL et les modèles sont déjà surchargeables par env.
Ce script sert donc surtout de : (1) selftest de l'endpoint AVANT de lancer un
build ; (2) générateur des exports d'env qui routent TOUTES les étapes
(claims, nommage, enrichissement, insights, opinion) vers le LLM local.

Sur la box GPU :
    # 1. servir le modèle (venv séparé du repo — vLLM a ses propres deps lourdes)
    uvx vllm serve google/gemma-4-12B-it --dtype bfloat16 \
        --max-model-len 8192 --gpu-memory-utilization 0.85 --port 8000
    # 2. vérifier l'endpoint (chat simple + mode JSON)
    uv run python -m pipeline.cluster.local_llm_client --selftest
    # 3. exporter le routage puis lancer les builds standard, inchangés
    eval "$(uv run python -m pipeline.cluster.local_llm_client --print-env)"
    uv run python -m backend.build_analysis --dataset <id>
    uv run python -m backend.build_opinion  --dataset <id>
"""

from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

from . import mistral_client
from .mistral_client import MistralError, _record_usage, _safe_reason

# Endpoint OpenAI-compatible local (vLLM `vllm serve` expose /v1/chat/completions).
API_URL = os.environ.get(
    "AGORA_LOCAL_LLM_URL", "http://localhost:8000/v1/chat/completions").rstrip("/")
# Modèle servi localement — aucune valeur de corpus, surchargeable par env.
MODEL = os.environ.get("AGORA_LOCAL_LLM_MODEL", "google/gemma-4-12B-it")
# Certains chat-templates (variants Gemma) refusent le rôle `system` : repli
# single-turn qui fond le system dans le premier tour user.
FOLD_SYSTEM = os.environ.get("AGORA_LOCAL_LLM_FOLD_SYSTEM", "0") == "1"
TIMEOUT = float(os.environ.get("AGORA_LOCAL_LLM_TIMEOUT", "120"))


def _fold_system(messages: list[dict]) -> list[dict]:
    """[{system}, {user}, …] → [{user: system + user}, …] (template sans system)."""
    if not messages or messages[0].get("role") != "system":
        return messages
    system, first_user, rest = messages[0], None, []
    for m in messages[1:]:
        if first_user is None and m.get("role") == "user":
            first_user = {"role": "user",
                          "content": f"{system['content']}\n\n{m['content']}"}
        else:
            rest.append(m)
    return ([first_user] if first_user else
            [{"role": "user", "content": system["content"]}]) + rest


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    json_mode: bool = False,
    timeout: float | None = None,
) -> str:
    """Un appel chat-completions vers l'endpoint LOCAL. Contrat = `mistral_client.chat`.

    Pas de clé requise ; les tokens sont comptés dans le MÊME accumulateur que
    l'API (le coût affiché d'un build reste honnête, à 0 € près).
    """
    import httpx

    if FOLD_SYSTEM:
        messages = _fold_system(messages)
    payload: dict = {
        "model": model or MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = httpx.post(API_URL, json=payload,
                          headers={"Content-Type": "application/json"},
                          timeout=timeout or TIMEOUT)
    except httpx.TimeoutException:
        raise MistralError(0, "timeout")
    except httpx.HTTPError as exc:
        raise MistralError(0, f"network_error:{type(exc).__name__}")

    if resp.status_code != 200:
        raise MistralError(resp.status_code, _safe_reason(resp))
    try:
        data = resp.json()
        _record_usage(payload["model"], data.get("usage") or {})
        return data["choices"][0]["message"]["content"] or ""
    except MistralError:
        raise
    except Exception:
        raise MistralError(0, "malformed_response")


def env_exports(model: str | None = None, url: str | None = None) -> list[str]:
    """Exports shell qui routent TOUTES les étapes LLM du build vers l'endpoint local.

    Le client `mistral_client` existant est déjà entièrement pilotable par env :
    on pointe son URL vers vLLM (clé factice, jamais envoyée nulle part d'autre)
    et on aligne chaque knob de modèle (claims / nommage / synthèse / opinion /
    enrichissement) sur le modèle local.
    """
    m = model or MODEL
    u = url or API_URL
    return [
        f"export AGORA_MISTRAL_URL={u}",
        "export MISTRAL_API_KEY=local-vllm",  # factice : vLLM n'exige pas de clé
        f"export AGORA_MISTRAL_MODEL={m}",
        f"export AGORA_MISTRAL_SYNTH_MODEL={m}",
        f"export AGORA_CLAIMS_API_MODEL={m}",
        f"export AGORA_OPINION_MODEL={m}",
        f"export AGORA_ENRICH_MODEL={m}",
    ]


def selftest() -> int:
    """Deux appels de fumée (chat simple + mode JSON). 0 si OK, 1 sinon."""
    checks = [
        ("chat simple", dict(json_mode=False),
         [{"role": "system", "content": "Réponds en un mot."},
          {"role": "user", "content": "Quelle est la capitale de la France ?"}]),
        ("mode JSON", dict(json_mode=True),
         [{"role": "user", "content": 'Réponds en JSON : {"ok": true}'}]),
    ]
    failures = 0
    print(f"endpoint : {API_URL}\nmodèle   : {MODEL}\nfold_system : {FOLD_SYSTEM}\n")
    for name, kw, messages in checks:
        t0 = perf_counter()
        try:
            out = chat(messages, max_tokens=64, temperature=0.0, **kw)
            print(f"  [ok  ] {name} ({perf_counter() - t0:.1f}s) : {out.strip()[:80]!r}")
        except MistralError as e:
            failures += 1
            print(f"  [FAIL] {name} : {e}", file=sys.stderr)
    if failures and not FOLD_SYSTEM:
        print("\nAstuce : si l'erreur mentionne le rôle 'system', réessayer avec "
              "AGORA_LOCAL_LLM_FOLD_SYSTEM=1.", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Client LLM local (vLLM) : selftest + exports d'env de routage.")
    ap.add_argument("--selftest", action="store_true",
                    help="teste l'endpoint (chat + JSON) et sort 0/1")
    ap.add_argument("--print-env", action="store_true",
                    help="imprime les exports qui routent les builds vers le LLM local")
    ap.add_argument("--model", default=None, help="surcharge du modèle local")
    ap.add_argument("--url", default=None, help="surcharge de l'endpoint")
    args = ap.parse_args(argv)

    global API_URL, MODEL
    if args.url:
        API_URL = args.url.rstrip("/")
    if args.model:
        MODEL = args.model

    if args.print_env:
        print("\n".join(env_exports(model=MODEL, url=API_URL)))
        return 0
    if args.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

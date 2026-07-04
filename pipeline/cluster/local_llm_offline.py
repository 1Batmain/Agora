"""Runner OFFLINE : vLLM `LLM()` in-process branché sur le seam `mistral_client.chat`,
puis exécution d'un build standard INCHANGÉ — pas de serveur HTTP à faire tourner.

Pourquoi : sur certaines combinaisons vllm/transformers, `vllm serve` échoue sur
Gemma (fallback transformers bogué) alors que l'API offline `LLM()` fonctionne.
Ce runner charge le modèle une fois via `LLM()`, redéclare `mistral_client.chat`
(même contrat, mêmes erreurs, même accumulateur de tokens) et lance le build
demandé dans le même process. Aucun module existant n'est modifié : le patch est
un rebinding à l'exécution du SEUL point de passage des appels LLM.

Les appels sont sérialisés par un verrou : `LLM()` n'est pas thread-safe, et les
builds parallélisent par threads. On perd le batching, ce qui est acceptable pour
quelques centaines d'avis sur un H100.

Environnement GPU (le repo épingle torch CPU — faire un venv séparé) :
    uv venv .venv-gpu --python 3.12
    uv pip install --python .venv-gpu vllm numpy scikit-learn sentence-transformers \
        igraph leidenalg stopwordsiso httpx einops langdetect
    PYTHONPATH=. .venv-gpu/bin/python -m pipeline.cluster.local_llm_offline \
        --model google/gemma-4-12B-it selftest
    PYTHONPATH=. .venv-gpu/bin/python -m pipeline.cluster.local_llm_offline \
        --model google/gemma-4-12B-it build_analysis --dataset <id>
    PYTHONPATH=. .venv-gpu/bin/python -m pipeline.cluster.local_llm_offline \
        --model google/gemma-4-12B-it build_opinion --dataset <id>
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import threading
from typing import Callable

from . import mistral_client
from .local_llm_client import _fold_system
from .mistral_client import MistralError

# Tous les knobs de modèle des étapes LLM du build (claims, nommage, synthèse,
# opinion, enrichissement) — alignés sur le modèle local avant tout import.
_MODEL_ENV_VARS = (
    "AGORA_MISTRAL_MODEL",
    "AGORA_MISTRAL_SYNTH_MODEL",
    "AGORA_CLAIMS_API_MODEL",
    "AGORA_OPINION_MODEL",
    "AGORA_ENRICH_MODEL",
    "AGORA_ARGMINE_MODEL",
)

# Builds standard exécutables via ce runner (modules inchangés).
_TARGETS = {
    "build_analysis": "backend.build_analysis",
    "build_opinion": "backend.build_opinion",
    "build_arguments": "backend.build_arguments",
    "build_children": "backend.build_children",
}

# complete(messages, temperature, max_tokens, json_mode) -> (texte, usage)
Complete = Callable[[list, float, int, bool], tuple[str, dict]]


def set_env(model: str) -> None:
    """Route toutes les étapes vers le modèle local ; clé factice (rien ne sort)."""
    for var in _MODEL_ENV_VARS:
        os.environ[var] = model
    os.environ["MISTRAL_API_KEY"] = "local-offline"


def make_offline_chat(complete: Complete, *, fold_system: bool = False,
                      model_id: str = "local") -> Callable:
    """Fabrique un `chat()` au contrat de `mistral_client.chat`, adossé à `complete`.

    Sérialisé (verrou), tokens comptés dans l'accumulateur commun, erreurs
    converties en `MistralError` (les appelants gardent leurs replis existants).
    """
    lock = threading.Lock()

    def chat(messages: list[dict], *, model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 512,
             json_mode: bool = False, timeout: float | None = None) -> str:
        del model, timeout  # le modèle est celui chargé ; pas de réseau, pas de timeout
        if fold_system:
            messages = _fold_system(messages)
        try:
            with lock:
                text, usage = complete(messages, temperature, max_tokens, json_mode)
        except MistralError:
            raise
        except Exception as e:  # GPU/parsing : repli standard des appelants
            raise MistralError(0, f"local_llm:{type(e).__name__}: {e}")
        mistral_client._record_usage(model_id, usage or {})
        return text or ""

    return chat


def install(chat: Callable) -> None:
    """Rebinde le seam : tous les `mistral_client.chat(...)` passent par le LLM local."""
    mistral_client.chat = chat
    mistral_client.available = lambda: True
    mistral_client.load_api_key = lambda: "local-offline"


# ── Câblage vLLM (importé tard : `--help` et tests fonctionnent sans CUDA) ────

def _make_vllm_complete(args: argparse.Namespace) -> Complete:
    from vllm import LLM, SamplingParams

    # Mode JSON : l'API a changé de nom selon la version de vLLM — on sonde,
    # même approche que le classifieur existant. Sans support : avertissement,
    # le prompt + les validations des appelants restent la seule contrainte.
    json_extra: dict = {}
    try:
        from vllm.sampling_params import GuidedDecodingParams  # vLLM ≤ 0.10
        json_extra = {"guided_decoding": GuidedDecodingParams(json_object=True)}
    except (ImportError, TypeError):
        try:
            from vllm.sampling_params import StructuredOutputsParams  # vLLM ≥ 0.11
            json_extra = {"structured_outputs": StructuredOutputsParams(json_object=True)}
        except (ImportError, TypeError):
            print("[warn] pas d'API structured-output dans ce vLLM — mode JSON "
                  "assuré par le prompt seul.", file=sys.stderr)

    # Checkpoints au format Mistral (ex. Ministral 3) : tokenizer/config/load
    # spécifiques, recommandés par les cartes de modèles mistralai.
    fmt = {k: v for k, v in (("tokenizer_mode", args.tokenizer_mode),
                             ("config_format", args.config_format),
                             ("load_format", args.load_format)) if v}
    print(f"Chargement de {args.model} (dtype={args.dtype}, "
          f"max_len={args.max_model_len}, gpu_mem={args.gpu_memory_utilization}, "
          f"{fmt or 'format HF'})…")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        trust_remote_code=True,
        # Échappatoire au chemin torch.compile (exige ninja + toolchain C) :
        # ~5-10 % plus lent, zéro dépendance de build.
        enforce_eager=args.enforce_eager,
        **fmt,
    )

    def complete(messages: list, temperature: float, max_tokens: int,
                 json_mode: bool) -> tuple[str, dict]:
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                  **(json_extra if json_mode else {}))
        out = llm.chat([messages], sampling, use_tqdm=False)[0]
        usage = {
            "prompt_tokens": len(out.prompt_token_ids or []),
            "completion_tokens": len(out.outputs[0].token_ids or []),
        }
        return out.outputs[0].text, usage

    return complete


def _selftest(chat: Callable) -> int:
    checks = [
        ("chat simple", dict(json_mode=False),
         [{"role": "system", "content": "Réponds en un mot."},
          {"role": "user", "content": "Quelle est la capitale de la France ?"}]),
        ("mode JSON", dict(json_mode=True),
         [{"role": "user", "content": 'Réponds en JSON : {"ok": true}'}]),
    ]
    failures = 0
    for name, kw, messages in checks:
        try:
            out = chat(messages, max_tokens=64, temperature=0.0, **kw)
            print(f"  [ok  ] {name} : {out.strip()[:80]!r}")
        except MistralError as e:
            failures += 1
            print(f"  [FAIL] {name} : {e}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Exécute un build Agora avec un LLM vLLM offline in-process.")
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--dtype", default="auto",
                    help="auto = dtype du checkpoint (bf16, fp8…) — le bon défaut")
    ap.add_argument("--tokenizer-mode", default=None,
                    help="'mistral' pour les checkpoints au format Mistral")
    ap.add_argument("--config-format", default=None,
                    help="'mistral' pour les checkpoints au format Mistral")
    ap.add_argument("--load-format", default=None,
                    help="'mistral' pour les checkpoints au format Mistral")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--fold-system", action="store_true",
                    help="fondre le rôle system dans le tour user (templates Gemma)")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="désactiver torch.compile (pas besoin de ninja/toolchain)")
    ap.add_argument("target", choices=[*_TARGETS, "selftest"],
                    help="build à exécuter (ses propres flags suivent, ex. --dataset)")
    ap.add_argument("target_args", nargs=argparse.REMAINDER,
                    help="arguments passés tels quels au build")
    args = ap.parse_args(argv)

    set_env(args.model)  # AVANT tout import backend (knobs lus à l'import)
    chat = make_offline_chat(_make_vllm_complete(args),
                             fold_system=args.fold_system, model_id=args.model)
    install(chat)

    if args.target == "selftest":
        return _selftest(chat)

    module = _TARGETS[args.target]
    sys.argv = [module, *args.target_args]
    print(f"→ {module} {' '.join(args.target_args)}")
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

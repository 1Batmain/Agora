"""PIPELINE COMPLET « fichier → analyse » en une commande — pensé pour la box GPU.

Enchaîne, dans UN SEUL process :
  1. `prepare_from_file`  : CSV/JSON arbitraire → JSONL canonique + descripteur généré
     (fonte des colonnes texte libre, heuristiques génériques de `pipeline.collect`) ;
  2. `backend.build_cache`: ideas + embeddings nomic (torch — AVANT de charger vLLM,
     pour que l'embedder profite du GPU sans se battre avec le KV cache) ;
  3. builds LLM           : `build_analysis` → `build_opinion` → `build_arguments`,
     routés vers un vLLM offline in-process (`local_llm_offline`) ou l'API Mistral.

Le défaut `--gpu-memory-utilization 0.70` (vs 0.85 du runner seul) laisse ~1/4 de la
VRAM à l'embedding des claims qui tourne PENDANT que vLLM est résident.

Usage (box GPU, venv `.venv-gpu` du GPU_RUNBOOK §2 activé) :
    PYTHONPATH=. python -m pipeline.ingest_full.full_run \
        --csv chemin/vers/consultation.csv --dataset ma-consultation \
        --question "Quelle est la question posée aux citoyens ?" \
        --model mistralai/Ministral-3-14B-Instruct-2512 \
        --tokenizer-mode mistral --config-format mistral --load-format mistral
    # re-lancer seulement les builds (cache déjà construit) :
    PYTHONPATH=. python -m pipeline.ingest_full.full_run --csv … --dataset … --resume
    # via l'API Mistral (pas de GPU) :
    MISTRAL_API_KEY=… python -m pipeline.ingest_full.full_run --csv … --llm api

Rapatriement : rsync `backend/cache/<dataset>/` vers la machine dev, puis `make dev`.
"""

from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path
from time import perf_counter

from backend.recluster import CACHE_DIR

from . import prepare

# Ordre des builds LLM : l'opinion dépend de l'analyse, les arguments des deux.
_LLM_BUILDS = ("backend.build_analysis", "backend.build_opinion", "backend.build_arguments")


def _log(msg: str) -> None:
    print(f"[full_run] {msg}", flush=True)


def _slugify(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "dataset"


def _step_prepare(args) -> dict:
    _log(f"1/3 · prepare : {args.csv} → dataset {args.dataset!r}")
    return prepare.prepare_from_file(
        Path(args.csv), args.dataset,
        question=args.question, context=args.context, label=args.label)


def _step_cache(args, descriptor_path: Path) -> dict:
    _log("2/3 · build_cache : ideas + embeddings nomic (torch)")
    from backend.build_cache import build_cache  # import tardif (torch)
    return build_cache(dataset=args.dataset, descriptor=str(descriptor_path),
                       min_chars=args.min_chars, label=args.label)


def _step_llm_builds(args) -> None:
    if args.llm == "vllm":
        # Même mécanique que le runner : vLLM offline in-process, seam rebindé.
        from pipeline.cluster.local_llm_offline import (
            _make_vllm_complete, install, make_offline_chat, set_env)
        set_env(args.model)
        chat = make_offline_chat(_make_vllm_complete(args),
                                 fold_system=args.fold_system, model_id=args.model)
        install(chat)
    else:
        from pipeline.cluster import mistral_client
        if not mistral_client.available():
            raise SystemExit("--llm api mais pas de clé Mistral (MISTRAL_API_KEY). Abandon.")

    for module in _LLM_BUILDS:
        _log(f"3/3 · {module} --dataset {args.dataset}")
        sys.argv = [module, "--dataset", args.dataset]
        runpy.run_module(module, run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pipeline complet : un fichier de consultation → analyse Agora servable.")
    ap.add_argument("--csv", required=True,
                    help="fichier d'entrée (.csv, .json ou .json.zip)")
    ap.add_argument("--dataset", default=None,
                    help="id du dataset (défaut : nom du fichier slugifié)")
    ap.add_argument("--question", default=None, help="question posée (cadre l'extraction)")
    ap.add_argument("--context", default=None, help="contexte de la consultation")
    ap.add_argument("--label", default=None, help="libellé d'affichage (UI)")
    ap.add_argument("--min-chars", type=int, default=1, help="filtre avis trop courts")
    ap.add_argument("--resume", action="store_true",
                    help="sauter prepare+cache si backend/cache/<dataset>/ existe déjà")
    ap.add_argument("--llm", choices=("vllm", "api"), default="vllm",
                    help="vllm = modèle local offline (défaut) ; api = clé Mistral")
    # Réglages vLLM — mêmes noms que pipeline.cluster.local_llm_offline.
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--fold-system", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--tokenizer-mode", default=None)
    ap.add_argument("--config-format", default=None)
    ap.add_argument("--load-format", default=None)
    args = ap.parse_args(argv)

    if not args.dataset:
        args.dataset = _slugify(Path(args.csv).stem)
        _log(f"dataset non fourni → {args.dataset!r}")

    t0 = perf_counter()
    cache_ready = (CACHE_DIR / args.dataset / "embeddings.npy").exists()
    if args.resume and cache_ready:
        _log(f"--resume : cache {args.dataset} présent, prepare+cache sautés")
    else:
        if args.resume:
            _log(f"--resume demandé mais pas de cache {args.dataset} — pipeline complet")
        summary = _step_prepare(args)
        _step_cache(args, summary["descriptor_path"])

    _step_llm_builds(args)

    took_min = (perf_counter() - t0) / 60
    _log(f"✓ pipeline complet en {took_min:.1f} min — artefacts : "
         f"{CACHE_DIR / args.dataset}")
    _log("Rapatrier sur la machine dev : "
         f"rsync -avz <box>:{CACHE_DIR / args.dataset}/ backend/cache/{args.dataset}/ "
         "puis `make dev`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Exécuteur Nativf pour les modèles GGUF.
Télécharge les modèles directement depuis HuggingFace et les exécute avec llama-cpp-python.
Contourne totalement les limitations et restrictions d'API de LM Studio / Ollama !
"""

import json
import time
import argparse
import os
from pathlib import Path

# Fix DLL load failed on Windows by injecting Torch's bundled CUDA DLLs
try:
    import torch
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
    os.add_dll_directory(r"C:\Users\aoutaleb\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.23.1")
except Exception:
    pass

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from research.api_quality_bench import eval_api_model, write_csv, DEFAULTS
from research import multilingual_data

class NativeGGUFEmbedder:
    def __init__(self, repo_id: str, filename: str):
        print(f"[{repo_id}] Téléchargement/Vérification du cache HuggingFace...")
        # Téléchargement ou récupération dans le cache local
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        print(f"[{repo_id}] Chargement du modèle en VRAM (CUDA)...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Max layers on GPU
            embedding=True,  # Forcer le mode embedding !
            verbose=False,
            n_ctx=4096
        )
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # llama.cpp gère le batching ou on le fait ici
        embeddings = self.llm.create_embedding(texts)
        return [e["embedding"] for e in embeddings["data"]]

def parse_hf_path(path: str) -> tuple[str, str]:
    """Extrait repo_id et filename d'un lien HF GGUF"""
    if "resolve/main/" in path:
        parts = path.split("resolve/main/")
        repo_id = parts[0].replace("https://huggingface.co/", "").strip("/")
        filename = parts[1]
        return repo_id, filename
    elif "hf.co/" in path:
        # Format "hf.co/user/repo:tag"
        # Since tags are hard to map to exact files if it's just Q4_K_M, we'll try to guess the filename or we'll just define them exactly.
        # Actually, let's just use explicit dict for the big models to be safe!
        return "", ""
    return "", ""

# Dictionnaire de secours pour les modèles sans URL directe dans le JSON
EXPLICIT_MODELS = {
    "e5-mistral-7b-instruct": r"C:\Users\aoutaleb\.lmstudio\models\second-state\E5-Mistral-7B-Instruct-Embedding-GGUF\e5-mistral-7b-instruct-Q4_K_M.gguf",
    "gte-Qwen2-7B-instruct": ("mradermacher/gte-Qwen2-7B-instruct-GGUF", "gte-Qwen2-7B-instruct.Q4_K_M.gguf"),
    "SFR-Embedding-Mistral": ("dranger003/SFR-Embedding-Mistral-GGUF", "sfr-embedding-mistral-q4_k_m.gguf"),
    "GritLM-7B": ("tensorblock/GritLM-7B-GGUF", "GritLM-7B-Q4_K_M.gguf"),
    "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf")
}

def orchestrate():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="research/api_models.json")
    ap.add_argument("--out", default="bench_native_results.csv")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        models_config = json.load(f)
        
    print("Chargement du corpus...")
    corpus = multilingual_data.load_balanced(
        n_topics=DEFAULTS["k"], per_cell=15, max_per_cell=50,
        min_chars=15, seed=DEFAULTS["seed"],
    )

    results = []
    
    for m in models_config:
        model_id = m["name"]
        print(f"\n{'='*50}\nTraitement de: {model_id}\n{'='*50}")
        
        repo_id, filename = parse_hf_path(m.get("path", ""))
        
        # Override with explicit dictionary if not a valid HF path
        if not repo_id:
            val = EXPLICIT_MODELS.get(model_id)
            if isinstance(val, str): # Absolute path
                model_path = val
            elif isinstance(val, tuple):
                repo_id, filename = val
        
        # Si on a repo_id et filename, on télécharge
        if repo_id and filename:
            print(f"[{repo_id}] Téléchargement/Vérification du cache HuggingFace...")
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as e:
                print(f"Erreur de téléchargement: {e}")
                continue
                
        if not model_path:
            print(f"Skipping {model_id}: Impossible de déterminer le chemin local ou le dépôt HF.")
            continue
            
        try:
            print(f"Chargement du modèle en VRAM (CUDA)...")
            self_llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                embedding=True,
                verbose=False,
                n_ctx=4096
            )
            
            class NativeWrapper:
                def embed_documents(self, texts):
                    embeddings = self_llm.create_embedding(texts)
                    return [e["embedding"] for e in embeddings["data"]]
            
            embedder = NativeWrapper()
            res = eval_api_model(m, corpus, DEFAULTS, embedder_instance=embedder)
            
            if res.error:
                print(f"  -> ERREUR: {res.error}")
            else:
                print(f"  -> OK: dim={res.dim} clusters={res.n_clusters} coh={res.coherence:.3f} nmi_lang={res.nmi_lang:.3f} nmi_topic={res.nmi_topic:.3f}")
                results.append(res)
                write_csv(results, args.out)
                
            # Forcer le nettoyage de la VRAM pour le prochain modèle
            del embedder
            
        except Exception as e:
            print(f"Erreur fatale sur {model_id}: {e}")

if __name__ == "__main__":
    orchestrate()

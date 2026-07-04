"""
Exécuteur robuste via subprocess llama-server.
Utilise le serveur llama.cpp interne d'Ollama pour forcer le mode embedding
sur N'IMPORTE QUEL modèle GGUF, sans limitation d'API.
"""

import json
import time
import argparse
import subprocess
import httpx
import os
import signal
from pathlib import Path

from huggingface_hub import hf_hub_download
from research.api_quality_bench import eval_api_model, write_csv, DEFAULTS, APIEmbedder
from research import multilingual_data

EXPLICIT_MODELS = {
    "e5-mistral-7b-instruct": r"C:\Users\aoutaleb\.lmstudio\models\second-state\E5-Mistral-7B-Instruct-Embedding-GGUF\e5-mistral-7b-instruct-Q4_K_M.gguf",
    "gte-Qwen2-7B-instruct": ("mradermacher/gte-Qwen2-7B-instruct-GGUF", "gte-Qwen2-7B-instruct.Q4_K_M.gguf"),
    "SFR-Embedding-Mistral": ("dranger003/SFR-Embedding-Mistral-GGUF", "sfr-embedding-mistral-q4_k_m.gguf"),
    "GritLM-7B": ("tensorblock/GritLM-7B-GGUF", "GritLM-7B-Q4_K_M.gguf"),
    "nomic-embed-text-v1.5": ("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    "bge-m3": ("ggml-org/bge-m3-Q8_0-GGUF", "bge-m3-q8_0.gguf"),
    "multilingual-e5-large": ("keisuke-miyako/multilingual-e5-large-gguf-q4_k_m", "multilingual-e5-large-Q4_k_m.gguf"),
    "jina-embeddings-v3": ("second-state/jina-embeddings-v3-GGUF", "jina-embeddings-v3-Q4_K_M.gguf"),
    "gte-multilingual-base": ("keisuke-miyako/gte-multilingual-base-gguf-f16", "gte-multilingual-base-gguf-f16.gguf")
}

def wait_for_server(url: str, timeout: int = 60):
    start = time.time()
    print("En attente du démarrage du serveur local...")
    while time.time() - start < timeout:
        try:
            res = httpx.get(url.replace("/v1/embeddings", "/health"), timeout=1.0)
            if res.status_code == 200:
                print("Serveur prêt !")
                return True
        except:
            pass
        time.sleep(1)
    return False

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
    
    server_exe = r"C:\Users\aoutaleb\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"

    for m in models_config:
        model_id = m["name"]
        print(f"\n{'='*50}\nTraitement de: {model_id}\n{'='*50}")
        
        val = EXPLICIT_MODELS.get(model_id)
        if not val:
            print(f"Modèle ignoré (non configuré explicitement): {model_id}")
            continue
            
        model_path = ""
        if isinstance(val, str):
            model_path = val
        else:
            repo_id, filename = val
            print(f"[{repo_id}] Téléchargement/Vérification du cache HuggingFace...")
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as e:
                print(f"Erreur de téléchargement: {e}")
                continue
                
        if not os.path.exists(model_path):
            print(f"Fichier GGUF introuvable: {model_path}")
            continue

        print(f"Démarrage de llama-server.exe pour {model_id}...")
        
        # Flags importants: --embedding pour FORCER l'embedding sur LLM, -ngl 99 pour GPU
        cmd = [server_exe, "-m", model_path, "--port", "8080", "--embedding", "-ngl", "99", "-c", "4096"]
        
        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP # Permet de kill proprement sous Windows
        )
        
        url = "http://localhost:8080/v1/embeddings"
        
        try:
            if not wait_for_server(url, timeout=120):
                print(f"Timeout: le serveur n'a pas démarré pour {model_id}")
                continue
                
            # Mettre à jour l'URL et utiliser l'APIEmbedder standard
            m["url"] = url
            embedder = APIEmbedder(url=url, model_path=model_path, batch_size=m.get("batch_size", 32))
            
            res = eval_api_model(m, corpus, DEFAULTS, embedder_instance=embedder)
            
            if res.error:
                print(f"  -> ERREUR LORS DU BENCHMARK: {res.error}")
            else:
                print(f"  -> OK: dim={res.dim} clusters={res.n_clusters} coh={res.coherence:.3f} nmi_lang={res.nmi_lang:.3f} nmi_topic={res.nmi_topic:.3f}")
                results.append(res)
                write_csv(results, args.out)
                
        except Exception as e:
            print(f"Erreur d'évaluation: {e}")
            
        finally:
            print("Arrêt du serveur...")
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=10)
            time.sleep(2) # Laisser la VRAM se vider

if __name__ == "__main__":
    orchestrate()

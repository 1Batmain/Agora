"""Orchestrateur Hybride pour LM Studio et Ollama.

Ce script lit `api_models.json`.
Pour `source == "lmstudio"`, il télécharge/charge/décharge le modèle via `lms`.
Pour `source == "ollama"`, il télécharge (si HF) via `ollama pull`, puis exécute le benchmark.
"""

import json
import subprocess
import time
import argparse

from research.api_quality_bench import eval_api_model, write_csv, DEFAULTS
from research import multilingual_data

LMS_PATH = r"C:\Users\aoutaleb\.lmstudio\bin\lms.exe"
OLLAMA_PATH = r"C:\Users\aoutaleb\AppData\Local\Programs\Ollama\ollama.exe"

def run_cmd(cmd: list[str]) -> bool:
    """Exécute une commande système. Retourne True si succès."""
    print(f">> {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERREUR d'exécution: {e}")
        return False
    except FileNotFoundError:
        print(f"ERREUR: Exécutable non trouvé {cmd[0]}")
        return False

def orchestrate():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="research/api_models.json")
    ap.add_argument("--out", default="bench_api_results.csv")
    ap.add_argument("--n-topics", type=int, default=6)
    ap.add_argument("--per-cell", type=int, default=10)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        models_config = json.load(f)

    # Démarrer LM Studio
    print("Vérification/Démarrage du serveur LM Studio...")
    run_cmd([LMS_PATH, "server", "start"])
    # Ollama démarre généralement automatiquement avec ollama serve ou tourne en arrière-plan.
    
    # Chargement du corpus
    print("Chargement du corpus...")
    corpus = multilingual_data.load_balanced(
        n_topics=args.n_topics, per_cell=args.per_cell, max_per_cell=50,
        min_chars=15, seed=DEFAULTS["seed"],
    )

    results = []
    
    for m in models_config:
        model_id = m["name"]
        model_path = m["path"]
        source = m["source"]
        
        print(f"\n{'='*50}")
        print(f"Traitement du modèle: {model_id} via {source.upper()}")
        print(f"{'='*50}")
        
        m_eval = dict(m)
        
        if source == "lmstudio":
            print("[1/4] Téléchargement (si distant)...")
            # Ignorer l'erreur si le modèle est déjà là ou si on utilise le nom local
            run_cmd([LMS_PATH, "get", "-y", model_path])
            
            print("[2/4] Chargement en VRAM...")
            loaded = run_cmd([LMS_PATH, "load", "-y", "--gpu", "max", "--identifier", model_id, model_path])
            if not loaded:
                print(f"Échec du chargement de {model_id}. Passage au suivant.")
                continue
                
            time.sleep(2)
            m_eval["path"] = model_id  # L'API LM Studio s'attend à l'identifiant court
            
            print("[3/4] Exécution du benchmark...")
            res = eval_api_model(m_eval, corpus, DEFAULTS)
            
            print("[4/4] Déchargement...")
            run_cmd([LMS_PATH, "unload", model_id])
            
        elif source == "ollama":
            print("[1/3] Téléchargement (pull)...")
            # Ollama gère le nom hf.co/ ou les tags.
            run_cmd([OLLAMA_PATH, "pull", model_path])
            
            # Pas besoin de charger manuellement, Ollama le fait à la volée.
            time.sleep(2)
            # L'API d'Ollama attend le path utilisé pour le pull comme nom de modèle
            m_eval["path"] = model_path 
            
            print("[2/3] Exécution du benchmark...")
            res = eval_api_model(m_eval, corpus, DEFAULTS)
            
            print("[3/3] Déchargement automatique par Ollama.")
            # Pour vider la RAM, on peut forcer l'unload sur Ollama en l'appelant avec run et timeout, ou stop, mais c'est optionnel.
            run_cmd([OLLAMA_PATH, "stop", model_path])
            
        else:
            print(f"Source inconnue: {source}")
            continue

        if res.error:
            print(f"  -> ERREUR LORS DU BENCHMARK: {res.error}")
        else:
            print(f"  -> OK: dim={res.dim} clusters={res.n_clusters} coh={res.coherence:.3f} nmi_lang={res.nmi_lang:.3f} nmi_topic={res.nmi_topic:.3f}")
            
        results.append(res)
        
        # Sauvegarde incrémentale
        write_csv(results, args.out)
        print("CSV mis à jour.")

    print(f"\nOrchestration terminée. Résultats sauvés dans {args.out}")

if __name__ == "__main__":
    orchestrate()

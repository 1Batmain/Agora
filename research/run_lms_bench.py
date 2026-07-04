"""Orchestrateur pour LM Studio (téléchargement, chargement, benchmark).

Ce script lit `api_models.json`, utilise la CLI `lms` pour télécharger
automatiquement les modèles, les charger en mémoire (VRAM) un par un,
et exécute le benchmark via l'API, avant de les décharger.
"""

import json
import subprocess
import time
from pathlib import Path
import os
import argparse

from research.api_quality_bench import eval_api_model, write_csv, DEFAULTS
from research import multilingual_data

LMS_PATH = r"C:\Users\aoutaleb\.lmstudio\bin\lms.exe"

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

    # Démarrage du serveur si ce n'est pas déjà fait
    print("Vérification/Démarrage du serveur LM Studio...")
    run_cmd([LMS_PATH, "server", "start"])
    time.sleep(3) # Attendre que le serveur soit prêt

    # Charger le corpus
    print("Chargement du corpus...")
    corpus = multilingual_data.load_balanced(
        n_topics=args.n_topics, per_cell=args.per_cell, max_per_cell=50,
        min_chars=15, seed=DEFAULTS["seed"],
    )

    results = []
    
    for m in models_config:
        model_id = m["name"]
        model_path = m["path"]
        
        print(f"\n{'='*50}")
        print(f"Traitement du modèle: {model_id} ({model_path})")
        print(f"{'='*50}")
        
        # 1. Télécharger (get) avec validation auto (-y)
        print("[1/4] Téléchargement...")
        run_cmd([LMS_PATH, "get", "-y", model_path])
        
        # 2. Charger en mémoire (load)
        print("[2/4] Chargement en VRAM...")
        # on donne un identifiant clair au modèle pour pouvoir requêter l'API et le décharger
        # On essaie d'utiliser --gpu max pour que ce soit rapide
        loaded = run_cmd([LMS_PATH, "load", "-y", "--gpu", "max", "--identifier", model_id, model_path])
        
        if not loaded:
            print(f"Échec du chargement de {model_id}. Passage au suivant.")
            continue
            
        time.sleep(2) # Laisser le temps à l'API d'être prête

        # 3. Exécuter le benchmark
        print("[3/4] Exécution du benchmark...")
        # L'API locale répond toujours au path de l'identifiant chargé
        m_eval = dict(m)
        m_eval["path"] = model_id 
        
        res = eval_api_model(m_eval, corpus, DEFAULTS)
        
        if res.error:
            print(f"  -> ERREUR LORS DU BENCHMARK: {res.error}")
        else:
            print(f"  -> OK: dim={res.dim} clusters={res.n_clusters} coh={res.coherence:.3f} nmi_lang={res.nmi_lang:.3f} nmi_topic={res.nmi_topic:.3f}")
            
        results.append(res)
        
        # 4. Décharger (unload) pour libérer la VRAM
        print("[4/4] Déchargement...")
        run_cmd([LMS_PATH, "unload", model_id])
        
        # Écriture incrémentale
        write_csv(results, args.out)
        print("CSV mis à jour.")

    print(f"\nOrchestration terminée. Résultats sauvés dans {args.out}")

if __name__ == "__main__":
    orchestrate()

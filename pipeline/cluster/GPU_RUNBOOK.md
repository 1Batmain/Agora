# Runbook — build LLM local sur une instance GPU neuve (H100)

Objectif : exécuter les étapes LLM du pipeline (`build_analysis` + `build_opinion`)
sur une box GPU avec un modèle local via vLLM **offline** (`LLM()`), sans clé
Mistral et sans qu'aucune donnée citoyenne ne sorte de la machine.
Le runner est `pipeline/cluster/local_llm_offline.py` : il rebinde le seam
`mistral_client.chat` puis lance les builds standard, inchangés.

Testé le 2026-07-04 : H100 80 Go, Gemma 12B (`google/gemma-4-12B-it`), BF16.
`vllm serve` (HTTP) est cassé sur certaines combos vllm/transformers avec Gemma
(fallback transformers bogué) — c'est la raison d'être du chemin offline.

## 0. Prérequis instance

- Driver NVIDIA fonctionnel (`nvidia-smi` répond) — standard sur les images GPU.
- Accès HF au modèle : Gemma est *gated* → `export HF_TOKEN=…` (ou
  `~/.cache/huggingface` déjà peuplé par un run précédent).
- Outils de build pour torch.compile :
  ```bash
  sudo apt install -y ninja-build build-essential
  ```
  (sinon, tout marche aussi avec `--enforce-eager`, ~5-10 % plus lent)

## 1. Code

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # si uv absent
git clone -b feat/collect-open-data <url-du-repo>
cd Analyse-des-consultations-citoyennes
```

## 2. Venv GPU séparé

Le repo épingle torch **CPU** (`tool.uv.sources` dans pyproject.toml) : ne JAMAIS
installer vLLM dans l'env projet, ni `uv run` ces commandes contre l'env projet.

```bash
uv venv .venv-gpu --python 3.12       # PAS 3.14 (combo vllm/transformers non testée)
uv pip install --python .venv-gpu vllm numpy scikit-learn sentence-transformers \
    igraph leidenalg stopwordsiso httpx einops langdetect
source .venv-gpu/bin/activate          # met aussi ninja/bin du venv sur le PATH
```

⚠️ Piège : invoquer `.venv-gpu/bin/python` par chemin SANS `source …/activate`
laisse le PATH sans `.venv-gpu/bin` → le sous-process vLLM ne trouve pas `ninja`.
Toujours activer (ou installer ninja via apt, cf. §0).

## 3. Donnée (depuis la machine dev)

Le cache dataset (ideas + embeddings + meta) est gitignoré — l'envoyer :

```bash
# sur la machine dev :
rsync -avz backend/cache/<dataset>/ \
    <gpu>:~/Analyse-des-consultations-citoyennes/backend/cache/<dataset>/
```

(Ce cache est produit sur la machine dev par `pipeline.ingest_full.prepare` +
`backend.build_cache --descriptor …` — voir `pipeline/ingest_full/__init__.py`.)

## 3bis. Choix du modèle

| Modèle | Poids | Empreinte | Flags spécifiques |
|---|---|---|---|
| `mistralai/Ministral-3-14B-Instruct-2512` (recommandé) | FP8, Apache 2.0 | ~14 Go → très à l'aise sur 80 Go | `--tokenizer-mode mistral --config-format mistral --load-format mistral` (vLLM ≥ 0.12) |
| `mistralai/Ministral-3-8B-Instruct-2512` | FP8, Apache 2.0 | ~8 Go | idem |
| `google/gemma-4-12B-it` | BF16, gated (HF_TOKEN) | ~24 Go | parfois `--fold-system` |

Ministral 3 supporte officiellement le rôle `system` et la sortie JSON — les deux
points où Gemma peut coincer. `--dtype auto` (défaut) prend le dtype du checkpoint.

## 4. Selftest puis builds

```bash
# Ministral 3 14B (recommandé) :
PYTHONPATH=. python -m pipeline.cluster.local_llm_offline \
    --model mistralai/Ministral-3-14B-Instruct-2512 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral \
    selftest

# ou Gemma :
PYTHONPATH=. python -m pipeline.cluster.local_llm_offline \
    --model google/gemma-4-12B-it selftest
```

Deux `[ok]` attendus (chat + mode JSON). Sinon :
- erreur mentionnant le rôle `system` → ajouter `--fold-system` ;
- erreur ninja/compilation → ajouter `--enforce-eager` (cf. §0).

Puis, avec les MÊMES flags (modèle + format) que le selftest vert :

```bash
PYTHONPATH=. python -m pipeline.cluster.local_llm_offline \
    --model <modèle> [flags format…] build_analysis --dataset <dataset>
PYTHONPATH=. python -m pipeline.cluster.local_llm_offline \
    --model <modèle> [flags format…] build_opinion  --dataset <dataset>
```

Les DEUX builds doivent tourner avant le rapatriement (gotcha
`.agent/notes/DEV_PROD.md`). Ordre de grandeur (198 avis, H100) :
~20-35 min pour l'analyse (dominée par l'extraction des claims, sérialisée),
~10-20 min pour l'opinion. Surveiller dans les logs que les claims parsent
(une rafale de replis « avis entier » = JSON Gemma en difficulté).

## 4bis. Tout-en-un : d'un CSV à l'analyse complète (une commande)

Pour une NOUVELLE consultation dont on a le fichier (pas besoin de passer par la
machine dev) : `pipeline.ingest_full.full_run` enchaîne prepare → embeddings →
analysis → opinion → arguments dans un seul process (vLLM chargé une fois,
`--gpu-memory-utilization 0.70` pour laisser de la VRAM à l'embedding des claims).

```bash
PYTHONPATH=. python -m pipeline.ingest_full.full_run \
    --csv chemin/vers/consultation.csv --dataset ma-consultation \
    --question "Quelle est la question posée aux citoyens ?" \
    --model mistralai/Ministral-3-14B-Instruct-2512 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral
# re-lancer seulement les builds LLM (cache déjà là) : ajouter --resume
```

## 5. Rapatrier et explorer (machine dev)

```bash
rsync -avz <gpu>:~/Analyse-des-consultations-citoyennes/backend/cache/<dataset>/ \
    backend/cache/<dataset>/
make dev    # → http://localhost:5180 — le dataset est découvert automatiquement
```

## Limites connues / notes

- **Appels sérialisés** : le runner verrouille `LLM()` (non thread-safe) → pas de
  batching. OK jusqu'à quelques centaines d'avis ; pour les gros corpus collectés
  (10⁵-10⁶ réponses), il faudra batcher les prompts dans un seul `llm.chat(batch)`
  (throughput ×10-50). Non fait — à mesurer le jour où le besoin arrive.
- **GPU partagé** : `--gpu-memory-utilization 0.85` par défaut ; si un autre job
  vLLM tourne en même temps, descendre les deux (~0.42) et les démarrer l'un
  APRÈS l'autre.
- **Mix de modèles** : chaque étape reste surchargeable individuellement
  (`AGORA_MISTRAL_MODEL`, `AGORA_CLAIMS_API_MODEL`, `AGORA_OPINION_MODEL`, …) —
  on peut garder les claims sur Gemma local et router le nommage vers l'API
  Mistral le jour où une clé est disponible.
- **Endpoint HTTP** : si un jour `vllm serve` refonctionne pour le modèle choisi,
  `pipeline/cluster/local_llm_client.py` (`--selftest`, `--print-env`) route les
  builds vers l'endpoint sans rien changer d'autre.

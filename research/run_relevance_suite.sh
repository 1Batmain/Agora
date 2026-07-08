#!/bin/bash
set -e
cd /home/bat/forge/agora-argmining
export MISTRAL_API_KEY=$(cat var/mistral.key 2>/dev/null || cat /home/bat/forge/agora/var/mistral.key)
DS=lutte-contre-les-fausses-informations
echo "### (1) calibration lutte: strict vs soft"
PYTHONPATH=. uv run python research/relevance_prefilter.py --dataset $DS --model mistral-large-latest --variant strict
PYTHONPATH=. uv run python research/relevance_prefilter.py --dataset $DS --model mistral-large-latest --variant soft
echo "### (2) lutte small (vs large) — même variant strict"
PYTHONPATH=. uv run python research/relevance_prefilter.py --dataset $DS --model mistral-small-latest --variant strict
echo "### (1) 2e corpus tiktok: strict vs soft"
PYTHONPATH=. uv run python research/relevance_prefilter.py --dataset tiktok --model mistral-large-latest --variant strict
PYTHONPATH=. uv run python research/relevance_prefilter.py --dataset tiktok --model mistral-large-latest --variant soft
echo "### SUITE DONE"

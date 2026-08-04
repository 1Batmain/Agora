"""Serveur de la page LIVE — lit des instantanés, ne calcule RIEN.

Séparation stricte, volontaire : le rejeu (`live.replay`) calcule et écrit
`snapshot.json` + `transcript.jsonl` ; ce serveur les sert. Deux conséquences utiles —
on peut recharger la page sans perturber le rejeu, et le serveur n'a besoin d'aucune clé.

C'est une application FastAPI SÉPARÉE de `backend.server` : la Console servie n'est ni
modifiée ni exposée à ce chantier expérimental. Port distinct.

Lancer :
    uv run --extra serve uvicorn live.server:app --port 8020
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

CACHE_DIR = Path(os.environ.get("AGORA_LIVE_CACHE", "var/live-cache"))
PAGE = Path(__file__).resolve().parent / "page.html"

app = FastAPI(title="Agora — synthèse live d'un débat", version="0.1")


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Aucun instantané. Lance `python -m live.replay` d'abord.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Course bénigne : le rejeu écrit atomiquement, mais un cache disque peut retarder.
        raise HTTPException(status_code=503, detail="Instantané en cours d'écriture.")


@app.get("/api/snapshot")
def snapshot() -> dict:
    """État courant du pipeline live (thèmes, opinions, orateurs, dérive)."""
    return _read_json(CACHE_DIR / "snapshot.json")


@app.get("/api/transcript")
def transcript(limit: int = 40) -> dict:
    """Flux ENTRANT : les derniers tours de parole tels qu'ils arrivent."""
    path = CACHE_DIR / "transcript.jsonl"
    if not path.exists():
        return {"turns": []}
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {"turns": rows[-limit:], "total": len(rows)}


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    if not PAGE.exists():
        raise HTTPException(status_code=500, detail="page.html introuvable")
    return PAGE.read_text(encoding="utf-8")

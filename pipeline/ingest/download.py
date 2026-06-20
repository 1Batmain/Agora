"""T-D1 — récupération idempotente des sources brutes dans `data/raw/`.

Aucune donnée n'est versionnée (`data/` est gitignored). Les téléchargements
sont idempotents : un fichier déjà présent (taille > 0) n'est pas re-téléchargé,
sauf `--force`. En cas d'échec réseau, on NE BLOQUE PAS : `build` retombera sur
l'échantillon synthétique.

Usage :
    uv run python -m pipeline.ingest.download            # télécharge tout
    uv run python -m pipeline.ingest.download --force     # re-télécharge
    uv run python -m pipeline.ingest.download --only xstance|tiktok
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import config

_UA = {"User-Agent": "agora-an-2026/ingest (open-data)"}


def _fetch(url: str, dest: Path, force: bool) -> bool:
    """Télécharge `url` -> `dest` de façon idempotente. True si dispo en sortie."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] {dest.name} déjà présent ({dest.stat().st_size} o)")
        return True
    print(f"  [get ] {url}")
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  [fail] {dest.name}: {e}", file=sys.stderr)
        return False
    # Écriture atomique (tmp + rename) pour rester idempotent même si interrompu.
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"  [ok  ] {dest.name} ({len(data)} o)")
    return True


def download_xstance(force: bool = False) -> bool:
    print("x-stance (ZurichNLP) :")
    return _fetch(config.XSTANCE_URL, config.XSTANCE_ZIP, force)


def download_tiktok(force: bool = False) -> bool:
    print("Consultation TikTok (Assemblée nationale) :")
    return _fetch(config.TIKTOK_URL, config.TIKTOK_CSV, force)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Téléchargement idempotent des sources data.")
    ap.add_argument("--force", action="store_true", help="re-télécharger même si présent")
    ap.add_argument("--only", choices=["xstance", "tiktok"], help="une seule source")
    args = ap.parse_args(argv)

    results = {}
    if args.only in (None, "xstance"):
        results["xstance"] = download_xstance(args.force)
    if args.only in (None, "tiktok"):
        results["tiktok"] = download_tiktok(args.force)

    ok = sum(results.values())
    print(f"\n{ok}/{len(results)} source(s) disponible(s) dans {config.RAW_DIR}")
    if ok == 0:
        print("Aucune source réseau : `build` utilisera l'échantillon synthétique.",
              file=sys.stderr)
    # On retourne 0 même en cas d'échec réseau : le fallback synthétique couvre.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

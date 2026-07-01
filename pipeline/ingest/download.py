"""T-D1 — récupération idempotente des sources brutes dans `data/raw/`.

Générique (audit #1) : on télécharge l'`url` de chaque **descripteur** vers son
`path`. Aucune source codée en dur. Les téléchargements sont idempotents : un
fichier déjà présent (taille > 0) n'est pas re-téléchargé, sauf `--force`. En cas
d'échec réseau, on NE BLOQUE PAS : `build` retombera sur l'échantillon synthétique.

Usage :
    uv run python -m pipeline.ingest.download                 # toutes les sources
    uv run python -m pipeline.ingest.download --force          # re-télécharge
    uv run python -m pipeline.ingest.download --only tiktok    # une source (par nom)
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .sources import load_descriptors

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Téléchargement idempotent des sources data.")
    ap.add_argument("--force", action="store_true", help="re-télécharger même si présent")
    ap.add_argument("--only", help="une seule source (nom du descripteur)")
    args = ap.parse_args(argv)

    descriptors = load_descriptors()
    results = {}
    for d in descriptors:
        if args.only and d.name != args.only:
            continue
        if not d.url:
            print(f"{d.name} : pas d'URL dans le descripteur — ignoré.")
            continue
        print(f"{d.name} :")
        results[d.name] = _fetch(d.url, d.resolved_path(), args.force)

    ok = sum(results.values())
    print(f"\n{ok}/{len(results)} source(s) disponible(s) dans {config.RAW_DIR}")
    if results and ok == 0:
        print("Aucune source réseau : `build` utilisera l'échantillon synthétique.",
              file=sys.stderr)
    # On retourne 0 même en cas d'échec réseau : le fallback synthétique couvre.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Téléchargement idempotent et poli des fichiers open data.

Miroir durci de `pipeline/ingest/download.py` : écriture atomique (.part → rename),
skip si présent en cache, plus un cap de taille (streaming) qui écarte les dumps
pathologiques AVANT transfert quand le serveur annonce Content-Length, un marqueur
de cache pour les fichiers vides côté serveur, un retry sur erreur transitoire et
un délai de politesse entre requêtes réseau (jamais sur les hits de cache).
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlsplit, urlunsplit

from . import config

_CHUNK = 1 << 16


def _encode_url(url: str) -> str:
    """IRI → URI : urllib exige de l'ASCII ; certains fichiers du portail ont des
    accents dans leur chemin. Le '%' reste sûr (pas de double encodage)."""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, quote(p.path, safe="/%"),
                       quote(p.query, safe="=&%"), p.fragment))


@dataclass(frozen=True)
class DownloadResult:
    status: str          # ok | cached | empty | too_large | error
    size_bytes: int = 0
    detail: str | None = None


def _urlopen(url: str, timeout: int):
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_page(url: str) -> bytes:
    """Fetch simple (pages HTML du portail) avec UA + délai de politesse."""
    time.sleep(config.REQUEST_DELAY_S)
    with _urlopen(_encode_url(url), config.TIMEOUT_S) as r:
        return r.read()


def download(url: str, dest: Path, *, force: bool = False,
             max_bytes: int = config.MAX_DOWNLOAD_BYTES,
             open_url: Callable = _urlopen,
             delay_s: float = 0.0) -> DownloadResult:
    """Télécharge `url` → `dest` de façon idempotente. Ne lève jamais : statut."""
    url = _encode_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        size = dest.stat().st_size
        return DownloadResult("empty" if size == 0 else "cached", size)

    if delay_s:
        time.sleep(delay_s)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for _attempt in range(2):  # 1 retry sur erreur transitoire
        try:
            with open_url(url, config.TIMEOUT_S) as resp:
                announced = resp.getheader("Content-Length")
                if announced is not None and int(announced) > max_bytes:
                    return DownloadResult("too_large", int(announced),
                                          f"Content-Length {announced} > cap {max_bytes}")
                written = 0
                with tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_bytes:
                            out.close()
                            tmp.unlink(missing_ok=True)
                            return DownloadResult("too_large", written,
                                                  f"flux > cap {max_bytes}")
                        out.write(chunk)
            tmp.replace(dest)
            # Un corps vide est conservé comme marqueur de cache (cas "fichier
            # publié mais vide" observé sur le portail).
            return DownloadResult("empty" if written == 0 else "ok", written)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            tmp.unlink(missing_ok=True)
    return DownloadResult("error", 0, str(last_err))

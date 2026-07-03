"""download.py — cache idempotent, cap de taille, marqueur vide, atomicité."""
import io
import urllib.error

import pytest

from pipeline.collect import download


class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, content_length=None):
        super().__init__(body)
        self._content_length = content_length

    def getheader(self, name, default=None):
        if name.lower() == "content-length" and self._content_length is not None:
            return str(self._content_length)
        return default

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener(body=b"data", content_length="auto", calls=None):
    def open_url(url, timeout):
        if calls is not None:
            calls.append(url)
        length = len(body) if content_length == "auto" else content_length
        return _FakeResponse(body, content_length=length)
    return open_url


def test_download_writes_atomically(tmp_path):
    dest = tmp_path / "f.csv"
    res = download.download("https://ex.fr/f.csv", dest, open_url=_opener(b"abc"))
    assert res.status == "ok"
    assert res.size_bytes == 3
    assert dest.read_bytes() == b"abc"
    assert not list(tmp_path.glob("*.part"))


def test_download_skips_cached(tmp_path):
    dest = tmp_path / "f.csv"
    dest.write_bytes(b"cached")
    calls = []
    res = download.download("https://ex.fr/f.csv", dest, open_url=_opener(calls=calls))
    assert res.status == "cached"
    assert calls == []
    assert dest.read_bytes() == b"cached"


def test_download_force_refetches(tmp_path):
    dest = tmp_path / "f.csv"
    dest.write_bytes(b"old")
    res = download.download("https://ex.fr/f.csv", dest, open_url=_opener(b"new"), force=True)
    assert res.status == "ok"
    assert dest.read_bytes() == b"new"


def test_download_rejects_large_content_length_before_transfer(tmp_path):
    dest = tmp_path / "big.zip"
    res = download.download("https://ex.fr/big.zip", dest,
                            open_url=_opener(b"x", content_length=500 * 2**20))
    assert res.status == "too_large"
    assert not dest.exists()


def test_download_rejects_streamed_overflow_without_header(tmp_path):
    dest = tmp_path / "big.zip"
    res = download.download("https://ex.fr/big.zip", dest,
                            open_url=_opener(b"x" * 2048, content_length=None),
                            max_bytes=1024)
    assert res.status == "too_large"
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_keeps_empty_marker(tmp_path):
    dest = tmp_path / "empty.json"
    res = download.download("https://ex.fr/empty.json", dest, open_url=_opener(b""))
    assert res.status == "empty"
    assert dest.exists() and dest.stat().st_size == 0
    # Re-run : le marqueur vide évite de re-solliciter le serveur.
    calls = []
    res2 = download.download("https://ex.fr/empty.json", dest, open_url=_opener(calls=calls))
    assert res2.status == "empty"
    assert calls == []


def test_download_retries_once_on_transient_error(tmp_path):
    attempts = []

    def flaky(url, timeout):
        attempts.append(url)
        if len(attempts) == 1:
            raise urllib.error.URLError("boom")
        return _FakeResponse(b"ok", content_length=2)

    res = download.download("https://ex.fr/f.csv", tmp_path / "f.csv", open_url=flaky)
    assert res.status == "ok"
    assert len(attempts) == 2


def test_download_error_after_retry(tmp_path):
    def broken(url, timeout):
        raise urllib.error.URLError("down")

    res = download.download("https://ex.fr/f.csv", tmp_path / "f.csv", open_url=broken)
    assert res.status == "error"
    assert "down" in res.detail

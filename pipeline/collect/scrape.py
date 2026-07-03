"""Découverte des consultations et de leurs fichiers open data (stdlib uniquement).

La liste des consultations est SCRAPÉE depuis la page d'index du portail —
aucun slug en dur. Le `fetch` est injectable partout (tests sans réseau).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Iterable
from urllib.parse import urljoin, urlsplit

Fetch = Callable[[str], bytes]

# Chemin d'une page de consultation sur le portail (relatif à l'hôte de l'index).
_CONSULTATION_PATH = re.compile(r"^/autres/consultations-citoyennes/([^/#?]+)/?$")
# Les fichiers de données du portail vivent sous /static/openData/.
_DATA_PATH_MARKER = "/static/openData/"

_FORMAT_BY_SUFFIX = (
    (".json.zip", "json_zip"),
    (".xml.zip", "xml_zip"),
    (".csv", "csv"),
    (".json", "json"),
    (".zip", "zip"),
)


@dataclass(frozen=True)
class Consultation:
    slug: str
    title: str
    page_url: str


@dataclass(frozen=True)
class DataFile:
    filename: str
    url: str
    format: str
    redundant: bool = False  # jumeau XML d'un json.zip de même stem


class _LinkParser(HTMLParser):
    """Collecte (href, texte de l'ancre) — le seul besoin, sans dépendance HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            text = re.sub(r"\s+", " ", "".join(self._text)).strip()
            self.links.append((self._href, text))
            self._href = None


def _parse_links(base_url: str, html: bytes) -> list[tuple[str, str]]:
    """Rend les liens de la page, hrefs absolutisés par rapport à `base_url`."""
    parser = _LinkParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    return [(urljoin(base_url, href), text) for href, text in parser.links]


def list_consultations(index_url: str, fetch: Fetch) -> list[Consultation]:
    """Scrape l'index → une entrée par consultation (dédupliquée, 1re occurrence)."""
    host = urlsplit(index_url).netloc
    seen: dict[str, Consultation] = {}
    for url, text in _parse_links(index_url, fetch(index_url)):
        parts = urlsplit(url)
        if parts.netloc != host:
            continue
        m = _CONSULTATION_PATH.match(parts.path)
        if not m or parts.fragment:
            continue
        slug = m.group(1)
        if slug not in seen:
            seen[slug] = Consultation(slug=slug, title=text, page_url=url)
    return list(seen.values())


def _tag_format(filename: str) -> str:
    lower = filename.lower()
    for suffix, fmt in _FORMAT_BY_SUFFIX:
        if lower.endswith(suffix):
            return fmt
    return "other"


def _mark_redundant_xml(files: Iterable[DataFile]) -> list[DataFile]:
    """Un `X.xml.zip` est redondant si son jumeau `X.json.zip` existe (règle de format)."""
    files = list(files)
    json_stems = {f.filename[: -len(".json.zip")] for f in files if f.format == "json_zip"}
    return [
        DataFile(f.filename, f.url, f.format, redundant=True)
        if f.format == "xml_zip" and f.filename[: -len(".xml.zip")] in json_stems
        else f
        for f in files
    ]


def list_data_files(page_url: str, fetch: Fetch) -> list[DataFile]:
    """Scrape une page de consultation → ses fichiers de données (même hôte, dédupliqués)."""
    host = urlsplit(page_url).netloc
    seen: dict[str, DataFile] = {}
    for url, _text in _parse_links(page_url, fetch(page_url)):
        parts = urlsplit(url)
        if parts.netloc != host or _DATA_PATH_MARKER not in parts.path:
            continue
        filename = parts.path.rsplit("/", 1)[-1]
        if filename and filename not in seen:
            seen[filename] = DataFile(filename=filename, url=url, format=_tag_format(filename))
    return _mark_redundant_xml(seen.values())

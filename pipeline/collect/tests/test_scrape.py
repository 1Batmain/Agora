"""scrape.py — découverte des consultations et de leurs fichiers de données."""
from pathlib import Path

from pipeline.collect import scrape

FIXTURES = Path(__file__).parent / "fixtures"
INDEX_URL = "https://data.assemblee-nationale.fr/autres/consultations-citoyennes"


def _fetch_fixture(name):
    def fetch(url):
        return (FIXTURES / name).read_bytes()
    return fetch


def test_list_consultations_extracts_slugs_and_titles():
    consultations = scrape.list_consultations(INDEX_URL, fetch=_fetch_fixture("index.html"))
    by_slug = {c.slug: c for c in consultations}
    # 3 consultations, dédupliquées, liens de navigation/externes/ancres ignorés.
    assert set(by_slug) == {"fin-de-vie", "cannabis-recreatif", "tiktok"}
    assert by_slug["fin-de-vie"].title == "Fin de vie"
    assert by_slug["cannabis-recreatif"].title == "Cannabis récréatif"  # texte multi-lignes aplati
    assert by_slug["tiktok"].page_url == f"{INDEX_URL}/tiktok"  # href absolu accepté


def test_list_data_files_relative_hrefs():
    page_url = f"{INDEX_URL}/fin-de-vie"
    files = scrape.list_data_files(page_url, fetch=_fetch_fixture("page_relative.html"))
    by_name = {f.filename: f for f in files}
    # Liens /static/ non-données (css) et hors /static/ exclus ; doublons dédupliqués.
    assert set(by_name) == {"FDV-Article-1.json.zip", "FDV-Article-1.xml.zip",
                            "FDV-Article-2.json.zip", "notice.xml.zip",
                            "dump.zip", "export.csv"}
    assert by_name["FDV-Article-1.json.zip"].url.startswith("https://data.assemblee-nationale.fr/static/")
    assert by_name["FDV-Article-1.json.zip"].format == "json_zip"
    assert by_name["export.csv"].format == "csv"
    assert by_name["dump.zip"].format == "zip"
    # Jumeau XML d'un json.zip → redondant ; XML sans jumeau → pas redondant.
    assert by_name["FDV-Article-1.xml.zip"].redundant is True
    assert by_name["notice.xml.zip"].redundant is False


def test_list_data_files_absolute_href_same_host_only():
    files = scrape.list_data_files(f"{INDEX_URL}/tiktok", fetch=_fetch_fixture("page_absolute.html"))
    assert [f.filename for f in files] == ["tiktok_appel_a_temoignages.csv"]
    assert files[0].url == ("https://data.assemblee-nationale.fr/static/openData/repository/"
                            "CONSULTATIONS_CITOYENNES/TIKTOK/tiktok_appel_a_temoignages.csv")


def test_list_data_files_none():
    assert scrape.list_data_files(f"{INDEX_URL}/vide", fetch=_fetch_fixture("page_nolinks.html")) == []

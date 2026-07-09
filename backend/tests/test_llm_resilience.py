"""Un 429 transitoire ne doit plus coûter un titre ni une synthèse.

Vécu deux fois : le « bug des 257 titres-labels », puis un rebuild tiktok servi avec 27 %
de titres en mots-clés et 37 % de synthèses absentes — parce que le quota Mistral avait
été atteint en cours de build et qu'aucun appelant ne réessayait. Le pipeline écrivait
`status: ready`.

Deux lignes de défense, testées ici :
  1. `mistral_client.chat` RÉESSAIE les erreurs transitoires (429, 5xx, réseau) — et ne
     réessaie jamais ce qui ne peut pas guérir (401, pas de clé).
  2. `build_analysis` REFUSE de servir un build dont l'enrichissement s'est effondré.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.build_analysis import DegradedEnrichmentError, _assert_enrichment_is_complete
from pipeline.cluster import mistral_client as MC


class _Resp:
    def __init__(self, status: int, payload: dict | None = None, retry_after=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = {"Retry-After": retry_after} if retry_after else {}

    def json(self):
        return self._payload


_OK = {"choices": [{"message": {"content": "un titre"}}], "usage": {}}


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    # `chat` importe `time` localement : on patche le module, pas un attribut de MC.
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(MC, "load_api_key", lambda: "clé-de-test")
    MC.reset_usage()


def _patch_post(monkeypatch, responses):
    """`responses` : liste de _Resp ou d'exceptions, consommée à chaque appel."""
    calls = {"n": 0}
    it = iter(responses)

    def fake_post(*_a, **_k):
        calls["n"] += 1
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def test_429_est_reessaye_puis_reussit(monkeypatch):
    calls = _patch_post(monkeypatch, [_Resp(429), _Resp(429), _Resp(200, _OK)])
    assert MC.chat([{"role": "user", "content": "x"}], model="m") == "un titre"
    assert calls["n"] == 3
    assert MC.get_exhausted()["count"] == 0        # rien de perdu


def test_5xx_et_reseau_sont_reessayes(monkeypatch):
    import httpx
    calls = _patch_post(monkeypatch, [_Resp(503), httpx.TimeoutException("t"), _Resp(200, _OK)])
    assert MC.chat([{"role": "user", "content": "x"}], model="m") == "un titre"
    assert calls["n"] == 3


def test_401_nest_jamais_reessaye(monkeypatch):
    """Une clé invalide ne guérit pas : réessayer ne fait que brûler du temps."""
    calls = _patch_post(monkeypatch, [_Resp(401, {"message": "unauthorized"})])
    with pytest.raises(MC.MistralError) as exc:
        MC.chat([{"role": "user", "content": "x"}], model="m")
    assert exc.value.status == 401
    assert calls["n"] == 1


def test_absence_de_cle_ne_reessaye_pas(monkeypatch):
    monkeypatch.setattr(MC, "load_api_key", lambda: None)
    with pytest.raises(MC.MistralError) as exc:
        MC.chat([{"role": "user", "content": "x"}], model="m")
    assert exc.value.status == 0 and exc.value.reason == "no_api_key"


def test_429_persistant_compte_lappel_perdu(monkeypatch):
    monkeypatch.setattr(MC, "MAX_RETRIES", 2)
    calls = _patch_post(monkeypatch, [_Resp(429)] * 3)
    with pytest.raises(MC.MistralError):
        MC.chat([{"role": "user", "content": "x"}], model="m")
    assert calls["n"] == 3                          # 1 essai + 2 réessais
    ex = MC.get_exhausted()
    assert ex["count"] == 1 and ex["by_status"]["429"] == 1


# --------------------------------------------------------------------------- #
# Garde-fou de build : un enrichissement effondré n'est pas `ready`
# --------------------------------------------------------------------------- #
def _tree(titles: dict[str, str]):
    return SimpleNamespace(nodes={i: SimpleNamespace(id=i, title=t) for i, t in titles.items()})


def test_build_refuse_un_enrichissement_degrade(monkeypatch):
    import backend.build_analysis as B
    monkeypatch.setattr(B.store, "read_insights", lambda *_a: None)      # 100 % manquants
    tree = _tree({f"n{i}": ("a · b · c" if i < 3 else "Un vrai titre") for i in range(10)})
    with pytest.raises(DegradedEnrichmentError) as exc:
        _assert_enrichment_is_complete("ds", tree, list(tree.nodes))
    msg = str(exc.value)
    assert "titres en repli" in msg and "synthèses absentes" in msg
    assert "AGORA_ALLOW_DEGRADED" in msg


def test_build_accepte_un_repli_marginal(monkeypatch):
    import backend.build_analysis as B
    monkeypatch.setattr(B.store, "read_insights", lambda *_a: {"markdown": "ok"})
    # 1 titre de repli sur 100 = 1 % < seuil 5 % : légitime, on ne lève pas.
    titles = {f"n{i}": ("a · b · c" if i == 0 else "Un vrai titre") for i in range(100)}
    _assert_enrichment_is_complete("ds", _tree(titles), list(titles))


def test_flag_allow_degraded_laisse_passer(monkeypatch):
    import backend.build_analysis as B
    monkeypatch.setattr(B.store, "read_insights", lambda *_a: None)
    monkeypatch.setattr(B, "ALLOW_DEGRADED", True)
    tree = _tree({f"n{i}": "a · b · c" for i in range(10)})
    B._assert_enrichment_is_complete("ds", tree, list(tree.nodes))        # ne lève pas

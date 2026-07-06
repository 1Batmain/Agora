"""Auto-test des 3 chemins d'extraction des claims (api / mac / auto→repli).

    uv run --extra contender python -m pipeline.claims.selftest_backends

Vérifie, sur quelques avis réels :
  1. DÉFAUT (rien d'exporté)            → backend `api`, claims extraits, format list[str] ;
  2. AGORA_CLAIMS_BACKEND=mac           → backend `mac` (SKIP honnête si le Mac est down) ;
  3. =auto avec Mac simulé injoignable  → bascule `api`, claims extraits.
Puis : les claims des 3 chemins ont le MÊME format (dict[str, list[str]] non vide).

N'imprime JAMAIS de secret (ni la clé Mistral, ni l'URL du Mac). Sortie non nulle si
un chemin requis échoue ; le chemin `mac` est OPTIONNEL (dépend de la dispo du Mac).
"""

from __future__ import annotations

import os
import sys

from pipeline.claims.backend import MacBackend, resolve_backend
from pipeline.claims.extract import extract_claims
from pipeline.claims.ollama import OllamaStats
from pipeline.claims.pipeline import Avis

# Quelques avis courts, multilingues, pour une extraction rapide et déterministe.
AVIS = [
    Avis(id="a1", text="Il faut plus de pistes cyclables et réduire la place de la voiture en ville."),
    Avis(id="a2", text="Le coût de la vie augmente trop vite, le pouvoir d'achat des familles baisse."),
    Avis(id="a3", text="Wir brauchen mehr Investitionen in erneuerbare Energien und weniger Bürokratie."),
    Avis(id="a4", text="Public transport should be free for students and run more often at night."),
]


def _check_format(claims: dict) -> None:
    """Lève AssertionError si la sortie n'a pas le format attendu."""
    assert isinstance(claims, dict), f"claims n'est pas un dict: {type(claims)}"
    assert claims, "claims vide"
    for aid, lst in claims.items():
        assert isinstance(lst, list) and lst, f"{aid}: liste vide/invalide"
        assert all(isinstance(x, str) and x.strip() for x in lst), f"{aid}: claim non-str"


def _run_path(label: str, *, backend: str | None, ollama_url: str | None = None
              ) -> tuple[bool, str, dict | None]:
    """Résout le backend, extrait, vérifie le format. → (ok, backend_name, claims)."""
    try:
        be = resolve_backend(backend, ollama_url=ollama_url)
        stats = OllamaStats()
        claims = extract_claims(AVIS, backend=be, stats=stats)
        _check_format(claims)
        n_claims = sum(len(v) for v in claims.values())
        print(f"  ✅ {label}: backend={be.name} model={be.model} "
              f"sovereign={be.sovereign} → {n_claims} claims / {len(claims)} avis "
              f"(calls={stats.calls}, errors={stats.errors})")
        return True, be.name, claims
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ {label}: {type(exc).__name__}: {exc}")
        return False, "", None


def main() -> int:
    # On part d'un environnement propre pour le défaut (n'altère pas la clé Mistral).
    for var in ("AGORA_CLAIMS_BACKEND", "AGORA_OLLAMA_URL"):
        os.environ.pop(var, None)

    failures: list[str] = []
    formats: list[dict] = []

    # 1) DÉFAUT → api
    print("[1] Défaut (rien d'exporté) → attendu: api")
    ok, name, claims = _run_path("défaut", backend=None)
    if not ok or name != "api":
        failures.append("défaut→api")
    elif claims:
        formats.append(claims)

    # 2) mac (opt-in) — OPTIONNEL : skip honnête si le Mac est injoignable.
    print("[2] AGORA_CLAIMS_BACKEND=mac → attendu: mac (ou SKIP si Mac down)")
    mac_url = os.environ.get("AGORA_OLLAMA_URL") or _read_mac_url()
    if mac_url and MacBackend(mac_url).probe():
        ok, name, claims = _run_path("mac", backend="mac", ollama_url=mac_url)
        if not ok or name != "mac":
            failures.append("mac")
        elif claims:
            formats.append(claims)
    else:
        print("  ⏭️  SKIP: Mac injoignable (attendu — le poste local est souvent éteint).")

    # 3) auto avec Mac simulé down → repli api
    print("[3] =auto, Mac injoignable (127.0.0.1:1) → attendu: repli api")
    ok, name, claims = _run_path("auto→repli", backend="auto", ollama_url="http://127.0.0.1:1")
    if not ok or name != "api":
        failures.append("auto→api")
    elif claims:
        formats.append(claims)

    # 4) Même format sur tous les chemins exécutés.
    print("[4] Cohérence de format entre chemins")
    if len(formats) >= 2:
        keys = {frozenset(f.keys()) for f in formats}
        if len(keys) == 1:
            print(f"  ✅ tous les chemins couvrent les mêmes {len(AVIS)} avis, format list[str].")
        else:
            print("  ❌ ensembles d'avis incohérents entre chemins.")
            failures.append("format")
    else:
        print("  ⏭️  un seul chemin a produit des claims — comparaison non applicable.")

    print()
    if failures:
        print(f"ÉCHEC: {', '.join(failures)}")
        return 1
    print("OK: tous les chemins requis passent (api défaut + auto repli; mac si dispo).")
    return 0


def _read_mac_url() -> str | None:
    """Lit l'URL du Mac depuis var/MAC_LOCAL_OLLAMA sans jamais l'imprimer."""
    from pathlib import Path

    f = Path(__file__).resolve().parents[2] / "var" / "MAC_LOCAL_OLLAMA"
    try:
        url = f.read_text(encoding="utf-8").strip()
        return url or None
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())

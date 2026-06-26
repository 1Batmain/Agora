"""Helper PARTAGÉ de l'« enrichissement LLM caché par contenu » — flux commun unique.

`backend.titles`, `backend.cluster_enrich` (hook/description) et `backend.insights`
répétaient LE MÊME squelette : clé de contenu → cache MÉMOIRE → cache DISQUE →
`mistral_client.available()` → `chat()` → `except MistralError` → repli → écriture
mem+disque. Ce module factorise CE flux dans `cached_llm(...)`.

⚠️ **RÉTRO-COMPAT DURE** : le helper ne décide NI la clé, NI le chemin disque, NI le
schéma JSON écrit — chaque appelant les fournit (sa `key`, son `disk_path`, ses
`decode`/`encode`). Les caches DÉJÀ présents sur disque continuent donc de résoudre à
l'identique : on déduplique le FLUX, pas le schéma de clé.

`cached_llm` ne lève JAMAIS sur une erreur LLM ou disque : repli gracieux (jamais un
crash, jamais une régénération silencieuse qui casserait un cache existant). Il renvoie
`(valeur, source)` où `source` indique d'où vient la valeur — utile à un appelant qui
re-tamponne sa sortie (cf. `insights`), ignoré par les appelants « valeur simple ».
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pipeline.cluster import mistral_client

# Sources possibles de la valeur renvoyée.
MEMORY = "memory"        # servie du cache mémoire (session courante)
DISK = "disk"            # relue du cache disque (persistant)
GENERATED = "generated"  # fraîchement produite par le LLM
FALLBACK = "fallback"    # repli (pas de clé / erreur API / sortie inexploitable)


def _identity(v: Any) -> Any:
    return v


def _write(mem_cache: dict, key: Any, disk_path: Path, value: Any,
           encode: Callable[[Any], Any]) -> Any:
    """Persiste `value` en mémoire + disque (le disque ne lève jamais). Renvoie `value`."""
    mem_cache[key] = value
    try:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_text(json.dumps(encode(value), ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return value


def cached_llm(
    *,
    mem_cache: dict,
    key: Any,
    disk_path: Path,
    build_messages: Callable[[], list[dict]],
    fallback_fn: Callable[..., Any],
    model: str,
    max_tokens: int,
    temperature: float,
    decode: Callable[[Any], Any] = _identity,
    encode: Callable[[Any], Any] = _identity,
    postprocess: Callable[[str], Any] = _identity,
    accept: Callable[[Any], bool] = bool,
    cache_fallback: bool = True,
    refresh: bool = False,
) -> tuple[Any, str]:
    """Exécute le flux LLM-caché commun et renvoie `(valeur, source)`. Ne lève jamais (LLM/disque).

    Flux : cache MÉMOIRE → cache DISQUE → (sur miss) `mistral_client.available()` →
    `chat(build_messages())` → `postprocess` → repli `fallback_fn` → écriture mem+disque.

    - `key` / `disk_path` : la clé MÉMOIRE et le chemin DISQUE PROPRES à l'appelant
      (calculés par SON schéma — c'est ce qui garantit la rétro-compat dure).
    - `decode(parsed_json) -> valeur | None` : extrait la valeur d'un JSON disque déjà
      parsé (ex. `data["title"]`). `None` ⇒ entrée disque ignorée (miss).
    - `encode(valeur) -> json` : sérialise la valeur à écrire (schéma de l'appelant).
    - `postprocess(str) -> valeur` : transforme la sortie brute du LLM.
    - `accept(valeur) -> bool` : la valeur est-elle exploitable (sinon repli) ?
    - `fallback_fn(reason, exc=None)` : repli ; `reason ∈ {no_api_key, api_error, rejected}`.
      `exc` est la `MistralError` pour `api_error`, sinon `None`.
    - `cache_fallback` : écrit-on AUSSI le repli (mem+disque) ? (`insights` : non — il
      veut réessayer dès que la clé revient.)

    `source ∈ {memory, disk, generated, fallback}`.
    """
    if not refresh:
        # 1) Cache mémoire (évite de relire le disque dans une même session).
        if key in mem_cache:
            v = mem_cache[key]
            if accept(v):
                return v, MEMORY
        # 2) Cache disque (persistant entre redémarrages).
        if disk_path.exists():
            try:
                data = json.loads(disk_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = None
            if data is not None:
                v = decode(data)
                if v is not None and accept(v):
                    mem_cache[key] = v
                    return v, DISK

    # 3) Miss → génération (repli gracieux à chaque étape).
    if not mistral_client.available():
        value = fallback_fn("no_api_key")
        if cache_fallback:
            _write(mem_cache, key, disk_path, value, encode)
        return value, FALLBACK

    try:
        content = mistral_client.chat(
            build_messages(), model=model,
            temperature=temperature, max_tokens=max_tokens,
        )
    except mistral_client.MistralError as exc:
        value = fallback_fn("api_error", exc)
        if cache_fallback:
            _write(mem_cache, key, disk_path, value, encode)
        return value, FALLBACK

    candidate = postprocess(content)
    if accept(candidate):
        return _write(mem_cache, key, disk_path, candidate, encode), GENERATED

    # Le LLM a répondu mais la sortie est inexploitable → repli.
    value = fallback_fn("rejected")
    if cache_fallback:
        _write(mem_cache, key, disk_path, value, encode)
    return value, FALLBACK

"""Phase de BUILD — traduit les avis non-français en FR, cachée et idempotente.

Généricité linguistique (cf. mémoire projet) : une consultation peut arriver dans
n'importe quelle langue (x-stance = DE/FR/IT). Le front affiche les avis **en français**
par défaut ; cette phase précalcule ces traductions au build, comme l'enrichissement.

Principe :
  - **Langue par avis** : on réutilise la langue déjà détectée à l'ingestion
    (`Idea.lang`) ; à défaut (absente / `und`), repli sur `pipeline.ingest.lang.detect_lang`.
  - **Traduction** (si `lang != fr`) : LLM CHEAP batché (`mistral-small-latest`), via
    `pipeline.translate.translate_batch`. Les avis déjà français ne sont PAS traduits
    (`text_fr = None`).
  - **Cache idempotent** : `backend/cache/<dataset>/translations.json`, aligné aux avis
    (clé = `avis_id`), validé par un hash du texte + le modèle. Un rebuild ne re-traduit
    QUE les avis nouveaux ou modifiés. Le fichier vit à la racine du dataset (hors
    `analysis/`), donc un `store.clear()` ne le détruit pas → réutilisé entre builds.

`text_fr` reste `None` si l'avis est déjà français OU si la traduction a échoué (repli
gracieux : le front montre alors l'original). Les claims/cibles restent en offsets sur
l'ORIGINAL (gate verbatim) — cette phase ne les touche pas.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable

from backend.recluster import dataset_dir
from pipeline.ingest.lang import detect_lang
from pipeline.translate import DEFAULT_TRANSLATE_MODEL, FR, is_french, translate_batch

TRANSLATIONS_NAME = "translations.json"
# Modèle de traduction (CHEAP), surchargeable par env — aucune valeur de corpus en dur.
TRANSLATE_MODEL = os.environ.get("AGORA_TRANSLATE_MODEL", DEFAULT_TRANSLATE_MODEL)

ProgressFn = Callable[[int, int], None]


def translations_path(dataset: str):
    return dataset_dir(dataset) / TRANSLATIONS_NAME


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _resolve_lang(avis_lang: str | None, text: str) -> str:
    """Langue effective d'un avis : `Idea.lang` si exploitable, sinon détection."""
    lang = (avis_lang or "").strip().lower()
    if lang and lang != "und":
        return lang
    return detect_lang(text, default="und")


def build_translations(
    dataset: str,
    avis: list,
    lang_of: dict[str, str] | None = None,
    *,
    model: str | None = None,
    on_progress: ProgressFn | None = None,
    refresh: bool = False,
) -> dict[str, dict]:
    """Traduit les avis non-FR en français (cachée, idempotente) → `{avis_id: {lang, text_fr}}`.

    `avis` : objets portant `.id` et `.text` (les `Avis` de `prepared.avis`). `lang_of` :
    map `avis_id -> code langue` connue (ex. `Idea.lang`) ; à défaut, langue détectée.
    Ne traduit QUE les avis non-français non encore cachés (ou dont le texte a changé).
    `refresh=True` ignore le cache disque (re-traduit tout le non-FR). Ne lève jamais
    sur une erreur LLM : repli gracieux (`text_fr = None`).
    """
    from backend.analysis_store import _read_json, write_json  # I/O atomique partagée

    use_model = model or TRANSLATE_MODEL
    lang_of = lang_of or {}
    path = translations_path(dataset)
    cached = (_read_json(path) or {}) if not refresh else {}
    if not isinstance(cached, dict):
        cached = {}

    # 1) Langue par avis + tri : déjà-FR (rien) / cache valide (réutilisé) / à traduire.
    resolved: dict[str, dict] = {}      # avis_id -> {lang, text, hash}
    to_translate: list[tuple[str, str]] = []  # (avis_id, text) des avis à (re)traduire
    for a in avis:
        aid = str(a.id)
        text = a.text or ""
        lang = _resolve_lang(lang_of.get(aid), text)
        h = _text_hash(text)
        resolved[aid] = {"lang": lang, "text": text, "hash": h}
        if is_french(lang) or not text.strip():
            continue  # déjà français (ou vide) → pas de traduction
        prev = cached.get(aid)
        if (isinstance(prev, dict) and prev.get("hash") == h
                and prev.get("model") == use_model and prev.get("text_fr")):
            continue  # cache HIT (texte+modèle inchangés) → on garde la traduction
        to_translate.append((aid, text))

    # 2) Traduction batchée des manquants (un seul appel LLM par lot).
    total = len(to_translate)
    fresh: dict[str, str | None] = {}
    if total:
        texts = [t for _, t in to_translate]
        results = translate_batch(texts, model=use_model)
        for (aid, _), tr in zip(to_translate, results):
            fresh[aid] = tr
        if on_progress:
            on_progress(total, total)

    # 3) Fusion + persistance du cache (aligné aux avis présents).
    out: dict[str, dict] = {}        # ce qu'on RENVOIE : {avis_id: {lang, text_fr}}
    new_cache: dict[str, dict] = {}  # ce qu'on PERSISTE : + hash + model (validation)
    for aid, info in resolved.items():
        lang, h = info["lang"], info["hash"]
        if is_french(lang) or not info["text"].strip():
            text_fr = None
        elif aid in fresh:
            text_fr = fresh[aid]              # traduction fraîche (peut être None si échec)
        else:
            prev = cached.get(aid)            # cache HIT validé en (1)
            text_fr = prev.get("text_fr") if isinstance(prev, dict) else None
        out[aid] = {"lang": lang, "text_fr": text_fr}
        entry = {"lang": lang, "text_fr": text_fr, "hash": h, "model": use_model}
        new_cache[aid] = entry

    write_json(path, new_cache)
    return out


def _main() -> None:
    """CLI : génère/rafraîchit `translations.json` d'un dataset, SANS rebuild complet.

    Traduit exactement les avis que le build servira (mêmes id+texte via `as_avis`), donc
    le cache est aligné : la phase `translate` du build le réutilisera tel quel (0 appel).

        uv run python -m backend.translate --dataset xstance
        uv run python -m backend.translate --dataset xstance --refresh
    """
    import argparse

    from backend.build_analysis import load_dataset
    from pipeline.claims.pipeline import as_avis

    ap = argparse.ArgumentParser(description="Traduction FR cachée des avis d'un dataset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=None, help=f"modèle de trad (défaut {TRANSLATE_MODEL}, cheap)")
    ap.add_argument("--refresh", action="store_true", help="ignore le cache (re-traduit tout le non-FR)")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    lang_of = {str(getattr(it, "id", "")): getattr(it, "lang", None) for it in ds.ideas}
    avis = as_avis(ds.ideas)
    print(f"[translate] {args.dataset} · {len(avis)} avis · modèle {args.model or TRANSLATE_MODEL}", flush=True)
    out = build_translations(
        args.dataset, avis, lang_of, model=args.model, refresh=args.refresh,
        on_progress=lambda d, t: print(f"[translate] traduits {d}/{t}", flush=True),
    )
    n_fr = sum(1 for v in out.values() if is_french(v["lang"]))
    n_tr = sum(1 for v in out.values() if v["text_fr"])
    n_fail = sum(1 for v in out.values() if not is_french(v["lang"]) and not v["text_fr"])
    print(f"[translate] ✓ {args.dataset} · FR (non traduits) {n_fr} · traduits {n_tr} · échecs {n_fail}")


if __name__ == "__main__":
    _main()

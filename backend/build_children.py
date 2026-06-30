"""BUILD — découpe une consultation MÈRE en consultations ENFANTS (1 par groupe).

GÉNÉRIQUE : on splite un dataset parent par la valeur d'un CHAMP de `props`
(`--by topic`, `--by question_id`, …). Pour chaque valeur distincte, on fabrique
un dataset ENFANT `id = <parent>__<slug>` avec un cache PROPRE
(`backend/cache/<child>/`) contenant le SOUS-ENSEMBLE des données du parent :

  - `ideas.jsonl`     : les lignes brutes du parent dont `props[<champ>]==valeur` ;
  - `embeddings.npy`  : les vecteurs d'avis TRANCHÉS (mêmes positions que ces lignes) ;
  - `claims.json`     : les claims du parent pour ces avis (RÉUTILISE l'extraction —
                        ne RÉ-EXTRAIT JAMAIS) ;
  - `claims_emb.npz`  : les embeddings de claims TRANCHÉS (empreinte recalculée) ;
  - `target_emb.npz`  : les embeddings de cibles TRANCHÉS (knob α) ;
  - `meta.json`       : `label`=nom du groupe, `parent_id`=<parent>, `status`=closed.

On lance ensuite l'ANALYSE de chaque enfant via le pipeline existant
(`build_analysis`, SANS --force) : comme claims/embeddings sont déjà cachés et
TRANCHÉS, l'enfant ne ré-extrait ni ne ré-embed — seuls le clustering + le naming
+ l'enrichissement (titres/accroches/insights, modèle CHEAP, cachés) tournent.

Enfin on écrit `children=[ids]` dans le `meta.json` du PARENT. Le parent reste un
CONTENEUR (pas d'analyse propre requise). Idempotent : un re-run réutilise les
caches (extraction du parent, analyses des enfants).

Usage CLI :
    MISTRAL_API_KEY=$(cat var/mistral.key) \\
    uv run --extra embed-contender python -m backend.build_children \\
        --parent xstance --by topic
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

from backend import build_analysis as ba
from backend.claims_endpoint import (
    CLAIMS_EMB_NAME,
    CLAIMS_NAME,
    TARGET_EMB_NAME,
    _emb_fingerprint,
    _save_claims_cache,
    _save_emb_cache,
    _save_target_cache,
    prepare_claims,
)
from backend.recluster import (
    EMB_NAME,
    IDEAS_NAME,
    META_NAME,
    cache_paths,
    dataset_dir,
    load_cache,
)


def _log(msg: str) -> None:
    print(f"[build_children] {msg}", flush=True)


def slugify(value: str) -> str:
    """Slug ASCII minuscule, générique : non-alphanum → '-', dédoublé, rogné.

    `"Foreign Policy"` → `"foreign-policy"`. Préserve les chiffres (`question_id`).
    Repli `"x"` si la valeur ne contient aucun caractère alphanumérique.
    """
    s = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return s or "x"


def _read_raw_rows(parent: str) -> list[dict]:
    """Lit les lignes BRUTES de `ideas.jsonl` du parent, dans l'ordre (= ordre des vecteurs)."""
    _, ideas_path, _ = cache_paths(parent)
    rows: list[dict] = []
    with open(ideas_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_value(row: dict, field: str):
    """Valeur du champ de split : `props[field]` prioritaire, repli top-level (cf. Idea.from_row)."""
    props = row.get("props") or {}
    if field in props:
        return props[field]
    return row.get(field)


def _read_meta(parent: str) -> dict:
    _, _, meta_path = cache_paths(parent)
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return {}


def _write_meta(dataset: str, meta: dict) -> None:
    _, _, meta_path = cache_paths(dataset)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _child_meta(child_id: str, label, parent: str, parent_meta: dict,
                rows: list[dict]) -> dict:
    """Meta d'un enfant : label=groupe, parent_id, status=closed, langues DÉRIVÉES des lignes."""
    langs = Counter(
        (r.get("props") or {}).get("lang") or r.get("lang") for r in rows
    )
    langs.pop(None, None)
    n = len(rows)
    return {
        "id": child_id,
        "label": str(label),
        "parent_id": parent,
        "status": "closed",
        "n_nodes": n,
        "n_loaded": n,
        "languages": [lg for lg, _ in langs.most_common()],
        "lang_counts": dict(langs.most_common()),
        "source": parent_meta.get("source", parent),
        "model_id": parent_meta.get("model_id"),
        "dim": parent_meta.get("dim"),
        "split": {"by_parent": parent, "field": None, "value": str(label)},
    }


def build_children(parent: str, by: str, *, backend: str | None = None,
                   model: str | None = None) -> list[str]:
    """Construit tous les enfants du parent splité par `props[by]`. Renvoie leurs ids."""
    rows = _read_raw_rows(parent)
    emb_path, _, _ = cache_paths(parent)
    parent_emb = np.load(emb_path).astype(np.float32)
    if len(rows) != parent_emb.shape[0]:
        raise RuntimeError(
            f"Parent désaligné ({parent}) : {len(rows)} ideas vs {parent_emb.shape[0]} vecteurs."
        )
    parent_meta = _read_meta(parent)

    # 1) Extraction + embeddings du PARENT (cachés). C'est l'UNIQUE passe LLM/torch :
    #    les enfants RÉUTILISENT ces claims/vecteurs (tranchés), sans rien ré-extraire.
    _log(f"{parent} · préparation des claims (extraction + embed, cachés)")
    parent_ds = SimpleNamespace(id=parent, ideas=load_cache(parent)[0])
    prep = prepare_claims(parent_ds, backend=backend, model=model)
    _log(f"{parent} · {len(prep.claim_texts)} claims sur {len(prep.avis)} avis "
         f"(ré-extraits: {prep.extracted}, embeddings recalculés: {prep.embedded})")

    # Index : avis_id -> indices de ses claims (dans l'ordre d'aplatissement du parent).
    # `claim_owner[r]` = index d'avis ; `prep.avis[ai].id` = id d'avis = id de ligne.
    rows_by_avis: dict[str, list[int]] = defaultdict(list)
    for r, ai in enumerate(prep.claim_owner):
        rows_by_avis[prep.avis[ai].id].append(r)

    # 2) Groupes : valeur du champ -> positions de lignes (ordre fichier = ordre vecteurs).
    groups: dict[object, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        val = _group_value(row, by)
        if val is None or str(val).strip() == "":
            continue  # avis sans la propriété de split → non rattaché (générique)
        groups[val].append(i)

    if not groups:
        raise RuntimeError(
            f"Aucune valeur pour le champ props {by!r} dans {parent} — rien à splitter."
        )
    _log(f"{parent} · {len(groups)} valeurs pour props.{by} : "
         + ", ".join(f"{v}({len(ix)})" for v, ix in sorted(groups.items(), key=lambda kv: str(kv[0]))))

    child_ids: list[str] = []
    for value, positions in sorted(groups.items(), key=lambda kv: str(kv[0])):
        slug = slugify(value)
        child_id = f"{parent}__{slug}"
        child_dir = dataset_dir(child_id)
        child_dir.mkdir(parents=True, exist_ok=True)
        child_rows = [rows[i] for i in positions]

        # 2a) ideas.jsonl + embeddings.npy TRANCHÉS (alignés aux mêmes positions).
        with open(child_dir / IDEAS_NAME, "w", encoding="utf-8") as fh:
            for row in child_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.save(child_dir / EMB_NAME, parent_emb[positions])

        # 2b) claims.json TRANCHÉ : RÉUTILISE l'extraction du parent (même `model`),
        #     pour tous les avis du groupe présents dans le cache d'extraction.
        child_claim_rows: list[int] = []
        child_claims: dict[str, list] = {}
        for row in child_rows:
            aid = str(row.get("id"))
            if aid in prep.claims_by_id:
                child_claims[aid] = prep.claims_by_id[aid]
                child_claim_rows.extend(rows_by_avis.get(aid, []))
        _save_claims_cache(child_dir / CLAIMS_NAME, prep.model, child_claims)

        # 2c) claims_emb.npz + target_emb.npz TRANCHÉS : on slice les vecteurs du parent
        #     aux lignes de claims du groupe, et on RECALCULE l'empreinte sur les textes
        #     de l'enfant (ordre identique à `_flatten` de l'enfant) → cache HIT au build.
        idx = np.asarray(child_claim_rows, dtype=np.intp)
        child_claim_texts = [prep.claim_texts[r] for r in child_claim_rows]
        claim_fp = _emb_fingerprint(prep.embedder, child_claim_texts)
        child_claim_vecs = (prep.claim_vecs[idx] if idx.size
                            else np.zeros((0, prep.claim_vecs.shape[1]), dtype=prep.claim_vecs.dtype))
        _save_emb_cache(child_dir / CLAIMS_EMB_NAME, claim_fp, child_claim_vecs)

        child_target_strings = []
        for r in child_claim_rows:
            tgt = prep.claim_target[r]
            if tgt is not None:
                s, e = tgt
                t = prep.avis[prep.claim_owner[r]].text[s:e].strip()
                child_target_strings.append(t if t else "")
            else:
                child_target_strings.append("")
        target_fp = _emb_fingerprint(prep.embedder, child_target_strings)
        child_target_vecs = (prep.target_vecs[idx] if idx.size
                             else np.zeros((0, prep.claim_vecs.shape[1]), dtype=prep.target_vecs.dtype))
        child_target_mask = (prep.target_mask[idx] if idx.size
                             else np.zeros((0,), dtype=bool))
        _save_target_cache(child_dir / TARGET_EMB_NAME, target_fp,
                           child_target_vecs, child_target_mask)

        # 2d) meta enfant.
        meta = _child_meta(child_id, value, parent, parent_meta, child_rows)
        meta["split"]["field"] = by
        _write_meta(child_id, meta)

        _log(f"{child_id} · {len(child_rows)} avis · {len(child_claim_rows)} claims tranchés "
             f"· analyse…")

        # 2e) ANALYSE de l'enfant (clustering + naming + enrich + opinion), SANS --force.
        #     claims/embeddings cachés & tranchés ⇒ zéro ré-extraction, zéro ré-embed.
        child_ds = ba.load_dataset(child_id)
        ba.build_analysis(child_ds, backend=backend, model=model)
        child_ids.append(child_id)

    # 3) `children` dans le meta du PARENT (le parent reste un conteneur).
    parent_meta = _read_meta(parent)
    parent_meta["children"] = child_ids
    _write_meta(parent, parent_meta)
    _log(f"{parent} · children={child_ids}")
    return child_ids


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Découpe une consultation mère en enfants (1 par valeur d'un champ props)."
    )
    ap.add_argument("--parent", required=True, help="id du dataset parent (sous backend/cache/)")
    ap.add_argument("--by", required=True, help="champ de props sur lequel splitter (ex. topic)")
    ap.add_argument("--backend", default=None, help="api (défaut) | mac | auto")
    ap.add_argument("--model", default=None, help="modèle d'extraction (défaut: build_analysis)")
    args = ap.parse_args()

    ids = build_children(args.parent, args.by, backend=args.backend, model=args.model)
    _log(f"✓ {len(ids)} enfants construits : {ids}")


if __name__ == "__main__":
    main()

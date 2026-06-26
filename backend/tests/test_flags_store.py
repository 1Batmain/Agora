"""FLAGS généralisés — feedback sur un AVIS *ou* une SYNTHÈSE de thème.

Verrouille le contrat sans réseau ni LLM (cache redirigé vers un tmp dir) :

  1. **Avis (rétro-compat)** : upsert par avis → relu, `avis_id` présent, type "avis".
  2. **Thème** : upsert avec `layer`+`category` → relu fidèlement, clé distincte
     d'un avis de MÊME id (pas de collision).
  3. **Coexistence** : `list_flags` rend les DEUX types.
  4. **Migration douce** : un `flags.json` à l'ANCIEN format `{avis_id: {avis_id,text}}`
     est relu en modèle complet (`target_type="avis"`, `avis_id` conservé).
  5. **Delete** ciblé : retirer le thème ne touche pas l'avis homonyme.

Lancer :  uv run python -m backend.test_flags_store
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from backend import flags_store as fs
from backend.analysis_store import write_json


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Redirige le cache vers un tmp dir → zéro disque réel, déterministe.
        fs.dataset_dir = lambda dataset, _root=Path(tmp): _root / dataset  # type: ignore
        (Path(tmp) / "ds").mkdir(parents=True)
        DS = "ds"

        # 1. Avis (rétro-compat) ------------------------------------------------
        a = fs.upsert_flag(DS, "avis", "42", "mauvaise découpe")
        _ok(a["target_type"] == "avis" and a["target_id"] == "42", "avis: type+id")
        _ok(a["avis_id"] == "42", "avis: avis_id conservé (front existant)")
        got = fs.get_flag(DS, "avis", "42")
        _ok(got and got["text"] == "mauvaise découpe", "avis: relu après upsert")

        # 2. Thème avec layer+category -----------------------------------------
        t = fs.upsert_flag(
            DS, "theme", "42", "résumé inventé", layer=2, category="Hallucination"
        )
        _ok(t["layer"] == 2 and t["category"] == "Hallucination", "thème: layer+category")
        gt = fs.get_flag(DS, "theme", "42")
        _ok(gt and gt["text"] == "résumé inventé", "thème: relu après upsert")
        _ok(fs.get_flag(DS, "avis", "42")["text"] == "mauvaise découpe",
            "pas de collision avis/thème de même id")

        # 3. Coexistence des deux types ----------------------------------------
        kinds = {f["target_type"] for f in fs.list_flags(DS)}
        _ok(kinds == {"avis", "theme"}, "list_flags rend les deux types")

        # 4. Migration douce de l'ancien format --------------------------------
        legacy = {"7": {"avis_id": "7", "text": "vieux flag",
                        "created_at": "2020", "updated_at": "2020"}}
        write_json(fs.flags_path("ds"), legacy)
        m = fs.get_flag(DS, "avis", "7")
        _ok(m and m["target_type"] == "avis" and m["avis_id"] == "7",
            "migration: ancien flag avis normalisé")
        _ok(m["text"] == "vieux flag", "migration: texte préservé")

        # 5. Delete ciblé (re-seed les deux types après l'écrasement legacy) ----
        fs.upsert_flag(DS, "avis", "99", "x")
        fs.upsert_flag(DS, "theme", "99", "y", layer=1, category="Mauvais résumé")
        _ok(fs.delete_flag(DS, "theme", "99") is True, "delete thème: True")
        _ok(fs.get_flag(DS, "theme", "99") is None, "delete thème: parti")
        _ok(fs.get_flag(DS, "avis", "99") is not None, "delete thème: avis homonyme intact")
        _ok(fs.delete_flag(DS, "theme", "99") is False, "delete absent: False")

    print("\nflags_store : OK ✅")


if __name__ == "__main__":
    run()

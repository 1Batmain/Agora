"""Descripteur de source DÉCLARATIF + lecteur générique unique.

Le cœur de la généricité de l'ingestion (résout l'audit #1). Au lieu d'un
`read_xstance()` / `read_tiktok()` codé en dur par corpus, **une consultation =
un descripteur** (fichier JSON, data/config) consommé par un seul
`read_generic(descriptor)`. Aucune particularité corpus ne vit dans le code :
encoding, délimiteur, indices/noms de colonnes, archive… sont des VALEURS du
descripteur.

Schéma du descripteur (cf. `descriptors/*.json`) :

```jsonc
{
  "name": "tiktok",            // identifiant de source (préfixe des ids, author_hash)
  "format": "csv",             // "csv" | "jsonl"
  "path": "data/raw/x.csv",    // relatif à la racine repo (ou absolu)
  "url": "https://…",          // optionnel : pour le téléchargement idempotent
  "encoding": "cp1252",        // défaut "utf-8"
  "delimiter": ";",            // csv uniquement, défaut ","
  "has_header": true,          // csv uniquement, défaut true
  "archive": "zip",            // optionnel : le `path` est une archive zip…
  "members": ["train.jsonl"],  // …et on lit ces fichiers membres (jsonl)
  "columns": {                 // mapping CHAMP CANONIQUE -> référence de colonne
    "id":   0,                 //   int  => index 0-based (csv sans noms)
    "text": 141,               //   str  => clé (jsonl, ou csv via en-tête nommé)
    "ts":   1,                 //   ts/author/lang/weight sont OPTIONNELS
    "author": "author",
    "lang": "language"
  },
  "lang_keep": ["fr"],         // optionnel : KNOB explicite de sous-ensemble langue
  "keep_where": {              // optionnel : KNOB déclaratif de filtrage par valeur
    "Type.de.contenu": ["Proposition", "Argument", "Modification"]
  },                           //   garde la ligne si CHAQUE colonne ∈ ensemble autorisé
                               //   (défaut absent = aucun filtre par valeur)
  "props": {                   // optionnel : MÉTADONNÉES de source à PRÉSERVER dans
    "question": "question",    //   `props` de chaque idée (nom_prop -> réf colonne).
    "topic": "topic",          //   Générique : aucun champ corpus en dur. Sert au
    "label": "label"           //   modèle mère→enfants (ex. xstance : question/topic/
  }                            //   label par item). N'écrase jamais les props canoniques.
}

Seuls `id` et `text` sont obligatoires dans `columns`. `author` non fourni →
on retombe sur la valeur d'`id` (1 réponse = 1 répondant). `lang` fourni par la
source est conservé tel quel ; sinon le langage est (re)détecté en aval (#13).
"""
from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import config

csv.field_size_limit(10_000_000)  # certains témoignages libres sont très longs

# Racine repo = deux niveaux au-dessus de ce fichier (pipeline/ingest/sources.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Champs canoniques que le mapping `columns` peut renseigner.
_REQUIRED_FIELDS = ("id", "text")
_OPTIONAL_FIELDS = ("ts", "author", "lang", "weight")
_KNOWN_FIELDS = _REQUIRED_FIELDS + _OPTIONAL_FIELDS


@dataclass
class SourceDescriptor:
    """Description déclarative d'un corpus à ingérer. Un corpus = une config."""

    name: str
    format: str  # "csv" | "jsonl"
    path: str
    columns: dict  # champ canonique -> int (index) | str (clé)
    url: str | None = None
    encoding: str = "utf-8"
    delimiter: str = ","
    has_header: bool = True
    archive: str | None = None  # "zip" ou None
    members: list[str] | None = None
    lang_keep: list[str] | None = None
    keep_where: dict | None = None  # filtre déclaratif par valeur de colonne brute
    # Métadonnées de source à conserver dans `props` (nom_prop -> réf colonne).
    # Déclaratif et générique : sert le modèle mère→enfants (question/topic/label…).
    props: dict | None = None
    # Statut de la consultation : "open" (participation en cours) | "closed"
    # (close, on n'expose que l'analyse). Défaut prudent = "closed".
    status: str = "closed"
    # Champs additionnels du JSON conservés sans être interprétés (forward-compat).
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format not in ("csv", "jsonl"):
            raise ValueError(f"{self.name}: format inconnu {self.format!r} (csv|jsonl)")
        if self.status not in ("open", "closed"):
            raise ValueError(f"{self.name}: status inconnu {self.status!r} (open|closed)")
        for f in _REQUIRED_FIELDS:
            if f not in self.columns:
                raise ValueError(f"{self.name}: colonne obligatoire absente: {f!r}")
        unknown = set(self.columns) - set(_KNOWN_FIELDS)
        if unknown:
            raise ValueError(f"{self.name}: champ(s) canonique(s) inconnu(s): {unknown}")

    # -- chargement -------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "SourceDescriptor":
        known = {
            "name", "format", "path", "columns", "url", "encoding",
            "delimiter", "has_header", "archive", "members", "lang_keep",
            "keep_where", "status", "props",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["extra"] = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "SourceDescriptor":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def resolved_path(self, root: Path | None = None) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() else (root or _REPO_ROOT) / p


def _uses_names(columns: dict) -> bool:
    """True si au moins une référence de colonne est un nom (str) plutôt qu'un index."""
    return any(isinstance(ref, str) for ref in columns.values())


def _get(record, ref):
    """Récupère une valeur par index (list/tuple) ou par clé (dict)."""
    if ref is None:
        return None
    try:
        if isinstance(ref, int) and isinstance(record, (list, tuple)):
            return record[ref] if 0 <= ref < len(record) else None
        return record.get(ref) if isinstance(record, dict) else None
    except (KeyError, IndexError, TypeError):
        return None


def _map_record(desc: SourceDescriptor, record) -> dict:
    """Mappe une ligne/objet brut vers l'enregistrement canonique de l'ingestion."""
    cols = desc.columns
    raw_id = _get(record, cols["id"])
    raw_id = "" if raw_id is None else str(raw_id)

    text = _get(record, cols["text"])
    text = "" if text is None else str(text)

    # author par défaut = id (1 réponse = 1 répondant) si non mappé.
    author = _get(record, cols["author"]) if "author" in cols else None
    author = raw_id if author is None else str(author)

    ts = _get(record, cols["ts"]) if "ts" in cols else None
    ts = "" if ts is None else str(ts)

    lang = _get(record, cols["lang"]) if "lang" in cols else None
    lang = "" if lang is None else str(lang).strip()

    rec = {
        "raw_id": raw_id,
        "text": text,
        "author": author,
        "source": desc.name,
        "ts": ts,
        "lang": lang,  # "" => (re)détection en aval (audit #13)
    }
    if "weight" in cols:
        w = _get(record, cols["weight"])
        try:
            rec["weight"] = float(w)
        except (TypeError, ValueError):
            rec["weight"] = 1.0
    # Métadonnées de source à préserver (déclaratif). On garde la valeur brute
    # (chaîne non vide après strip) ; les manquantes sont simplement omises.
    if desc.props:
        extra: dict = {}
        for prop_name, ref in desc.props.items():
            val = _get(record, ref)
            if val is None:
                continue
            sval = str(val).strip()
            if sval:
                extra[prop_name] = sval
        if extra:
            rec["props"] = extra
    return rec


def _passes_lang_keep(desc: SourceDescriptor, rec: dict) -> bool:
    """Filtre langue EXPLICITE (knob). Absent => on garde tout (défaut multilingue)."""
    if not desc.lang_keep:
        return True
    return rec.get("lang", "") in desc.lang_keep


def _passes_keep_where(desc: SourceDescriptor, record) -> bool:
    """Filtre déclaratif par VALEUR de colonne brute (knob), avant mapping canonique.

    `keep_where` = {réf_colonne: [valeurs autorisées]}. La ligne passe si CHAQUE
    colonne référencée a une valeur dans son ensemble autorisé. Réf identique au
    mapping `columns` (nom str pour jsonl/csv-nommé, index int pour csv brut ; une
    clé JSON numérique « 7 » est coercée en index). Absent => on garde tout.
    """
    if not desc.keep_where:
        return True
    for ref, allowed in desc.keep_where.items():
        if isinstance(ref, str) and ref.lstrip("-").isdigit():
            ref = int(ref)
        val = _get(record, ref)
        val = "" if val is None else str(val)
        if val not in allowed:
            return False
    return True


def _read_csv(desc: SourceDescriptor, path: Path) -> Iterator[dict]:
    with open(path, encoding=desc.encoding, newline="") as f:
        if _uses_names(desc.columns):
            # Colonnes nommées -> on s'appuie sur l'en-tête (DictReader).
            for row in csv.DictReader(f, delimiter=desc.delimiter):
                if _passes_keep_where(desc, row):
                    yield _map_record(desc, row)
        else:
            rd = csv.reader(f, delimiter=desc.delimiter)
            if desc.has_header:
                next(rd, None)
            for row in rd:
                if _passes_keep_where(desc, row):
                    yield _map_record(desc, row)


def _read_jsonl_lines(desc: SourceDescriptor, lines) -> Iterator[dict]:
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode(desc.encoding, errors="replace")
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and _passes_keep_where(desc, obj):
            yield _map_record(desc, obj)


def _read_jsonl(desc: SourceDescriptor, path: Path) -> Iterator[dict]:
    if desc.archive == "zip":
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            members = desc.members or [n for n in names if n.endswith(".jsonl")]
            present = set(names)
            for member in members:
                if member not in present:
                    continue
                with z.open(member) as fh:
                    yield from _read_jsonl_lines(desc, fh)
    else:
        with open(path, "rb") as fh:
            yield from _read_jsonl_lines(desc, fh)


def load_descriptors(paths: list[Path] | None = None) -> list["SourceDescriptor"]:
    """Charge des descripteurs explicites, ou tous ceux de `descriptors/` (triés).

    Les descripteurs `status: open` (consultations sans fichier source — collecte
    live via le backend) sont ignorés : ils n'ont pas les champs `format/path/columns`.
    """
    if paths:
        return [SourceDescriptor.from_json(p) for p in paths]
    files = sorted(config.DESCRIPTORS_DIR.glob("*.json"))
    result = []
    for p in files:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("status") == "open":
            continue
        result.append(SourceDescriptor.from_dict(raw))
    return result


def read_generic(desc: SourceDescriptor, root: Path | None = None) -> Iterator[dict]:
    """Lit N'IMPORTE QUELLE source décrite par `desc` -> enregistrements canoniques.

    Ne bloque pas si le fichier est absent (yield vide) : `build` enchaîne les
    autres sources / le repli synthétique. Le filtre `lang_keep` (knob explicite)
    n'est appliqué que s'il est défini dans le descripteur.
    """
    path = desc.resolved_path(root)
    if not path.exists():
        return
    reader = _read_csv if desc.format == "csv" else _read_jsonl
    for rec in reader(desc, path):
        if _passes_lang_keep(desc, rec):
            yield rec

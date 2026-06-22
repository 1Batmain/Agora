"""Modèle de CLAIM EXTRACTIF : une portion VERBATIM de l'avis (zéro hallucination).

Le LLM ne reformule plus — il SÉLECTIONNE des portions de l'avis. Pour garantir la
fidélité, chaque portion renvoyée par le LLM est VALIDÉE comme une **sous-chaîne
exacte** du texte de l'avis, et on en dérive nous-mêmes les offsets (le LLM ne
fournit pas de positions, peu fiables) :

  1. correspondance exacte (`str.find`) → offsets directs ;
  2. sinon, alignement TOLÉRANT aux espaces (le petit modèle recompacte parfois les
     blancs/sauts de ligne) : on cherche dans une version à espaces normalisés et on
     remappe vers les positions d'origine ;
  3. sinon, REJET (on n'invente jamais d'offset → aucun mot hors de l'avis).

Le `text` d'un `Claim` est TOUJOURS retranché du texte d'origine (`avis_text[start:end]`),
donc verbatim par construction, quelle que soit la voie d'alignement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """Une idée du citoyen, ancrée comme portion verbatim de son avis.

    `text` = `avis_text[start:end]` (sous-chaîne exacte). `start`/`end` sont des
    offsets de CARACTÈRES dans le texte de l'avis (mi-ouverts, `end` exclu). Un
    claim sans ancrage connu (repli legacy) a `start == end == -1`.
    """

    text: str
    start: int
    end: int

    @property
    def anchored(self) -> bool:
        return self.start >= 0 and self.end > self.start

    def is_verbatim(self, avis_text: str) -> bool:
        """Vrai si le claim correspond EXACTEMENT à `avis_text[start:end]`."""
        return self.anchored and avis_text[self.start:self.end] == self.text

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end}


def as_claim(obj, *, avis_text: str | None = None) -> Claim:
    """Normalise une valeur (Claim / dict cache / str legacy) en `Claim`.

    `dict` : `{text, start, end}` (cache disque). `str` : claim sans offsets (repli
    legacy) — ré-ancré sur `avis_text` si fourni, sinon `start=end=-1`.
    """
    if isinstance(obj, Claim):
        return obj
    if isinstance(obj, dict):
        return Claim(text=str(obj.get("text", "")),
                     start=int(obj.get("start", -1)),
                     end=int(obj.get("end", -1)))
    text = str(obj)
    if avis_text is not None:
        i = avis_text.find(text)
        if i >= 0:
            return Claim(text=avis_text[i:i + len(text)], start=i, end=i + len(text))
    return Claim(text=text, start=-1, end=-1)


def whole_avis_claim(avis_text: str) -> Claim:
    """Claim de repli couvrant l'avis entier (jamais perdu si l'extraction échoue)."""
    return Claim(text=avis_text, start=0, end=len(avis_text))


# --------------------------------------------------------------------------- #
# Normalisation des espaces avec carte d'index → position d'origine
# --------------------------------------------------------------------------- #
def _normalize_ws(s: str) -> tuple[str, list[int]]:
    """`s` à blancs compactés (runs d'espaces → 1 espace) + carte index→pos d'origine.

    `idx_map[k]` = position dans `s` du k-ième caractère de la chaîne normalisée.
    Permet de retrouver les offsets ORIGINAUX après une recherche sur la version
    normalisée.
    """
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


def _locate(avis_text: str, candidate: str, search_from: int,
            norm_cache: tuple[str, list[int]] | None) -> Claim | None:
    """Ancre `candidate` dans `avis_text` (exact puis tolérant aux espaces), ou None."""
    cand = candidate.strip()
    if not cand:
        return None

    # 1) Correspondance exacte (priorité aux occurrences après `search_from`).
    i = avis_text.find(cand, search_from)
    if i < 0:
        i = avis_text.find(cand)
    if i >= 0:
        return Claim(text=avis_text[i:i + len(cand)], start=i, end=i + len(cand))

    # 2) Alignement tolérant aux espaces (remappé vers les positions d'origine).
    norm_text, idx_map = norm_cache if norm_cache is not None else _normalize_ws(avis_text)
    norm_cand, _ = _normalize_ws(cand)
    if not norm_cand:
        return None
    j = norm_text.find(norm_cand)
    if j < 0:
        return None
    start = idx_map[j]
    end = idx_map[j + len(norm_cand) - 1] + 1
    return Claim(text=avis_text[start:end], start=start, end=end)


def align_spans(avis_text: str, candidates: list[str]) -> list[Claim]:
    """Ancre chaque portion candidate dans l'avis → `list[Claim]` verbatim.

    Les candidats non ancrables (ni exact ni tolérant aux espaces) sont REJETÉS —
    garantie zéro mot inventé. Un curseur par texte évite que des candidats identiques
    se collent tous à la 1ʳᵉ occurrence (offsets distincts pour des répétitions).
    """
    norm_cache = _normalize_ws(avis_text)
    cursor: dict[str, int] = {}
    out: list[Claim] = []
    for cand in candidates:
        key = cand.strip()
        claim = _locate(avis_text, cand, cursor.get(key, 0), norm_cache)
        if claim is None:
            continue
        out.append(claim)
        cursor[key] = claim.end
    return out

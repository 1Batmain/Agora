"""Modèle de CLAIM EXTRACTIF MULTI-SPANS : 1..N portions VERBATIM + cible (zéro hallu).

Le LLM ne reformule plus — il SÉLECTIONNE des portions de l'avis. Un claim peut prendre
PLUSIEURS portions NON-CONTIGUËS de l'avis (« parts »), p.ex. la phrase qui pose l'idée
+ la fin d'une phrase ultérieure qui s'y réfère → UNE seule unité de sens. Il porte aussi
une **cible** (`target`) : l'aspect dont il parle, lui AUSSI une portion VERBATIM de l'avis
(p.ex. « temps passé sur l'écran »), normalisée en aspect propre seulement EN AVAL.

Pour garantir la fidélité, chaque portion (chaque part ET la target) renvoyée par le LLM
est VALIDÉE comme **sous-chaîne exacte** du texte de l'avis, et on en dérive nous-mêmes
les offsets (le LLM ne fournit pas de positions, peu fiables) :

  1. correspondance exacte (`str.find`) → offsets directs ;
  2. sinon, alignement TOLÉRANT aux espaces (le petit modèle recompacte parfois les
     blancs/sauts de ligne) : on cherche dans une version à espaces normalisés et on
     remappe vers les positions d'origine ;
  3. sinon, REJET (on n'invente jamais d'offset → aucun mot hors de l'avis).

Le `text` d'un `Claim` est TOUJOURS le JOINT des sous-chaînes d'origine (`avis_text[s:e]`
par span, séparées par `SPAN_JOIN`), donc verbatim par construction. Mono-span = liste
de 1 → rétro-compatible avec l'ancien modèle (`start`/`end` exposés en propriétés).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Séparateur d'affichage entre portions non-contiguës d'un claim multi-spans. Visualise
# l'omission (et reste neutre pour l'embedding du texte joint). Mono-span → jamais utilisé.
SPAN_JOIN = " … "

# Un span = (start, end) en offsets de CARACTÈRES dans l'avis (mi-ouvert, `end` exclu).
Span = tuple[int, int]


@dataclass(frozen=True)
class Claim:
    """Une idée du citoyen, ancrée comme 1..N portions verbatim de son avis, + cible.

    `spans` = portions verbatim (offsets caractères mi-ouverts) ; `text` = leur JOINT
    (`SPAN_JOIN`). `target` = la cible/aspect verbatim (un span sous-portion de l'avis),
    ou `None`. Un claim sans ancrage connu (repli legacy) a un unique span `(-1, -1)`.
    """

    text: str
    spans: tuple[Span, ...] = field(default_factory=tuple)
    target: Span | None = None

    @property
    def start(self) -> int:
        """Borne gauche (1er span) — rétro-compat avec l'ancien modèle mono-span."""
        return self.spans[0][0] if self.spans else -1

    @property
    def end(self) -> int:
        """Borne droite (dernier span) — rétro-compat avec l'ancien modèle mono-span."""
        return self.spans[-1][1] if self.spans else -1

    @property
    def anchored(self) -> bool:
        """Vrai si TOUTES les portions sont ancrées (offsets valides, non vides)."""
        return bool(self.spans) and all(e > s >= 0 for s, e in self.spans)

    def is_verbatim(self, avis_text: str) -> bool:
        """Vrai si le texte joint == join des `avis_text[s:e]` ET la target (si présente) ancrée."""
        if not self.anchored:
            return False
        joined = SPAN_JOIN.join(avis_text[s:e] for s, e in self.spans)
        if joined != self.text:
            return False
        if self.target is not None:
            ts, te = self.target
            if not (0 <= ts < te <= len(avis_text)):
                return False
        return True

    def to_dict(self) -> dict:
        """Sérialisation cache : `{text, spans:[[s,e],...], target:[s,e]|null}`."""
        return {
            "text": self.text,
            "spans": [[s, e] for s, e in self.spans],
            "target": list(self.target) if self.target is not None else None,
        }


def _as_span_tuple(val) -> Span | None:
    """`[s,e]`/`(s,e)` → `(int, int)`, sinon None (tolérant aux entrées cache/LLM)."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            return (int(val[0]), int(val[1]))
        except (TypeError, ValueError):
            return None
    return None


def as_claim(obj, *, avis_text: str | None = None) -> Claim:
    """Normalise une valeur (Claim / dict cache / str legacy) en `Claim` multi-spans.

    `dict` NOUVEAU format : `{text, spans:[[s,e],...], target}`. `dict` LEGACY :
    `{text, start, end}` → un seul span. `str` : claim sans offsets (repli legacy) —
    ré-ancré sur `avis_text` si fourni, sinon span `(-1, -1)`.
    """
    if isinstance(obj, Claim):
        return obj
    if isinstance(obj, dict):
        text = str(obj.get("text", ""))
        target = _as_span_tuple(obj.get("target"))
        spans_raw = obj.get("spans")
        if isinstance(spans_raw, list) and spans_raw:
            spans = tuple(s for s in (_as_span_tuple(x) for x in spans_raw) if s is not None)
            if spans:
                return Claim(text=text, spans=spans, target=target)
        # Repli LEGACY mono-span {text, start, end}.
        start = int(obj.get("start", -1))
        end = int(obj.get("end", -1))
        return Claim(text=text, spans=((start, end),), target=target)
    text = str(obj)
    if avis_text is not None:
        i = avis_text.find(text)
        if i >= 0:
            return Claim(text=avis_text[i:i + len(text)], spans=((i, i + len(text)),))
    return Claim(text=text, spans=((-1, -1),))


def whole_avis_claim(avis_text: str) -> Claim:
    """Claim de repli couvrant l'avis entier (jamais perdu si l'extraction échoue)."""
    return Claim(text=avis_text, spans=((0, len(avis_text)),), target=None)


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
            norm_cache: tuple[str, list[int]] | None) -> Span | None:
    """Ancre `candidate` dans `avis_text` (exact puis tolérant aux espaces) → span, ou None."""
    cand = candidate.strip()
    if not cand:
        return None

    # 1) Correspondance exacte (priorité aux occurrences après `search_from`).
    i = avis_text.find(cand, search_from)
    if i < 0:
        i = avis_text.find(cand)
    if i >= 0:
        return (i, i + len(cand))

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
    return (start, end)


def align_spans(avis_text: str, specs: list[dict]) -> list[Claim]:
    """Ancre chaque spec `{parts:[...], target}` → `Claim` multi-spans verbatim.

    Pour chaque claim : chaque **part** est ancrée (exact ou tolérant aux espaces) ;
    les parts non ancrables sont REJETÉES (garantie zéro mot inventé). Un claim sans
    AUCUNE part ancrée est entièrement écarté. La **target** est ancrée de même (1ʳᵉ
    occurrence) ; non ancrable → `target=None` (le claim reste). Un curseur par texte
    évite que des portions identiques se collent toutes à la 1ʳᵉ occurrence (offsets
    distincts pour des répétitions).
    """
    norm_cache = _normalize_ws(avis_text)
    cursor: dict[str, int] = {}
    out: list[Claim] = []
    for spec in specs:
        parts = spec.get("parts") if isinstance(spec, dict) else None
        if not isinstance(parts, list):
            continue
        spans: list[Span] = []
        texts: list[str] = []
        for part in parts:
            key = str(part).strip()
            if not key:
                continue
            span = _locate(avis_text, part, cursor.get(key, 0), norm_cache)
            if span is None:
                continue
            spans.append(span)
            texts.append(avis_text[span[0]:span[1]])
            cursor[key] = span[1]
        if not spans:
            continue
        target: Span | None = None
        traw = spec.get("target")
        if isinstance(traw, str) and traw.strip():
            # 1ʳᵉ occurrence : la cible est typiquement une sous-portion d'une des parts.
            target = _locate(avis_text, traw, 0, norm_cache)
        out.append(Claim(text=SPAN_JOIN.join(texts), spans=tuple(spans), target=target))
    return out

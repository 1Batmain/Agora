"""Transcripts de séance → TOURS DE PAROLE exploitables (aucun LLM, déterministe).

Premier maillon du pipeline LIVE (`.agent/notes/LIVE_SCENARIO.md`). Convertit les comptes
rendus XML de l'Assemblée nationale (open data, Licence ouverte) en tours de parole
**nommés et horodatés**, prêts pour l'extraction de claims.

Ce module est ISOLÉ du pipeline servi : il n'importe rien de `backend/`, ne touche aucun
cache de consultation, et ne peut donc rien casser de l'Agora existant.

## Ce qu'il fait, et pourquoi — tout est adossé à une mesure sur les 601 séances

1. **Filtre procédural par `code_grammaire`.** Le corpus TYPE lui-même la nature de chaque
   prise de parole : on garde `PAROLE`/`DISC`/`RAP` (médiane 172–234 caractères) et on écarte
   `INTERRUPTION` (22,4 % du corpus, médiane 27 c : « Très bien ! »), `SCRUT`, `SUSP`, `ADOP`…
2. **Filtre du perchoir DURCI.** `roledebat="president"` n'est pas toujours posé : **1 629
   tours** (1,7 %) portent « M./Mme le·a président·e » sans l'attribut. On filtre donc AUSSI
   sur le libellé d'orateur — sinon la synthèse se remplit de rappels à l'ordre.
3. **Recollage par orateur.** **36,9 %** des tours retenus suivent le même orateur : une
   intervention est COUPÉE en plusieurs paragraphes par les interruptions (6,9 % reprennent
   même en plein milieu de phrase, avec un « … » initial). **L'unité n'est pas le paragraphe,
   c'est la suite de paragraphes d'un même orateur.** Recoller AVANT de filtrer sur la
   longueur, sinon on jette des morceaux qui, réunis, formaient un raisonnement.
4. **Réactions séparées du verbatim.** « (Applaudissements sur les bancs du groupe X.) » est
   une annotation éditoriale, pas de la parole : la laisser pollue les claims (observé). On
   l'extrait dans `reactions` — c'est d'ailleurs un SIGNAL (qui applaudit qui), pas un déchet.

   ⚠️ **On ne supprime JAMAIS une parenthèse par défaut.** Mesuré : sur 25 455 parenthèses,
   ~19 % sont des **sigles prononcés par l'orateur** — « (PLFSS) », « (CSG) », « (ZAN) »,
   « (Drees) ». Les retirer casserait le verbatim. Seul ce qui matche le lexique de réaction
   est retiré ; l'inconnu est CONSERVÉ (même principe que le registre d'embedders : ne rien
   faire vaut mieux que faire faux).

Source : `https://data.assemblee-nationale.fr/travaux-parlementaires/debats`
(`syseron.xml.zip`, rafraîchi quotidiennement).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

# --------------------------------------------------------------------------------- #
# Règles de filtrage — chacune adossée à une mesure (cf. LIVE_DATA_RECON.md)
# --------------------------------------------------------------------------------- #

# Types de prise de parole PORTEURS DE FOND. Le reste du corpus est de la procédure.
SUBSTANTIVE_CODES = frozenset({"PAROLE", "DISC", "RAP"})

# Ce qui ROMPT une intervention sans être émis. ⚠️ Distinction cruciale, apprise sur données
# réelles : une INTERRUPTION (« Très bien ! ») ne rompt PAS le fil — l'orateur reprend après ;
# un scrutin, une suspension, une adoption, SI. Et surtout le PERCHOIR : « La parole est à
# M. X » est exactement le signal qu'une intervention se termine et qu'une autre commence.
#
# Sans cette notion de frontière, filtrer le perchoir AVANT de recoller rendait adjacents deux
# passages du même orateur séparés par tout un débat : mesuré sur la séance retraite, 80
# paragraphes fusionnés en un faux « tour » de 20 655 caractères. Les fixtures ne contenaient
# pas de reprise de parole par le perchoir — seules les données réelles l'ont montré.
BOUNDARY_CODES = frozenset({
    "SCRUT", "SCR", "SCRUPUB", "SUSP", "ADOP", "REJET", "ANN", "FIN", "VOTE",
    "ODJ", "MOTION", "OUV", "RETRAIT", "ART", "ADT", "AM8ENDSORT", "SORT", "QOSD", "QG",
})

# Le perchoir : `roledebat` d'abord, puis le LIBELLÉ (l'attribut manque sur 1,7 % des tours).
CHAIR_ROLE = "president"
CHAIR_LABEL_RE = re.compile(r"^\s*(M\.|Mme)\s+l[ea]\s+président", re.IGNORECASE)

# Didascalie mise en forme par l'éditeur du compte rendu (« (La séance est ouverte…) »).
STAGE_STYLE = "Info Italiques"

# Réactions de séance + didascalies éditoriales. Tout ce qui NE matche PAS reste dans le
# texte : les sigles prononcés (« (PLFSS) ») doivent survivre intacts.
REACTION_RE = re.compile(
    # Réactions sonores de l'hémicycle.
    r"applaudi|exclamation|protest|sourire|rire[s.]|murmure|hu[ée]e|brouhaha|"
    r"approbation|interruption|mouvement|vives? |claquement|bruit|tumulte|"
    # Didascalies éditoriales (le rédacteur décrit la séance, personne ne parle).
    r"coupe le micro|temps de parole|il est proc[ée]d[ée]|la s[ée]ance est|"
    # Gestes décrits par le compte rendu — sans ambiguïté dans ce registre.
    r"fait signe|agite|brandit|se l[èe]ve|reste assis|quitte l|applaudit",
    re.IGNORECASE,
)
# Interjection citée : « Ah ! », « Oh ! » — de la réaction, pas de la parole de l'orateur.
QUOTED_SHOUT_RE = re.compile(r"^\s*«")

# Borne haute de la parenthèse considérée. Mesuré : 0,3 % des parenthèses dépassent 200 c
# (max 446), et 99 % d'entre elles sont des réactions en cascade (« …– Exclamations… –
# M. X agite… »). Une borne à 200 les laissait polluer le verbatim. `[^()]` ne peut pas
# franchir une parenthèse : élargir ne risque pas d'avaler du texte imbriqué.
PARENTHETICAL_RE = re.compile(r"\(([^()]{1,500})\)")

# Marque de reprise en cours de phrase, laissée par l'éditeur au recollage d'une
# intervention interrompue (6,9 % des tours). Purement typographique.
RESUME_MARK_RE = re.compile(r"^\s*(…|\.\.\.)\s*")

# Longueur minimale d'un tour EXPLOITABLE, appliquée APRÈS recollage. Aligné sur le seuil
# utilisé à la reconnaissance ; c'est un paramètre de mesure, pas une vérité.
MIN_CHARS = 200

# Suffixe numérique collé au libellé d'orateur dans `<orateurs>` (« Mme X 795050 »).
SPEAKER_ID_SUFFIX_RE = re.compile(r"\s*\d{5,}.*$")


@dataclass
class Turn:
    """Un tour de parole recollé : ce qu'UN orateur a dit d'affilée.

    `paragraph_ids` garde la trace des paragraphes d'origine — la traçabilité jusqu'à la
    source est le produit, pas un bonus (même exigence que la provenance des verbatims).
    """

    id: str
    session: str
    seq: int                       # `ordre_absolu_seance` du 1er paragraphe (ordre de séance)
    speaker: str                   # libellé affiché, ex. « Mme Mathilde Panot »
    speaker_id: str | None         # `id_acteur` stable, ex. « PA795050 » (joignable aux acteurs AN)
    role: str                      # `roledebat` (ex. « rapporteur »), "" si absent
    stime: float | None            # timecode en secondes depuis l'ouverture (90,4 % du corpus)
    text: str
    reactions: list[str] = field(default_factory=list)
    paragraph_ids: list[str] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class SessionMeta:
    """Métadonnées d'une séance (date, numéro, sujets de l'ordre du jour)."""

    uid: str
    date: str
    date_label: str
    session: str
    num_seance: str
    president: str
    topics: list[str]


def _tag(elem) -> str:
    return elem.tag.split("}")[-1]


def _text_of(elem) -> str:
    return " ".join("".join(elem.itertext()).split())


def split_reactions(text: str) -> tuple[str, list[str]]:
    """Sépare le verbatim des réactions de séance.

    Renvoie `(texte_nettoyé, réactions)`. Une parenthèse qui ne matche NI le lexique de
    réaction NI une interjection citée est LAISSÉE dans le texte : c'est très probablement
    un sigle prononcé (« (PLFSS) »), et le supprimer casserait l'ancrage verbatim.
    """
    reactions: list[str] = []

    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        if REACTION_RE.search(inner) or QUOTED_SHOUT_RE.match(inner):
            reactions.append(inner.strip())
            return " "
        return m.group(0)                      # inconnu → on GARDE tel quel

    cleaned = PARENTHETICAL_RE.sub(_replace, text)
    return " ".join(cleaned.split()), reactions


def is_chair(speaker_label: str, role: str) -> bool:
    """Le perchoir, détecté par le rôle OU par le libellé (l'attribut manque 1,7 % du temps)."""
    return role == CHAIR_ROLE or bool(CHAIR_LABEL_RE.match(speaker_label or ""))


def read_session_meta(path: Path | str) -> SessionMeta:
    root = ET.parse(str(path)).getroot()
    vals: dict[str, str] = {}
    topics: list[str] = []
    for e in root.iter():
        name = _tag(e)
        if name == "intitule":
            t = _text_of(e)
            if t and t not in topics:
                topics.append(t)
        elif name in ("dateSeance", "dateSeanceJour", "session", "numSeance",
                      "presidentSeance", "uid") and name not in vals:
            vals[name] = (e.text or "").strip()
    return SessionMeta(
        uid=vals.get("uid", Path(path).stem),
        date=vals.get("dateSeance", ""),
        date_label=vals.get("dateSeanceJour", ""),
        session=vals.get("session", ""),
        num_seance=vals.get("numSeance", ""),
        president=vals.get("presidentSeance", ""),
        topics=topics,
    )


def _raw_paragraphs(root, session_uid: str) -> Iterator[dict]:
    """Paragraphes classés en trois catégories, dans l'ordre de séance.

    `kind` vaut :
      - `"turn"`     : parole de fond d'un orateur → à émettre (et à recoller) ;
      - `"boundary"` : rupture d'intervention (perchoir qui donne la parole, scrutin,
        suspension…) → NON émis, mais **ferme** le tour en cours ;
      - `"skip"`     : transparent (interruption, didascalie) → n'interrompt rien.

    La distinction boundary/skip est ce qui empêche de fusionner deux interventions
    distinctes du même orateur (cf. `BOUNDARY_CODES`).
    """
    for p in root.iter():
        if _tag(p) != "paragraphe":
            continue
        code = (p.attrib.get("code_grammaire") or "").split("_")[0]

        speaker_raw, text, stime = "", "", None
        for child in p:
            if _tag(child) == "orateurs":
                speaker_raw = _text_of(child)
            elif _tag(child) == "texte":
                text = _text_of(child)
                stime = child.attrib.get("stime")

        speaker = SPEAKER_ID_SUFFIX_RE.sub("", speaker_raw).strip()
        role = p.attrib.get("roledebat", "")

        if code in BOUNDARY_CODES:
            kind = "boundary"
        elif code not in SUBSTANTIVE_CODES:
            kind = "skip"                       # INTERRUPTION & co : ne rompent pas le fil
        elif p.attrib.get("code_style") == STAGE_STYLE:
            kind = "skip"
        elif is_chair(speaker, role):
            kind = "boundary"                   # « La parole est à M. X » = fin d'intervention
        elif not text:
            kind = "skip"
        else:
            kind = "turn"

        yield {
            "kind": kind,
            "session": session_uid,
            "seq": int(p.attrib.get("ordre_absolu_seance") or 0),
            "speaker": speaker,
            "speaker_id": p.attrib.get("id_acteur") or None,
            "role": role,
            "stime": float(stime) if stime else None,
            "text": text,
            "pid": p.attrib.get("id_syceron") or str(p.attrib.get("ordre_absolu_seance") or ""),
        }


def stitch(paragraphs: list[dict], *, min_chars: int = MIN_CHARS) -> list[Turn]:
    """Recolle les paragraphes CONSÉCUTIFS d'un même orateur en un tour, puis filtre.

    L'ORDRE compte : recoller D'ABORD, filtrer sur la longueur ENSUITE. L'inverse jetterait
    des fragments courts qui, réunis, forment l'intervention (36,9 % des paragraphes sont
    une suite du même orateur).

    L'identité d'orateur se juge sur `id_acteur` quand il existe (90,1 % des cas) — deux
    homonymes de libellé ne seront pas confondus — et sur le libellé sinon.

    Chaque entrée porte un `kind` (`turn` / `boundary` / `skip`, cf. `_raw_paragraphs`). Un
    dict sans `kind` est traité comme `turn` : c'est le défaut PRUDENT pour les appelants
    directs (tests, autres sources de transcript) — on n'invente pas de frontière.
    """
    turns: list[Turn] = []
    current: Turn | None = None
    current_key: tuple | None = None

    for p in paragraphs:
        kind = p.get("kind", "turn")
        if kind == "skip":
            continue
        if kind == "boundary":
            # Ferme l'intervention en cours SANS rien émettre : le prochain paragraphe du
            # même orateur sera une intervention NOUVELLE, pas une suite.
            if current is not None:
                turns.append(current)
                current, current_key = None, None
            continue

        key = (p["speaker_id"] or p["speaker"],)
        clean, reactions = split_reactions(p["text"])
        clean = RESUME_MARK_RE.sub("", clean)
        if not clean:
            # Le paragraphe n'était QUE de la réaction : on garde le signal sur le tour en
            # cours, sans rompre le recollage (une salve d'applaudissements n'interrompt pas
            # l'intervention).
            if current is not None:
                current.reactions.extend(reactions)
            continue

        if current is not None and key == current_key:
            current.text = f"{current.text} {clean}".strip()
            current.reactions.extend(reactions)
            current.paragraph_ids.append(p["pid"])
            continue

        if current is not None:
            turns.append(current)
        current = Turn(
            id=f"{p['session']}:{p['seq']}",
            session=p["session"],
            seq=p["seq"],
            speaker=p["speaker"],
            speaker_id=p["speaker_id"],
            role=p["role"],
            stime=p["stime"],
            text=clean,
            reactions=list(reactions),
            paragraph_ids=[p["pid"]],
        )
        current_key = key

    if current is not None:
        turns.append(current)

    return [t for t in turns if t.n_chars >= min_chars]


def parse_session(path: Path | str, *, min_chars: int = MIN_CHARS) -> list[Turn]:
    """Un compte rendu XML → ses tours de parole exploitables, dans l'ordre de séance."""
    root = ET.parse(str(path)).getroot()
    meta = read_session_meta(path)
    paragraphs = sorted(_raw_paragraphs(root, meta.uid or Path(path).stem),
                        key=lambda p: p["seq"])
    return stitch(paragraphs, min_chars=min_chars)


def write_jsonl(turns: list[Turn], out_path: Path | str) -> int:
    """Écrit les tours en JSONL (une ligne = un tour). Renvoie le nombre écrit."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for t in turns:
            fh.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    return len(turns)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Comptes rendus AN (XML) → tours de parole JSONL (aucun LLM).")
    ap.add_argument("xml", nargs="+", help="fichier(s) de compte rendu XML")
    ap.add_argument("--out", required=True, help="fichier JSONL de sortie")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = ap.parse_args()

    all_turns: list[Turn] = []
    for src in args.xml:
        turns = parse_session(src, min_chars=args.min_chars)
        meta = read_session_meta(src)
        print(f"  {Path(src).name} · {meta.date_label or '?'} · {len(turns)} tours"
              f" · {meta.topics[0][:60] if meta.topics else ''}", flush=True)
        all_turns.extend(turns)

    n = write_jsonl(all_turns, args.out)
    speakers = {t.speaker_id or t.speaker for t in all_turns}
    chars = sum(t.n_chars for t in all_turns)
    print(f"\n{n} tours · {len(speakers)} orateurs · {chars/1000:.0f} k caractères → {args.out}")


if __name__ == "__main__":
    main()

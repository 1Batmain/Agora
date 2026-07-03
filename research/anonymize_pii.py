"""Anonymisation renforcée — post-traite du texte DÉJÀ passé par `strip_pii`
(`pipeline.ingest.normalize`) pour rattraper ce que le regex de base laisse
passer (cf. `research/audit_privacy.md` #3) : pseudos réseaux sociaux sans
`@`, adresses postales, téléphones hors format strict, et noms de personnes
via une petite passe NER.

Volontairement SANS LLM (coût/latence à l'échelle du corpus) : regex étendu +
spaCy `fr_core_news_lg` (CPU, ~545 Mo, meilleur rappel que `sm`) + `phonenumbers`
(validation téléphone indépendante du regex). Ce n'est PAS branché sur le
pipeline d'ingestion live (`pipeline/ingest/normalize.py`) — script autonome
à lancer en post-traitement sur du texte déjà exporté/caché.

Ce corpus étant destiné à devenir des DONNÉES PUBLIQUES GOUVERNEMENTALES,
la priorité est le rappel (mieux vaut sur-masquer que sous-masquer) :
  - Noms de personnes (PER) : toujours masqués.
  - Lieux (LOC — villes, pays) : PAS masqués par défaut (« Paris », « la
    France » sont du contenu thématique utile aux embeddings), SAUF quand le
    contexte immédiat indique une résidence (« j'habite à/au/en… »,
    « je vis à… », « domicilié(e) à… ») — signal fort de PII, contrairement à
    une simple mention topique du même lieu.
  - Téléphone : DEUX détecteurs indépendants en union — `phonenumbers`
    (parseur/validateur Google, robuste aux formats atypiques) + le regex
    FR/international ci-dessous (filet pour les numéros mal formés que
    `phonenumbers` rejette). Deux méthodes différentes ratent des choses
    différentes ; l'union capture strictement plus que l'une seule.
  - Adresse postale : regex structurel (numéro + type de voie, ou code postal
    + ville) — aucun NER français standard ne fait mieux sur ce motif précis.
  - Identifiants administratifs (numéro de dossier, de sécurité sociale,
    fiscal, client, d'immatriculation…) : PAS de format universel possible
    (un dossier peut avoir n'importe quelle longueur) → on ancre sur
    l'INTITULÉ que le citoyen utilise lui-même (« mon numéro de dossier
    est… ») plutôt que sur la forme du nombre, seul moyen générique de rester
    UNE MÉTHODE POUR TOUTES LES ADMINISTRATIONS.

AVERTISSEMENT — aucun pipeline automatique ne garantit une anonymisation
« complète » sur du texte libre (rappel NER jamais à 100 %, fautes de frappe,
surnoms…). Pour des données publiques, ce script est une passe de réduction du
risque, pas une preuve : prévoir un échantillonnage humain avant publication.

Installation :
    uv add --optional anonymize spacy phonenumbers
    uv run --extra anonymize python -m spacy download fr_core_news_lg

Usage :
    # texte brut, une entrée par ligne, sur stdin
    echo "Jean Dupont, 06 12 34 56 78, insta: jean.d92" | \
        uv run --extra anonymize python research/anonymize_pii.py

    # JSONL, en ne touchant qu'un champ (ex. re-scrub d'un cache existant)
    uv run --extra anonymize python research/anonymize_pii.py \
        backend/cache/granddebat/ideas.jsonl --field text --output /tmp/ideas.scrubbed.jsonl

    # regex seule (pas de spaCy, encore plus rapide, ne masque pas les noms/lieux)
    ... --no-ner
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import phonenumbers  # noqa: E402 — détecteur téléphone indépendant du regex (parseur/validateur)

from pipeline.ingest.normalize import _EMAIL, _HANDLE, _URL  # noqa: E402 — regex partagés avec le pipeline live

# --- Téléphone : filet regex (audit #3 : faux positifs dates/IDs, resserré vs.
# l'original) EN PLUS de `phonenumbers` (cf. `_mask_phones`) — deux détecteurs
# indépendants, l'union rattrape ce que l'un ou l'autre rate seul.
# FR : 0X suivi de 4 groupes de 2 chiffres, ou +33/0033 X suivi des mêmes 4 groupes.
# Séparateur (espace/point/tiret) optionnel et indépendant par groupe → tolère les
# formats mixtes ("06.12 34-56 78") comme l'absence totale de séparateur.
_PHONE_FR = re.compile(r"(?<!\d)(?:(?:\+33|0033)\s?[1-9]|0[1-9])(?:[\s.\-]?\d{2}){4}(?!\d)")
# International générique (+ suivi de 8 à 15 chiffres) : ne couvre que les numéros
# préfixés '+', pour ne pas rouvrir les faux positifs sur de longs identifiants FR.
_PHONE_INTL = re.compile(r"(?<!\d)\+(?!33)\d{1,3}[\s.\-]?(?:\d[\s.\-]?){7,13}\d(?!\d)")

# --- Adresses postales ---
_STREET_TYPES = (
    r"rue|avenue|av\.?|boulevard|bd|impasse|chemin|allée|allee|place|route|quai|"
    r"square|villa|cours|passage|sentier|voie|cité|cite|résidence|residence|"
    r"hameau|clos|lotissement|lieu-dit"
)
# Numéro + (bis/ter) + type de voie + nom (ex. "12 bis rue de la République").
# Le nom de voie est borné à 5 mots (pas un blob de 40 caractères) : sans borne,
# en l'absence de virgule après l'adresse, le regex avalait la suite de la phrase
# ("... rue de la République 75011 Paris et je pense que la réforme...").
_ADDRESS_STREET = re.compile(
    rf"(?i)\b\d{{1,4}}\s*(?:bis|ter|quater)?,?\s*(?:{_STREET_TYPES})\s+"
    r"(?:de\s+la\s+|de\s+l['’]|de\s+|du\s+|des\s+|d['’])?"
    r"(?:[A-Za-zÀ-ÖØ-öø-ÿ][\w\-'’]*\s+){0,4}[A-Za-zÀ-ÖØ-öø-ÿ0-9][\w\-'’]*"
)
# Code postal FR (5 chiffres) + ville (mot capitalisé qui suit).
_ADDRESS_POSTAL = re.compile(r"\b\d{5}\s+[A-ZÀ-Ý][\w\-'’]{1,30}\b")

# --- Identifiants administratifs annoncés par leur intitulé ---
# Aucun format universel (un "numéro de dossier" varie par administration) :
# on masque le nombre qui suit un intitulé explicite, pas sa forme. Générique
# à toute administration citant son propre identifiant en clair.
_ID_LABELS = (
    r"num[ée]ro\s+de\s+dossier|n[°o]\s*de\s+dossier|dossier\s+n[°o]|"
    r"num[ée]ro\s+de\s+s[ée]curit[ée]\s+sociale|num[ée]ro\s+de\s+s[ée]cu\b|s[ée]cu\b|"
    r"num[ée]ro\s+fiscal|num[ée]ro\s+client|num[ée]ro\s+d['’]allocataire|"
    r"num[ée]ro\s+de\s+s[ée]jour|num[ée]ro\s+d['’]identifiant|"
    r"immatriculation|plaque\s+d['’]immatriculation|"
    r"r[ée]f[ée]rence(?:\s+(?:client|dossier|allocataire))?"
)
_ID_NUMBER = re.compile(
    rf"(?i)(?:{_ID_LABELS})\s*(?:est|:|=|n[°o])?\s*((?:\d[\s\-]?){{2,20}}\d)"
)

# --- Pseudos réseaux sociaux annoncés sans '@' ("insta: jean.dupont92") ---
# Le '@mention' classique est déjà couvert par `_HANDLE` (normalize.py).
_SOCIAL_KEYWORD = re.compile(
    r"(?i)\b(?:insta(?:gram)?|snap(?:chat)?|tiktok|telegram|whats?app)\s*[:=]?\s*@?([\w][\w.]{1,29})"
)


def _looks_like_handle(token: str) -> bool:
    """Écarte les mots ordinaires (« tous », « hier ») : un pseudo contient un
    chiffre, un point ou un underscore — un mot de discours courant n'en a pas."""
    return bool(re.search(r"[\d._]", token))


def _mask_social_handles(text: str) -> str:
    def repl(m: re.Match) -> str:
        return "[reseau]" if _looks_like_handle(m.group(1)) else m.group(0)

    return _SOCIAL_KEYWORD.sub(repl, text)


def _mask_id_numbers(text: str) -> str:
    """Masque le nombre suivant un intitulé d'identifiant administratif
    (« numéro de dossier », « numéro de sécu »…) — garde l'intitulé, ne
    masque que le nombre lui-même (`m.group(1)`)."""

    def repl(m: re.Match) -> str:
        start, end = m.span(1)
        return m.group(0)[: start - m.start()] + "[numero]" + m.group(0)[end - m.start():]

    return _ID_NUMBER.sub(repl, text)


def _mask_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Remplace des intervalles (start, end, token) par leur token. Les
    intervalles qui se chevauchent sont fusionnés (le premier retenu gagne) —
    utile quand deux détecteurs indépendants matchent le même passage."""
    if not spans:
        return text
    kept: list[tuple[int, int, str]] = []
    for start, end, token in sorted(spans):
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, token))
    pieces, last = [], 0
    for start, end, token in kept:
        pieces.append(text[last:start])
        pieces.append(token)
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


def _mask_phones(text: str) -> str:
    """Union de deux détecteurs indépendants : `phonenumbers` (parseur/
    validateur Google — robuste aux formats atypiques, faux positifs rares
    car il valide un VRAI numéro) puis le regex FR/international en filet
    (attrape les numéros mal formés/tapés que `phonenumbers` rejette)."""
    spans = [(m.start, m.end, "[tel]") for m in phonenumbers.PhoneNumberMatcher(text, "FR")]
    text = _mask_spans(text, spans)
    text = _PHONE_FR.sub("[tel]", text)
    text = _PHONE_INTL.sub("[tel]", text)
    return text


def anonymize_regex(text: str) -> str:
    """Passe regex + `phonenumbers` (aucune dépendance NER) : email/URL/mention
    (partagés avec le pipeline live) + réseaux sociaux + adresses + téléphone."""
    if not text:
        return text
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[url]", text)
    text = _HANDLE.sub("[mention]", text)
    text = _mask_social_handles(text)
    text = _ADDRESS_STREET.sub("[adresse]", text)
    text = _ADDRESS_POSTAL.sub("[adresse]", text)
    text = _mask_id_numbers(text)
    text = _mask_phones(text)
    return text


_NLP = None
_NER_MODEL = "fr_core_news_lg"  # "lg" > "sm" en rappel (données publiques : le rappel prime sur la vitesse)


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load(
                _NER_MODEL,
                disable=["tagger", "parser", "lemmatizer", "attribute_ruler", "morphologizer"],
            )
        except OSError as exc:
            raise RuntimeError(
                f"modèle spaCy '{_NER_MODEL}' introuvable — installer avec :\n"
                "  uv add --optional anonymize spacy\n"
                f"  uv run --extra anonymize python -m spacy download {_NER_MODEL}"
            ) from exc
    return _NLP


# Un lieu (LOC — ville, pays, région) n'est PII que si le contexte immédiat dit
# qu'on y HABITE ; sinon c'est du contenu thématique légitime (« les transports
# à Paris », « la fiscalité en France ») qu'on ne veut pas détruire. Ancré en
# fin de fenêtre (`\s*$`) : la formule doit précéder l'entité IMMÉDIATEMENT,
# pas juste apparaître quelque part avant (sinon « j'habite à Paris mais Lyon
# est mieux » masquerait aussi Lyon).
_RESIDENCE_TRIGGERS = re.compile(
    r"(?i)(?:j'|je\s+)?(?:habite|vis|r[ée]side|loge|suis\s+(?:domicili[ée]e?|install[ée]e?))"
    r"\s*(?:à|au|aux|en|dans|d['’])?\s*$"
)
_RESIDENCE_WINDOW = 30  # caractères de contexte examinés avant l'entité


def _has_residence_context(text: str, start: int) -> bool:
    ctx = text[max(0, start - _RESIDENCE_WINDOW):start]
    return bool(_RESIDENCE_TRIGGERS.search(ctx))


def mask_entities_batch(texts: list[str], batch_size: int = 256, n_process: int = 1) -> list[str]:
    """Masque via NER spaCy : PER (noms) toujours ; LOC (lieux) seulement si le
    contexte immédiat indique une résidence (cf. `_has_residence_context`)."""
    nlp = _get_nlp()
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        spans = []
        for ent in doc.ents:
            if ent.label_ == "PER":
                spans.append((ent.start_char, ent.end_char, "[nom]"))
            elif ent.label_ == "LOC" and _has_residence_context(doc.text, ent.start_char):
                spans.append((ent.start_char, ent.end_char, "[lieu]"))
        out.append(_mask_spans(doc.text, spans))
    return out


def anonymize_batch(
    texts: list[str], use_ner: bool = True, batch_size: int = 256, n_process: int = 1
) -> list[str]:
    masked = [anonymize_regex(t) for t in texts]
    if use_ner:
        masked = mask_entities_batch(masked, batch_size=batch_size, n_process=n_process)
    return masked


def anonymize(text: str, use_ner: bool = True) -> str:
    return anonymize_batch([text], use_ner=use_ner)[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Anonymisation renforcée (regex étendu + NER spaCy) d'un texte déjà passé par strip_pii."
    )
    ap.add_argument(
        "input", nargs="?",
        help="Fichier à traiter (défaut : stdin). Texte brut = une entrée par ligne, sauf --field.",
    )
    ap.add_argument("--field", help="Champ à anonymiser dans un JSONL (une entrée JSON par ligne).")
    ap.add_argument("-o", "--output", help="Fichier de sortie (défaut : stdout).")
    ap.add_argument("--no-ner", action="store_true", help="Regex seule (pas de spaCy, ne masque pas les noms).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-process", type=int, default=1, help="Processus spaCy en parallèle (CPU-bound).")
    args = ap.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    lines = raw.splitlines()

    if args.field:
        rows = [json.loads(line) for line in lines if line.strip()]
        texts = [row.get(args.field) or "" for row in rows]
        masked = anonymize_batch(texts, use_ner=not args.no_ner, batch_size=args.batch_size, n_process=args.n_process)
        for row, m in zip(rows, masked):
            row[args.field] = m
        out_lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    else:
        masked = anonymize_batch(lines, use_ner=not args.no_ner, batch_size=args.batch_size, n_process=args.n_process)
        out_lines = masked

    output = "\n".join(out_lines) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()

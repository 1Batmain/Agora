"""classify.py — heuristique statistique open_text / closed / date / numeric / empty."""
from pipeline.collect import classify


def _profile(columns):
    """columns : dict libellé -> liste de valeurs (une 'table' en colonnes)."""
    header = list(columns)
    n = max(len(v) for v in columns.values())
    rows = [[columns[h][i] if i < len(columns[h]) else None for h in header]
            for i in range(n)]
    return {q.question: q for q in classify.profile_columns(header, lambda: iter(rows))}


def test_kinds():
    long_texts = [f"Ceci est un témoignage libre numéro {i}, assez long pour être "
                  f"considéré comme du texte ouvert par l'heuristique." for i in range(20)]
    profiled = _profile({
        "id": [str(i) for i in range(20)],
        "date": [f"2023-01-{i + 1:02d} 10:00:00" for i in range(20)],
        "date_fr": [f"{i + 1:02d}/01/2023" for i in range(20)],
        "choix": ["Oui" if i % 2 else "Non" for i in range(20)],
        "temoignage": long_texts,
        "vide": [""] * 20,
    })
    assert profiled["id"].kind == "numeric"
    assert profiled["date"].kind == "date"
    assert profiled["date_fr"].kind == "date"
    assert profiled["choix"].kind == "closed"
    assert profiled["temoignage"].kind == "open_text"
    assert profiled["vide"].kind == "empty"


def test_short_but_diverse_is_open_text():
    # avg_len modeste mais forte diversité → texte libre (règle faible).
    values = [f"réponse originale numéro {i} sur le sujet" for i in range(30)]
    assert _profile({"q": values})["q"].kind == "open_text"


def test_repeated_long_label_is_not_open_text():
    # Une question longue RÉPÉTÉE (export agrégé) n'est pas du texte libre :
    # la règle forte (longueur) exige aussi un plancher de diversité.
    values = ["Que pensez-vous de la place du Parlement dans les institutions "
              "de la Cinquième République ?"] * 200
    assert _profile({"q": values})["q"].kind == "closed"


def test_duplicated_payload_with_long_outliers_is_open_text():
    # Cas réel (colonne "Contribution" des exports agrégés) : réponses courtes
    # très dupliquées ("Marie Curie" ×500) MAIS vraies contributions longues
    # présentes → le max_len élevé + une diversité absolue suffisante signent
    # une colonne de texte libre.
    values = ["Marie Curie"] * 300
    values += [f"Réponse développée distincte numéro {i} qui argumente longuement."
               for i in range(120)]
    values += ["Une très longue contribution citoyenne. " * 30]  # ~1200 caractères
    q = _profile({"q": values})["q"]
    assert q.kind == "open_text"


def test_closed_choice_stays_closed_despite_volume():
    # Une vraie colonne fermée (choix courts, max_len petit) ne bascule pas.
    values = (["Oui"] * 200) + (["Non"] * 200) + (["Sans opinion"] * 100)
    assert _profile({"q": values})["q"].kind == "closed"


def test_too_few_answers_is_not_open_text():
    values = ["Un texte pourtant très long qui ressemble fort à du texte libre."] * 3
    assert _profile({"q": values})["q"].kind != "open_text"


def test_stats_reported():
    q = _profile({"q": ["aa", "bb", "aa", ""]})["q"]
    assert q.n_answers == 3
    assert q.n_distinct == 2
    assert q.avg_len == 2.0
    assert q.max_len == 2
    assert q.question_index == 0

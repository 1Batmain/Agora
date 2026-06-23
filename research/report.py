"""Rendu Markdown du scorecard (`eval/report.md`).

Tableau Leiden vs HDBSCAN : NMI / ARI / pureté / silhouette (moyenne ± écart-type)
+ stabilité bootstrap + coût + N. Section honnêteté (taille d'échantillon, ce qui
n'est pas couvert) conformément au Playbook §5.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import mean_std

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "eval" / "report.md"


def _fmt(mean, std, prec: int = 3) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} ± {std:.{prec}f}"


def write_report(results: dict, out_path: str | Path = DEFAULT_OUT) -> Path:
    meta = results["meta"]
    runs = results["runs"]
    per_q = results["per_question"]
    approaches = meta["approaches"]
    out = Path(out_path)

    L: list[str] = []
    L.append("# Banc d'arbitrage — Leiden vs UMAP+HDBSCAN (x-stance, eval-as-truth)")
    L.append("")
    L.append(
        "Vérité terrain : commentaires x-stance labellisés **FAVOR / AGAINST** par "
        "question politique. Pour chaque question on embed les commentaires "
        f"(`{meta['model_id']}`, CPU), on clusterise avec chaque approche, et on "
        "compare le clustering aux labels."
    )
    L.append("")
    L.append("## Conditions")
    L.append("")
    L.append(f"- **Échantillon** : {meta['sample_questions']} questions "
             f"(sur {meta['n_questions_available']} exploitables), langue `{meta['lang']}`.")
    L.append(f"- **Filtre question** : ≥ {meta['min_comments']} commentaires, "
             f"≥ {meta['min_per_class']} par classe (FAVOR et AGAINST).")
    L.append(f"- **Embeddings** : {meta['n_embeddings']} commentaires encodés, "
             f"dim e5-small, seed `{meta['seed']}`.")
    p = meta["params"]
    L.append(f"- **Leiden** : k-NN k={p['k']}, seuil cosine={p['threshold']}, "
             f"résolution={p['resolution']}.")
    L.append(f"- **HDBSCAN** : UMAP(n_neighbors={p['n_neighbors']}, "
             f"n_components={p['n_components']}) + HDBSCAN(min_cluster_size={p['min_cluster_size']}).")
    L.append(f"- **Reproductible** : seed `{meta['seed']}` (Leiden, HDBSCAN, "
             f"échantillonnage, bootstrap). Python {meta['python']}.")
    L.append("")

    # --- Scorecard agrégé ---
    L.append("## Scorecard (moyenne ± écart-type sur les questions)")
    L.append("")
    header = "| Métrique | " + " | ".join(approaches) + " |"
    sep = "|" + "---|" * (len(approaches) + 1)
    L.append(header)
    L.append(sep)

    def metric_line(label, attr, prec=3, lower_is_better=False):
        cells = []
        for name in approaches:
            vals = getattr(runs[name], attr)
            m, s = mean_std(vals)
            n_valid = sum(1 for v in vals if v is not None)
            txt = _fmt(m, s, prec)
            if n_valid != len(vals):
                txt += f" ({n_valid}/{len(vals)})"
            cells.append(txt)
        return f"| {label} | " + " | ".join(cells) + " |"

    L.append(metric_line("NMI ↑ (vs labels)", "nmi"))
    L.append(metric_line("ARI ↑ (vs labels)", "ari"))
    L.append(metric_line("Pureté ↑", "purity"))
    L.append(metric_line("Silhouette ↑ (interne)", "silhouette"))
    L.append(metric_line("Stabilité ↑ (ARI bootstrap)", "stability"))
    L.append(metric_line("Nb clusters", "n_clusters", prec=1))
    L.append(metric_line("Nb bruit (-1)", "n_noise", prec=1))
    L.append(metric_line("Latence clustering (s)", "cluster_seconds", prec=3))
    L.append("")
    L.append("> ↑ = plus haut est meilleur. NMI/ARI/pureté mesurent l'accord avec la "
             "vérité FAVOR/AGAINST (2 classes) ; ARI=0 ≈ hasard. La **silhouette** est "
             "interne (séparation dans l'espace d'embedding), indépendante des labels. "
             "La **pureté** monte mécaniquement avec le nb de clusters — à lire avec la "
             "ligne « Nb clusters ».")
    L.append("")

    # --- Coût ---
    L.append("## Coût")
    L.append("")
    L.append(f"- Embeddings : **{meta['n_embeddings']}** vecteurs en "
             f"**{meta['embed_seconds']} s** (partagés par les deux approches).")
    for name in approaches:
        total = sum(runs[name].cluster_seconds)
        L.append(f"- {name} : clustering total **{total:.3f} s** "
                 f"sur {len(runs[name].cluster_seconds)} questions.")
    if meta["bootstrap"]:
        L.append(f"- Stabilité : {meta['bootstrap']} ré-échantillons "
                 f"(fraction {meta['bootstrap_frac']}) par question et par approche.")
    L.append(f"- Wall-clock total : **{meta['wall_seconds']} s**.")
    L.append("")

    # --- Détail par question ---
    L.append("## Détail par question")
    L.append("")
    cols = ["qid", "N", "FAV/AGN"]
    for name in approaches:
        cols += [f"{name} NMI", f"{name} ARI", f"{name} #cl"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for r in per_q:
        cells = [str(r["question_id"]), str(r["n"]), f"{r['n_favor']}/{r['n_against']}"]
        for name in approaches:
            m = r[name]
            cells += [f"{m['nmi']:.2f}", f"{m['ari']:.2f}", str(m["n_clusters"])]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("Questions (libellés) :")
    for r in per_q:
        q = r["question"]
        q = q if len(q) <= 110 else q[:107] + "…"
        L.append(f"- `{r['question_id']}` — {q}")
    L.append("")

    # --- Honnêteté ---
    L.append("## Honnêteté (Playbook §5) — ce qui n'est PAS couvert")
    L.append("")
    L.append(f"- **Échelle** : {meta['sample_questions']} questions, "
             f"{meta['n_embeddings']} commentaires. Échantillon modeste, à élargir "
             "(`--sample-questions`) pour resserrer les écarts-types.")
    L.append("- **Vérité terrain à 2 classes** : x-stance n'a que FAVOR/AGAINST. NMI/ARI "
             "pénalisent un clustering qui trouve > 2 groupes même sémantiquement valides "
             "(sous-thèmes d'argumentation). La silhouette nuance ce biais.")
    L.append("- **Domaine** : x-stance = votations suisses (FR). Transfert vers la "
             "consultation TikTok (témoignages libres, pas de labels) NON validé ici — "
             "c'est précisément pourquoi on n'a pas de vérité terrain sur TikTok.")
    L.append("- **Params figés** : un seul jeu de paramètres par approche (défauts "
             "pipeline). Pas de sweep d'hyperparamètres ici.")
    L.append("- **Bruit HDBSCAN** compté comme un cluster pour NMI/ARI/pureté (honnête) "
             "et exclu de la silhouette.")
    if meta["bootstrap"] == 0:
        L.append("- **Stabilité non mesurée** (`--no-bootstrap`).")
    L.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out

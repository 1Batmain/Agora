"""Palette catégorielle (style base viz `dummy`) — couleur = cluster_id Leiden."""

# Palette qualitative lisible sur fond sombre (essaim de nœuds).
PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
    "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
NOISE_COLOR = "#555555"  # cluster_id == -1 (bruit HDBSCAN)


def color_for(cluster_id: int) -> str:
    """Couleur stable pour un cluster_id (−1 = bruit)."""
    if cluster_id < 0:
        return NOISE_COLOR
    return PALETTE[cluster_id % len(PALETTE)]

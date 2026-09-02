"""
Step 6: Publication-quality co-occurrence network visualization.

Loads the edge list, builds a NetworkX graph from the top N edges,
and renders a styled network plot.

Usage:
    python src/06_network_viz.py --dataset muscle_atrophy
    python src/06_network_viz.py --dataset neuropathy --top-edges 300
"""

import argparse

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from config import DATASETS, get_paths


def main():
    parser = argparse.ArgumentParser(description="Network visualization")
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument(
        "--top-edges", type=int, default=200,
        help="Number of top edges to include (default: 200)"
    )
    args = parser.parse_args()

    paths = get_paths(args.dataset)
    info = DATASETS[args.dataset]
    prefix = info["prefix"]
    paths["figures"].mkdir(parents=True, exist_ok=True)

    # ── Load edge list ────────────────────────────────────────────────────
    edge_path = paths["processed"] / f"{prefix}_edge_list.csv"
    print(f"[6] Loading: {edge_path.name}")
    edge_df = pd.read_csv(edge_path)
    edge_df = edge_df.nlargest(args.top_edges, "Count")

    # ── Build graph ───────────────────────────────────────────────────────
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["Count"])

    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

    # Node sizing by degree
    node_sizes = [G.degree(n) * 300 for n in G.nodes()]

    # Edge widths normalized by weight
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [w / max_w * 10 for w in weights]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 18), dpi=300)
    fig.patch.set_facecolor("white")

    nx.draw_networkx_edges(
        G, pos, width=edge_widths, alpha=0.3,
        edge_color="#555555", ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color="#1D9E75", alpha=0.75,
        edgecolors="#0F6E56", linewidths=2, ax=ax
    )

    # Label top 50% nodes by degree
    degrees = sorted([G.degree(n) for n in G.nodes()], reverse=True)
    threshold = degrees[int(len(G) * 0.5)]
    labels = {n: n for n in G.nodes() if G.degree(n) >= threshold}

    nx.draw_networkx_labels(
        G, pos, labels, font_size=14,
        font_weight="bold", font_family="sans-serif", ax=ax
    )

    ax.set_title(
        f"Word Co-occurrence Network — {info['description']}",
        fontsize=30, fontweight="bold", pad=30
    )
    ax.axis("off")
    plt.tight_layout()

    out_path = paths["figures"] / "network_publication.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    ✓ Saved: {out_path.name}")
    print(f"    Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


if __name__ == "__main__":
    main()

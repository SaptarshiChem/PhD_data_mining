"""
Step 8: Filter clustering results to keep specific clusters.

Exports a subset of documents belonging to user-specified clusters
as a final curated document list.

Usage:
    python src/08_cluster_filter.py --dataset neuropathy --clusters 2 7 8 9
"""

import argparse

import pandas as pd

from config import DATASETS, get_paths


def main():
    parser = argparse.ArgumentParser(
        description="Filter documents by cluster ID"
    )
    parser.add_argument(
        "--dataset", required=True, choices=DATASETS.keys()
    )
    parser.add_argument(
        "--clusters", nargs="+", type=int, required=True,
        help="Cluster IDs to keep (e.g., --clusters 2 7 8 9)"
    )
    args = parser.parse_args()

    paths = get_paths(args.dataset)

    # ── Load ──────────────────────────────────────────────────────────────
    results_path = paths["results"] / "clustering_results.csv"
    print(f"[8] Loading: {results_path}")
    df = pd.read_csv(results_path)

    # ── Filter ────────────────────────────────────────────────────────────
    filtered = df[df["cluster"].isin(args.clusters)]
    print(f"    Kept clusters {args.clusters}: "
          f"{len(filtered)} / {len(df)} documents")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = paths["processed"] / "final_document_list.csv"
    filtered.to_csv(out_path, index=False)
    print(f"    ✓ Saved: {out_path}")


if __name__ == "__main__":
    main()

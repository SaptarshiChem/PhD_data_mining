"""
Step 5: Build a word co-occurrence network.

Computes word co-occurrence pairs WITHIN each document, producing
a meaningful edge list for network analysis.

Usage:
    python src/05_cooccurrence.py --dataset muscle_atrophy
"""

import itertools
from collections import Counter

import pandas as pd

from config import DATASETS, get_paths, parse_dataset_arg


def main():
    dataset = parse_dataset_arg("Co-occurrence network construction")
    paths = get_paths(dataset)
    prefix = DATASETS[dataset]["prefix"]

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"[5] Loading: {prefix}_noStop.pkl")
    df = pd.read_pickle(paths["processed"] / f"{prefix}_noStop.pkl")
    documents = df["no_stopwords"].dropna()

    # Tokenize: unique sorted tokens per document
    tokens_per_doc = documents.apply(lambda x: sorted(set(x.split())))

    # ── Per-document co-occurrence pairs ──────────────────────────────────
    print("    Computing per-document co-occurrence pairs...")
    pair_counter = Counter()
    for doc_tokens in tokens_per_doc:
        for pair in itertools.combinations(doc_tokens, 2):
            pair_counter[pair] += 1

    # ── Build edge list ───────────────────────────────────────────────────
    edge_df = pd.DataFrame(
        [(w1, w2, count) for (w1, w2), count in pair_counter.most_common()],
        columns=["source", "target", "Count"],
    )

    out_path = paths["processed"] / f"{prefix}_edge_list.csv"
    edge_df.to_csv(out_path, index=False)
    print(f"    ✓ Saved: {out_path.name} ({len(edge_df):,} edges)")
    print("    Top 10 co-occurring pairs:")
    print(edge_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

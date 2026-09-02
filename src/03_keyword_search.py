"""
Step 3: Boolean keyword search over the literature corpus.

Supports AND (must), OR (any_of), and NOT (exclude) operators
for flexible querying of the cleaned text corpus.

Usage:
    python src/03_keyword_search.py --dataset muscle_atrophy

Modify the example searches at the bottom of main() for your needs.
"""

import pandas as pd

from config import DATASETS, get_paths, parse_dataset_arg


def load_corpus(dataset: str) -> pd.DataFrame:
    """Load the cleaned text corpus for the given dataset."""
    paths = get_paths(dataset)
    prefix = DATASETS[dataset]["prefix"]
    return pd.read_pickle(paths["processed"] / f"{prefix}_cleaned.pkl")


def search(df, must=None, any_of=None, exclude=None):
    """
    Boolean keyword search.

    Parameters
    ----------
    df : DataFrame with a 'text' column.
    must : list[str] — all terms must appear (AND).
    any_of : list[str] — at least one must appear (OR).
    exclude : list[str] — none of these may appear (NOT).

    Returns
    -------
    DataFrame with columns: year, title, journal.
    """
    mask = pd.Series(True, index=df.index)

    if must:
        for word in must:
            mask &= df["text"].str.contains(word, case=False, na=False)

    if any_of:
        or_mask = pd.Series(False, index=df.index)
        for word in any_of:
            or_mask |= df["text"].str.contains(word, case=False, na=False)
        mask &= or_mask

    if exclude:
        for word in exclude:
            mask &= ~df["text"].str.contains(word, case=False, na=False)

    results = df[mask]
    print(f"Found {len(results)} papers.")
    return results[["year", "title", "journal"]]


def main():
    dataset = parse_dataset_arg("Keyword search")
    df = load_corpus(dataset)

    # ── Example searches (modify for your needs) ─────────────────────────
    print("\n--- AND: 'microgravity' AND 'muscle' ---")
    print(search(df, must=["microgravity", "muscle"]).head(10))

    print("\n--- OR: 'autophagy' OR 'apoptosis' ---")
    print(search(df, any_of=["autophagy", "apoptosis"]).head(10))


if __name__ == "__main__":
    main()

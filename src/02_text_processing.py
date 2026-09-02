"""
Step 2: Text preprocessing — clean and remove stopwords.

Combines title + abstract into a single text field, lowercases,
removes punctuation and extra whitespace, then removes English
and domain-specific stopwords.

Usage:
    python src/02_text_processing.py --dataset muscle_atrophy
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords

from config import DATASETS, CUSTOM_STOPWORDS, get_paths, parse_dataset_arg

nltk.download("stopwords", quiet=True)


def main():
    dataset = parse_dataset_arg("Text preprocessing")
    paths = get_paths(dataset)
    prefix = DATASETS[dataset]["prefix"]

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"[2] Loading: {prefix}_trimmed.pkl")
    df = pd.read_pickle(paths["processed"] / f"{prefix}_trimmed.pkl")

    # ── Combine title + abstract ──────────────────────────────────────────
    df["text"] = df["title"].fillna("") + " " + df["abstract"].fillna("")
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    df["text"] = df["text"].str.replace(r"[^\w\s]", "", regex=True)

    # Save cleaned (before stopword removal)
    df.to_pickle(paths["processed"] / f"{prefix}_cleaned.pkl")
    df.to_csv(paths["processed"] / f"{prefix}_cleaned.csv", index=False)
    print(f"    ✓ Saved: {prefix}_cleaned.csv / .pkl")

    # ── Remove stopwords ──────────────────────────────────────────────────
    all_stopwords = set(stopwords.words("english")).union(CUSTOM_STOPWORDS)

    df["no_stopwords"] = df["text"].apply(
        lambda t: " ".join(w for w in t.split() if w not in all_stopwords)
    )

    df.to_pickle(paths["processed"] / f"{prefix}_noStop.pkl")
    df.to_csv(paths["processed"] / f"{prefix}_noStop.csv", index=False)
    print(f"    ✓ Saved: {prefix}_noStop.csv / .pkl ({len(df)} rows)")


if __name__ == "__main__":
    main()

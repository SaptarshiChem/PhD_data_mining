"""
Step 1: Load RIS bibliography file and create a clean DataFrame.

Reads the raw RIS file, selects key metadata columns, deduplicates
by DOI, plots publications per year, and saves the trimmed dataset.

Usage:
    python src/01_load_ris.py --dataset muscle_atrophy
"""

import pandas as pd
import rispy
import matplotlib.pyplot as plt

from config import DATASETS, get_paths, parse_dataset_arg


def main():
    dataset = parse_dataset_arg("Load and clean RIS bibliography data")
    paths = get_paths(dataset)
    info = DATASETS[dataset]

    paths["processed"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)

    # ── Load RIS ──────────────────────────────────────────────────────────
    ris_path = paths["raw"] / info["ris_file"]
    print(f"[1] Loading: {ris_path.name}")

    with open(ris_path, "r", encoding="utf-8") as f:
        entries = rispy.load(f)
    print(f"    ✓ Loaded {len(entries)} entries")

    # ── Select & rename columns ───────────────────────────────────────────
    df = pd.DataFrame(entries)
    df = df[["title", "abstract", "authors", "keywords",
             "year", "secondary_title", "doi"]]
    df = df.rename(columns={"secondary_title": "journal"})

    # ── Deduplicate by DOI ────────────────────────────────────────────────
    df["doi_clean"] = df["doi"].str.lower().str.strip()
    n_missing = df["doi_clean"].isna().sum()
    print(f"    Rows with missing DOI: {n_missing}")

    before = len(df)
    df = df.drop_duplicates(subset=["doi_clean"], keep="first")
    df = df.drop(columns=["doi_clean"])
    print(f"    ✓ Deduplicated: {before} → {len(df)} entries")

    # ── Plot publications per year ────────────────────────────────────────
    year_counts = df["year"].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    plt.plot(year_counts.index, year_counts.values, marker="o")
    plt.xlabel("Publication Year")
    plt.ylabel("Number of Papers")
    plt.title(f"Publications per Year — {info['description']}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "publications_per_year.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("    ✓ Saved: publications_per_year.png")

    # ── Save ──────────────────────────────────────────────────────────────
    prefix = info["prefix"]
    df.to_csv(paths["processed"] / f"{prefix}_trimmed.csv", index=False)
    df.to_pickle(paths["processed"] / f"{prefix}_trimmed.pkl")
    print(f"    ✓ Saved: {prefix}_trimmed.csv / .pkl ({len(df)} rows)")


if __name__ == "__main__":
    main()

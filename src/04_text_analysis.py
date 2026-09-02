"""
Step 4: Exploratory text analysis.

Generates word frequencies, bigrams, trigrams, TF-IDF scores,
and a word cloud from the stopword-filtered corpus.

Usage:
    python src/04_text_analysis.py --dataset muscle_atrophy
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

from config import DATASETS, get_paths, parse_dataset_arg


def compute_ngrams(documents, n, label):
    """Compute n-gram frequencies and return as a sorted DataFrame."""
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(documents)
    df = pd.DataFrame({
        label: vectorizer.get_feature_names_out(),
        "Frequency": X.sum(axis=0).A1,
    })
    return df.sort_values("Frequency", ascending=False)


def main():
    dataset = parse_dataset_arg("Exploratory text analysis")
    paths = get_paths(dataset)
    prefix = DATASETS[dataset]["prefix"]

    paths["results"].mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"[4] Loading: {prefix}_noStop.pkl")
    df = pd.read_pickle(paths["processed"] / f"{prefix}_noStop.pkl")
    df = df.dropna(subset=["no_stopwords"])
    documents = df["no_stopwords"].astype(str)

    # ── Word frequencies ──────────────────────────────────────────────────
    print("    Computing word frequencies...")
    word_freq = compute_ngrams(documents, 1, "Word")
    word_freq.to_csv(paths["results"] / "word_frequency.csv", index=False)

    top50 = word_freq.head(50)
    plt.figure(figsize=(12, 10))
    plt.barh(top50["Word"], top50["Frequency"])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(paths["results"] / "top50_words.png", dpi=300)
    plt.close()
    print("    ✓ word_frequency.csv, top50_words.png")

    # ── Word cloud ────────────────────────────────────────────────────────
    print("    Generating word cloud...")
    cloud = WordCloud(width=1800, height=1000, background_color="white")
    cloud.generate(" ".join(documents))
    cloud.to_file(paths["results"] / "wordcloud.png")
    print("    ✓ wordcloud.png")

    # ── Bigrams ───────────────────────────────────────────────────────────
    print("    Computing bigrams...")
    bigram_df = compute_ngrams(documents, 2, "Bigram")
    bigram_df.to_csv(paths["results"] / "bigram_frequency.csv", index=False)
    print("    ✓ bigram_frequency.csv")

    # ── Trigrams ──────────────────────────────────────────────────────────
    print("    Computing trigrams...")
    trigram_df = compute_ngrams(documents, 3, "Trigram")
    trigram_df.to_csv(paths["results"] / "trigram_frequency.csv", index=False)
    print("    ✓ trigram_frequency.csv")

    # ── TF-IDF ────────────────────────────────────────────────────────────
    print("    Computing TF-IDF scores...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(documents)
    tfidf_df = pd.DataFrame({
        "Word": tfidf.get_feature_names_out(),
        "TFIDF": X.mean(axis=0).A1,
    }).sort_values("TFIDF", ascending=False)
    tfidf_df.to_csv(paths["results"] / "tfidf_scores.csv", index=False)
    print("    ✓ tfidf_scores.csv")

    print(f"\n    Done — results in: {paths['results']}")


if __name__ == "__main__":
    main()

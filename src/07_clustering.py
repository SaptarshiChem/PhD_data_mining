"""
Step 7: Document-level K-means clustering for subtopic discovery.

Builds a TF-IDF matrix, finds optimal k via elbow + silhouette
analysis, fits K-means, and generates PCA visualizations.

Usage:
    python src/07_clustering.py --dataset muscle_atrophy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

from config import DATASETS, get_paths, parse_dataset_arg

warnings.filterwarnings("ignore")


def main():
    dataset = parse_dataset_arg("Document clustering")
    paths = get_paths(dataset)
    info = DATASETS[dataset]
    prefix = info["prefix"]

    paths["figures"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    print("=" * 70)
    print(f"  CLUSTERING — {info['description']}")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = pd.read_csv(paths["processed"] / f"{prefix}_noStop.csv")
    texts = df["no_stopwords"].fillna("")
    print(f"      ✓ {len(df)} documents")

    # ── 2. TF-IDF ────────────────────────────────────────────────────────
    print("\n[2/6] Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    print(f"      ✓ Shape: {tfidf_matrix.shape}")

    # ── 3. Elbow + Silhouette ─────────────────────────────────────────────
    print("\n[3/6] Finding optimal k (2–10)...")
    k_range = range(2, 11)
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(tfidf_matrix)
        inertias.append(km.inertia_)
        sil = silhouette_score(tfidf_matrix, km.labels_)
        sil_scores.append(sil)
        print(f"      k={k}: Inertia={km.inertia_:.2f}, Silhouette={sil:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, sil_scores, "go-", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "01_elbow_analysis.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("      ✓ Saved: 01_elbow_analysis.png")

    # ── 4. Fit final model ────────────────────────────────────────────────
    optimal_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\n      ► Optimal k = {optimal_k} "
          f"(silhouette = {max(sil_scores):.3f})")

    print(f"\n[4/6] Fitting K-means with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"      Cluster {u}: {c} docs ({100 * c / len(df):.1f}%)")

    # ── 5. Top terms per cluster ──────────────────────────────────────────
    print(f"\n[5/6] Top terms per cluster...")
    terms = np.array(tfidf.get_feature_names_out())
    top_terms = {}

    for cid in range(optimal_k):
        centroid = kmeans.cluster_centers_[cid]
        top_idx = np.argsort(centroid)[-10:][::-1]
        top_terms[cid] = list(terms[top_idx])
        print(f"      Cluster {cid}: {', '.join(top_terms[cid][:5])}")

    # ── 6. Visualizations ─────────────────────────────────────────────────
    print(f"\n[6/6] Creating visualizations...")

    # PCA scatter
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix.toarray())

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
        s=100, alpha=0.6, edgecolors="black", linewidth=0.5,
    )
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
    )
    ax.set_title(
        f"Document Clustering (k={optimal_k})", fontweight="bold"
    )
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "02_cluster_visualization.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("      ✓ Saved: 02_cluster_visualization.png")

    # Cluster size bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        range(optimal_k), counts,
        color="steelblue", edgecolor="black",
        linewidth=1.5, alpha=0.8,
    )
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Documents")
    ax.set_title(
        f"Cluster Size Distribution (k={optimal_k})", fontweight="bold"
    )
    ax.set_xticks(range(optimal_k))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "03_cluster_sizes.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("      ✓ Saved: 03_cluster_sizes.png")

    # ── Export ─────────────────────────────────────────────────────────────
    results_df = df[["title", "abstract", "year", "journal", "doi"]].copy()
    results_df["cluster"] = labels
    results_df["top_terms"] = results_df["cluster"].map(
        lambda c: "; ".join(top_terms[c][:5])
    )
    results_df.to_csv(
        paths["results"] / "clustering_results.csv", index=False
    )

    summary_df = pd.DataFrame({
        "Cluster_ID": range(optimal_k),
        "Document_Count": counts,
        "Percentage": [f"{100 * c / len(df):.1f}%" for c in counts],
        "Top_5_Terms": [
            "; ".join(top_terms[i][:5]) for i in range(optimal_k)
        ],
    })
    summary_df.to_csv(
        paths["results"] / "cluster_summary.csv", index=False
    )

    print("\n" + summary_df.to_string(index=False))
    print("\n      ✓ clustering_results.csv, cluster_summary.csv")


if __name__ == "__main__":
    main()

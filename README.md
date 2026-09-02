# PhD Data Mining — Spaceflight Bibliometric Analysis

A text mining and NLP pipeline for analyzing spaceflight-related biomedical literature from RIS bibliography exports. The pipeline processes multiple research domains (muscle atrophy, neuropathy, neuroscience) through a shared codebase.

## Project Structure

```
PhD_data_mining/
├── src/                        # All pipeline scripts
│   ├── config.py               # Dataset registry & paths
│   ├── 01_load_ris.py          # Load & clean RIS files
│   ├── 02_text_processing.py   # Text cleaning + stopwords
│   ├── 03_keyword_search.py    # Boolean search (AND/OR/NOT)
│   ├── 04_text_analysis.py     # Word freq, TF-IDF, n-grams
│   ├── 05_cooccurrence.py      # Co-occurrence network
│   ├── 06_network_viz.py       # Network visualization
│   ├── 07_clustering.py        # K-means + PCA
│   └── 08_cluster_filter.py    # Filter clusters
│
├── data/                       # Per-dataset data
│   ├── muscle_atrophy/
│   │   ├── raw/                # Original .ris files
│   │   └── processed/          # Generated CSVs & pickles
│   ├── neuropathy/
│   └── neuroscience/
│
├── outputs/                    # Per-dataset outputs
│   ├── muscle_atrophy/
│   │   ├── figures/            # Plots & visualizations
│   │   └── results/            # CSVs, word clouds, etc.
│   ├── neuropathy/
│   └── neuroscience/
│
└── files/                      # Standalone analysis outputs
```

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/PhD_data_mining.git
cd PhD_data_mining

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

Every script accepts a `--dataset` flag to specify which corpus to process:

```bash
# Available datasets: muscle_atrophy, neuropathy, neuroscience
cd src/

# Step 1: Load RIS bibliography file
python 01_load_ris.py --dataset muscle_atrophy

# Step 2: Text preprocessing
python 02_text_processing.py --dataset muscle_atrophy

# Step 3: Keyword search (interactive — edit the script for your queries)
python 03_keyword_search.py --dataset muscle_atrophy

# Step 4: Exploratory text analysis (word freq, TF-IDF, word cloud)
python 04_text_analysis.py --dataset muscle_atrophy

# Step 5: Build co-occurrence network
python 05_cooccurrence.py --dataset muscle_atrophy

# Step 6: Visualize network (optional: --top-edges 300)
python 06_network_viz.py --dataset muscle_atrophy

# Step 7: K-means clustering
python 07_clustering.py --dataset muscle_atrophy

# Step 8: Filter specific clusters (e.g., keep clusters 2, 7, 8, 9)
python 08_cluster_filter.py --dataset muscle_atrophy --clusters 2 7 8 9
```

## Adding a New Dataset

1. Add your `.ris` file to `data/<dataset_name>/raw/`
2. Register it in `src/config.py`:

```python
DATASETS = {
    # ... existing datasets ...
    "my_new_dataset": {
        "ris_file": "my_file.ris",
        "prefix": "my_prefix",
        "description": "My New Research Area",
    },
}
```

3. Run the pipeline:
```bash
python src/01_load_ris.py --dataset my_new_dataset
# ... and so on
```

## Pipeline Outputs

| Step | Outputs |
|------|---------|
| 01 | `{prefix}_trimmed.csv/pkl`, `publications_per_year.png` |
| 02 | `{prefix}_cleaned.csv/pkl`, `{prefix}_noStop.csv/pkl` |
| 03 | Console output (search results) |
| 04 | `word_frequency.csv`, `bigram/trigram_frequency.csv`, `tfidf_scores.csv`, `wordcloud.png`, `top50_words.png` |
| 05 | `{prefix}_edge_list.csv` |
| 06 | `network_publication.png` |
| 07 | `clustering_results.csv`, `cluster_summary.csv`, elbow/PCA/distribution plots |
| 08 | `final_document_list.csv` |

## Dependencies

- Python ≥ 3.9
- pandas, numpy, matplotlib, seaborn
- scikit-learn, nltk, rispy, networkx, wordcloud

See [requirements.txt](requirements.txt) for exact versions.

## License

This project is part of PhD research. Please contact the author before use.

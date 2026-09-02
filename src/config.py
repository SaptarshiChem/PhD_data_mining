"""
Configuration module for the PhD Data Mining pipeline.

Defines dataset registry, directory paths, and shared constants.
All scripts import from this module to avoid hardcoded paths.
"""

import argparse
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset Registry ─────────────────────────────────────────────────────────
# Add new datasets here — no need to modify any scripts.
DATASETS = {
    "muscle_atrophy": {
        "ris_file": "Spaceflight_MA.ris",
        "prefix": "spaceflight_muscle",
        "description": "Spaceflight & Muscle Atrophy",
    },
    "neuropathy": {
        "ris_file": "Spaceflight_np_1.ris",
        "prefix": "spaceflight_np",
        "description": "Spaceflight & Neuropathy",
    },
    "neuroscience": {
        "ris_file": "spaceflight_neur.ris",
        "prefix": "spaceflight_neur",
        "description": "Spaceflight & Neuroscience",
    },
}

# ── Custom Stopwords (domain-specific) ───────────────────────────────────────
CUSTOM_STOPWORDS = {
    "study", "studies", "result", "results", "method", "methods",
    "analysis", "data", "using", "used", "show", "shown",
    "demonstrate", "demonstrated", "significant", "significantly",
    "suggest", "suggested", "associated", "observed", "however",
    "therefore", "mice", "human", "male", "female", "albino",
    "cell", "patient", "conclude", "group", "conclusion",
    "rat", "rats",
}


def get_paths(dataset_name: str) -> dict:
    """
    Return a dict of Path objects for the given dataset.

    Keys: 'raw', 'processed', 'figures', 'results'
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASETS.keys())}"
        )
    return {
        "raw": PROJECT_ROOT / "data" / dataset_name / "raw",
        "processed": PROJECT_ROOT / "data" / dataset_name / "processed",
        "figures": PROJECT_ROOT / "outputs" / dataset_name / "figures",
        "results": PROJECT_ROOT / "outputs" / dataset_name / "results",
    }


def parse_dataset_arg(description: str = "") -> str:
    """Parse the --dataset CLI argument. Used by all pipeline scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
        help="Which dataset to process",
    )
    args = parser.parse_args()
    return args.dataset

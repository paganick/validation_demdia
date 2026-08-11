#!/usr/bin/env python3
"""
Human-vs-AI cosine similarity by decoding variant (Bluesky ablation), one
panel per model, grouped bars per selection method x decoding variant.

Reads hyperparameter_cosine_comparison.csv (from
compute_hyperparameter_cosine_comparison.py). Produces
hyperparameter_cosine_comparison.png (Supplementary Figure, Section 13).

Usage:
    python analysis/plot_hyperparameter_cosine_comparison.py \
        results_hyperparameters/hyperparameter_cosine_comparison.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import with_plot_style

MODEL_ORDER = ["Llama-3.1-8B", "Mistral-7B-Instruct-v0.2"]
SELECTION_ORDER = ["random", "ML-optimal", "cosine-optimal"]
VARIANT_ORDER = ["baseline", "looser"]
VARIANT_COLORS = {
    "baseline": "#8FC1E8",
    "looser": "#0B3D6B",
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot human-vs-AI cosine similarity by decoding variant (Bluesky ablation).")
    parser.add_argument("input_csv", help="Path to hyperparameter_cosine_comparison.csv")
    parser.add_argument("--output-dir", default=None, help="Where to save the figure (default: alongside input CSV)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    medians = df.groupby(["model", "decoding_variant", "selection_method"])["cosine_similarity"].median()

    with with_plot_style(False):
        fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(8 * len(MODEL_ORDER), 6.5), sharey=True)

        x = np.arange(len(SELECTION_ORDER))
        n_variants = len(VARIANT_ORDER)
        width = 0.8 / n_variants

        for ax, model in zip(axes, MODEL_ORDER):
            for vi, variant in enumerate(VARIANT_ORDER):
                heights = []
                for sel in SELECTION_ORDER:
                    key = (model, variant, sel)
                    heights.append(medians.get(key, np.nan))
                offsets = x + (vi - (n_variants - 1) / 2) * width
                ax.bar(offsets, heights, width=width, color=VARIANT_COLORS[variant],
                       label=variant, zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels(SELECTION_ORDER, fontsize=13)
            ax.set_title(model, fontsize=16, pad=10)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            if ax is axes[0]:
                ax.set_ylabel("Cosine similarity to human ground truth (median)", fontsize=14)

        handles = [plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[v]) for v in VARIANT_ORDER]
        fig.legend(handles, VARIANT_ORDER, loc="lower center", bbox_to_anchor=(0.5, -0.08),
                   ncol=len(VARIANT_ORDER), fontsize=12, frameon=True,
                   title="Decoding variant", title_fontsize=13)

        plt.suptitle("Human-vs-AI Cosine Similarity by Decoding Variant (Bluesky)", fontsize=16, y=1.03)
        plt.tight_layout()
        output_path = output_dir / "hyperparameter_cosine_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

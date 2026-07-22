#!/usr/bin/env python3
"""
Alternative to the stacked-bar Empath feature frequency plot: a features x models
binary heatmap, one panel per platform.

The underlying data (empath_stats.csv, produced by generate_SOTA_main_figures.py's
prepare_empath_feature_stats() or generate_config_main_figures.py's
prepare_config_empath_feature_stats()) is a presence/absence matrix -- for each
(model, platform), a feature was or was not statistically significant for that
model's one best/reference configuration. The stacked bar dressed this up as a
"share of models" percentage, which becomes hard to read once every model is a
similar shade of gray (9 same-size segments, in an arbitrary stacking order,
with no visual anchor other than color). A heatmap shows the same information
-- which specific models flag which features -- directly via row/column
position instead of via reading segment order in a bar, and needs no
per-model color at all: a single fill vs. empty is enough.

Usage:
    python analysis/plot_empath_feature_heatmap.py results_PNAS_revision/figures/main_figures/SOTA_empath_stats.csv --top-n 20
    python analysis/plot_empath_feature_heatmap.py results_PNAS_revision/figures/main_figures/config_empath_stats_random.csv --top-n 20
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import (
    with_plot_style, format_dataset_name, get_dataset_color, get_ordered_models,
    MODEL_PALETTE,
)

PLATFORM_ORDER = ["bluesky", "twitter", "reddit"]


def main():
    parser = argparse.ArgumentParser(description="Features x models significance heatmap (Empath).")
    parser.add_argument("input_csv", help="Path to *_empath_stats*.csv (dataset, feature, model, count)")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features per platform")
    parser.add_argument("--output-dir", default=None, help="Where to save the figure (default: alongside input CSV)")
    parser.add_argument("--format", default="png", choices=["png", "pdf"], help="Output format")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_platform(name):
        name = name.lower()
        if "twitter" in name or name == "x":
            return "twitter"
        return name

    df["platform_norm"] = df["dataset"].apply(normalize_platform)
    platforms = [p for p in PLATFORM_ORDER if p in df["platform_norm"].unique()]
    all_models = get_ordered_models(df["model"].unique().tolist())

    with with_plot_style(False):
        fig, axes = plt.subplots(1, len(platforms), figsize=(4.2 * len(platforms), 6.5))
        if len(platforms) == 1:
            axes = [axes]

        for ax, platform in zip(axes, platforms):
            pdata = df[df["platform_norm"] == platform]
            totals = pdata.groupby("feature")["count"].sum().sort_values(ascending=False)
            top_features = totals.head(args.top_n).index.tolist()

            matrix = np.zeros((len(top_features), len(all_models)))
            for i, feature in enumerate(top_features):
                present = set(pdata.loc[pdata["feature"] == feature, "model"])
                for j, model in enumerate(all_models):
                    matrix[i, j] = 1.0 if model in present else 0.0

            for i in range(len(top_features)):
                for j, model in enumerate(all_models):
                    if matrix[i, j]:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=MODEL_PALETTE.get(model, "#404040"),
                                                    edgecolor="white", linewidth=1.0))

            ax.set_xlim(0, len(all_models))
            ax.set_ylim(0, len(top_features))
            ax.invert_yaxis()
            ax.set_xticks(np.arange(len(all_models)) + 0.5)
            ax.set_xticklabels(all_models, rotation=45, ha="right", fontsize=8.5)
            for tick, model in zip(ax.get_xticklabels(), all_models):
                tick.set_color('black')
            ax.set_yticks(np.arange(len(top_features)) + 0.5)
            ax.set_yticklabels(top_features, fontsize=9)

            # Thin gray grid at cell boundaries (both filled and empty cells), so a
            # square can always be traced back to its row/column even when empty.
            ax.set_xticks(np.arange(len(all_models) + 1), minor=True)
            ax.set_yticks(np.arange(len(top_features) + 1), minor=True)
            ax.grid(which="minor", color="#bbbbbb", linewidth=0.5, zorder=0)
            ax.tick_params(which="minor", length=0)
            ax.set_axisbelow(True)
            # Deliberately no set_aspect("equal"): panel width should stay fixed
            # and consistent across platforms regardless of how many feature rows
            # each one has -- rows simply get taller/shorter to fill the panel.
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#cccccc")

            pcolor = get_dataset_color(platform)
            ax.set_title(format_dataset_name(platform), fontsize=13, color=pcolor, pad=8)

        plt.suptitle("Significant Empath Features by Model (filled = statistically significant)", fontsize=13, y=1.02)
        plt.tight_layout()
        output_path = output_dir / (Path(args.input_csv).stem.replace("_stats", "").replace("_random", "") + f"_heatmap.{args.format}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

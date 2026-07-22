#!/usr/bin/env python3
"""
BERT classification false-negative vs. false-positive rate, per model and
condition, one panel per platform.

  - x-axis: FN rate (an AI reply misclassified as human -- "escaped detection")
  - y-axis: FP rate (a human reply misclassified as AI -- "wrongly flagged")
  - color: model (reusing the existing MODEL_PALETTE from plotting_utils.py)
  - marker shape: condition, split into two families --
      Reference (BL+PE): filled circle
      Best config (unselected / +Cosine-optimal / +ML-optimal): square / triangle / diamond

A diagonal FN=FP line makes the asymmetry (points above the line = classifier
more prone to false-flagging real humans than missing AI) immediately visible.

Reads bert_fn_fp_rates.csv (from the R1.6 FN/FP extraction).

Usage:
    python analysis/plot_bert_fn_fp.py results_PNAS_revision/configuration_optimization_figures/bert_fn_fp_rates.csv
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import MODEL_PALETTE, format_dataset_name, get_dataset_color, get_ordered_models

PLATFORM_ORDER = ["bluesky", "twitter", "reddit"]
CONDITION_ORDER = [
    "Reference (BL+PE)",
    "Best config",
    "Best config + Cosine-optimal",
    "Best config + ML-optimal",
]
CONDITION_MARKERS = {
    "Reference (BL+PE)":            "o",
    "Best config":                  "s",
    "Best config + Cosine-optimal": "^",
    "Best config + ML-optimal":     "D",
}
MODEL_ORDER = get_ordered_models([
    "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B",
    "Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.2",
    "gemma-3-4b-it", "Qwen2.5-7B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B", "Apertus-8B-2509",
])
MODEL_DISPLAY = {
    "gemma-3-4b-it": "Gemma-3-4B",
}


def main():
    parser = argparse.ArgumentParser(description="Plot BERT FN vs FP rate per model/condition/platform.")
    parser.add_argument("input_csv", help="Path to bert_fn_fp_rates.csv")
    parser.add_argument("--output-dir", default=None, help="Where to save the figure (default: alongside input CSV)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    platforms = [p for p in PLATFORM_ORDER if p in df["platform"].unique()]

    fig, axes = plt.subplots(1, len(platforms), figsize=(6.5 * len(platforms), 6), sharex=True, sharey=True)
    if len(platforms) == 1:
        axes = [axes]

    max_val = max(df["fn_rate"].max(), df["fp_rate"].max()) * 1.08

    # Marker area scales with accuracy: min accuracy in the data -> 40pt, max -> 260pt.
    acc_min, acc_max = df["accuracy"].min(), df["accuracy"].max()

    def size_for_accuracy(acc):
        if acc_max == acc_min:
            return 120
        return 40 + (acc - acc_min) / (acc_max - acc_min) * (260 - 40)

    for ax, platform in zip(axes, platforms):
        pdata = df[df["platform"] == platform]
        pcolor = get_dataset_color(platform)

        ax.plot([0, max_val * 100], [0, max_val * 100], color="#707070", linewidth=1.3, linestyle="--", zorder=1)

        for model in MODEL_ORDER:
            mdata = pdata[pdata["model"] == model]
            if mdata.empty:
                continue
            color = MODEL_PALETTE.get(model, "#808080")
            for condition in CONDITION_ORDER:
                row = mdata[mdata["condition"] == condition]
                if row.empty:
                    continue
                marker = CONDITION_MARKERS[condition]
                ax.scatter(
                    row["fn_rate"] * 100, row["fp_rate"] * 100,
                    color=color, marker=marker, s=size_for_accuracy(row["accuracy"].iloc[0]),
                    edgecolor="white", linewidth=0.6,
                    alpha=1.0, zorder=3,
                )

        ax.set_xlim(-2, max_val * 100)
        ax.set_ylim(-2, max_val * 100)
        ax.set_title(format_dataset_name(platform), fontsize=16, color=pcolor, pad=10)
        ax.set_xlabel("FN rate (%) -- AI escaping detection", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("FP rate (%) -- real human flagged as AI", fontsize=14)

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MODEL_PALETTE[m], markeredgecolor="white",
               markersize=9, label=MODEL_DISPLAY.get(m, m))
        for m in MODEL_ORDER
    ]
    condition_handles = [
        Line2D([0], [0], marker=CONDITION_MARKERS[c], linestyle="", markerfacecolor="#555555",
               markeredgecolor="white", markersize=9, label=c)
        for c in CONDITION_ORDER
    ]

    size_values = [acc_min, (acc_min + acc_max) / 2, acc_max]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#555555", markeredgecolor="white",
               markersize=np.sqrt(size_for_accuracy(v)) * 0.9, label=f"{v * 100:.0f}%")
        for v in size_values
    ]

    legend_y = -0.20
    legend1 = fig.legend(
        handles=model_handles, loc="lower center", bbox_to_anchor=(0.2, legend_y),
        ncol=3, fontsize=12, frameon=True, title="Model", title_fontsize=13
    )
    for text, model in zip(legend1.get_texts(), MODEL_ORDER):
        text.set_color('black')
    legend2 = fig.legend(
        handles=condition_handles, loc="lower center", bbox_to_anchor=(0.6, legend_y),
        ncol=2, fontsize=12, frameon=True, title="Condition", title_fontsize=13
    )
    fig.legend(
        handles=size_handles, loc="lower center", bbox_to_anchor=(0.87, legend_y),
        ncol=1, fontsize=12, frameon=True, title="Accuracy (marker size)", title_fontsize=13
    )
    fig.add_artist(legend1)
    fig.add_artist(legend2)

    plt.suptitle(
        "BERT Classification Errors: False Negatives vs. False Positives",
        fontsize=13, y=1.05
    )
    plt.tight_layout()
    output_path = output_dir / "bert_fn_fp_scatter.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

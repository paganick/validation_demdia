#!/usr/bin/env python3
"""
Compare same-context similarity distributions across the three reviewer-response
configs explored for Concern 1 (AI homogenization):

  1. SOTA          — noft, ctx0, style0 (no retrieved history, no style exemplars)
  2. History+Style — noft, ctx1, style10 (retrieved conversation history + 10 style
                      exemplars, still zero-shot)
  3. Fine-tuned    — ft, ctx1, style10 (same conditioning, plus LoRA fine-tuning)

These represent increasing depth of personalization. For each platform, plots a
grouped bar chart (one group per distribution, one bar per config) for the three
config-dependent distributions:
  - human_vs_ai          (same user, same context — the paper's main metric)
  - ai_ai_same_ctx       (different users, same context, same model)
  - ai_intra_same_user_ctx (same user, same context, same model)

human_human_same_ctx doesn't depend on the AI config (it's purely human data), so
it's drawn as a reference line rather than a fourth bar group.

Usage:
    python analysis/plot_config_comparison.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import with_plot_style, format_dataset_name, get_dataset_color

REPO_ROOT = Path(__file__).resolve().parent.parent

# config name -> (same_ctx_sims_all.csv, cosine_baselines_all.csv)
CONFIGS = {
    "SOTA": (
        REPO_ROOT / "same_context_similarity" / "same_ctx_sims_all.csv",
        REPO_ROOT / "results_PNAS_revision" / "cosine_baselines" / "cosine_baselines_all.csv",
    ),
    "History+Style": (
        REPO_ROOT / "same_context_similarity_style_history" / "same_ctx_sims_all.csv",
        REPO_ROOT / "cosine_baselines_style_history" / "cosine_baselines_all.csv",
    ),
    "Fine-tuned": (
        REPO_ROOT / "same_context_similarity_finetuned" / "same_ctx_sims_all.csv",
        REPO_ROOT / "cosine_baselines_finetuned" / "cosine_baselines_all.csv",
    ),
}

# 3-step ordinal blue ramp (references/palette.md "Sequential hue"), light -> dark,
# used here to encode increasing personalization depth rather than unordered identity.
CONFIG_COLORS = {
    "SOTA": "#6da7ec",
    "History+Style": "#2a78d6",
    "Fine-tuned": "#184f95",
}
CONFIG_ORDER = ["SOTA", "History+Style", "Fine-tuned"]

BAR_DISTRIBUTIONS = ["human_vs_ai", "ai_ai_same_ctx", "ai_intra_same_user_ctx"]
BAR_LABELS = {
    "human_vs_ai": "Human vs AI",
    "ai_ai_same_ctx": "AI-AI\n(diff. users)",
    "ai_intra_same_user_ctx": "Intra-AI\n(same user)",
}

REFERENCE_LINE_COLOR = "#52514e"  # text-secondary from palette.md, used as a muted neutral


def load_config_data(same_ctx_path, baselines_path):
    same_ctx = pd.read_csv(same_ctx_path)

    baselines = pd.read_csv(baselines_path)
    baselines = baselines[baselines["distribution"] == "human_vs_ai"].copy()
    baselines["model"] = None

    combined = pd.concat([same_ctx, baselines], ignore_index=True)
    return combined[combined["platform"] != "bluesky"]


def main():
    dfs = {}
    for config_name, (same_ctx_path, baselines_path) in CONFIGS.items():
        if not same_ctx_path.exists() or not baselines_path.exists():
            print(f"Skipping {config_name}: missing {same_ctx_path} or {baselines_path}", file=sys.stderr)
            continue
        dfs[config_name] = load_config_data(same_ctx_path, baselines_path)

    if not dfs:
        print("No config data found.", file=sys.stderr)
        sys.exit(1)

    PLATFORM_ORDER = ["bluesky", "twitter", "reddit"]
    available = set.union(*[set(df["platform"].unique()) for df in dfs.values()])
    platforms = [p for p in PLATFORM_ORDER if p in available]

    output_dir = REPO_ROOT / "same_context_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stats table (median, IQR) per config x platform x distribution      #
    # ------------------------------------------------------------------ #
    rows = []
    for config_name, df in dfs.items():
        for (platform, dist), g in df.groupby(["platform", "distribution"]):
            rows.append({
                "config": config_name, "platform": platform, "distribution": dist,
                "count": len(g), "median": g["similarity"].median(),
                "q25": g["similarity"].quantile(0.25), "q75": g["similarity"].quantile(0.75),
            })
    stats = pd.DataFrame(rows)
    stats_path = output_dir / "config_comparison_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Stats saved to: {stats_path}")

    # ------------------------------------------------------------------ #
    # Grouped bar chart                                                    #
    # ------------------------------------------------------------------ #
    with with_plot_style(False):
        fig, axes = plt.subplots(1, len(platforms), figsize=(6.5 * len(platforms), 5.5), sharey=True)
        if len(platforms) == 1:
            axes = [axes]

        n_groups = len(BAR_DISTRIBUTIONS)
        n_bars = len(CONFIG_ORDER)
        bar_width = 0.24
        group_positions = np.arange(n_groups)

        for ax, pname in zip(axes, platforms):
            pcolor = get_dataset_color(pname)

            for i, config_name in enumerate(CONFIG_ORDER):
                if config_name not in dfs:
                    continue
                offset = (i - (n_bars - 1) / 2) * bar_width
                medians, q25s, q75s = [], [], []
                for dist in BAR_DISTRIBUTIONS:
                    sub = stats[
                        (stats["config"] == config_name) & (stats["platform"] == pname)
                        & (stats["distribution"] == dist)
                    ]
                    if len(sub) == 0:
                        medians.append(np.nan)
                        q25s.append(np.nan)
                        q75s.append(np.nan)
                    else:
                        medians.append(sub["median"].iloc[0])
                        q25s.append(sub["q25"].iloc[0])
                        q75s.append(sub["q75"].iloc[0])

                x = group_positions + offset
                err_low = np.array(medians) - np.array(q25s)
                err_high = np.array(q75s) - np.array(medians)
                ax.bar(
                    x, medians, width=bar_width * 0.92, color=CONFIG_COLORS[config_name],
                    label=config_name, zorder=3,
                )
                ax.errorbar(
                    x, medians, yerr=[err_low, err_high], fmt="none",
                    ecolor="black", elinewidth=1, capsize=2, alpha=0.6, zorder=4,
                )

            # Human-human same-context reference line (config-independent baseline)
            hh_vals = [
                df.loc[(df["platform"] == pname) & (df["distribution"] == "human_human_same_ctx"), "similarity"]
                for df in dfs.values()
            ]
            hh_vals = pd.concat([v for v in hh_vals if len(v)]) if any(len(v) for v in hh_vals) else None
            if hh_vals is not None and len(hh_vals):
                hh_median = hh_vals.median()
                ax.axhline(hh_median, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5, zorder=2)
                ax.text(
                    n_groups - 0.5 + 0.15, hh_median, "Human-Human\n(same context)",
                    color=REFERENCE_LINE_COLOR, fontsize=8.5, va="center", ha="left"
                )

            ax.set_title(format_dataset_name(pname), fontsize=13, color=pcolor, pad=10)
            ax.set_xticks(group_positions)
            ax.set_xticklabels([BAR_LABELS[d] for d in BAR_DISTRIBUTIONS], fontsize=9.5)
            ax.grid(True, axis="y", alpha=0.3, zorder=0)
            ax.tick_params(axis="y", labelsize=10)
            ax.set_xlim(-0.6, n_groups - 0.5 + 0.9)
            if ax is axes[0]:
                ax.set_ylabel("Cosine similarity (median)", fontsize=12)

        legend_handles = [
            Patch(facecolor=CONFIG_COLORS[c], label=c) for c in CONFIG_ORDER if c in dfs
        ]
        fig.legend(
            handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.14),
            ncol=len(legend_handles), fontsize=10.5, frameon=True, title="Config (increasing personalization depth)"
        )
        plt.suptitle(
            "Homogenization vs Personalization Depth: SOTA → History+Style → Fine-tuned",
            fontsize=13.5, y=1.04
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        output_path = output_dir / "config_comparison_boxplot.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot same-context cosine similarity distributions (reviewer calibration analysis):
one boxplot per platform, colored by distribution, legend below the panels.

Reads same_ctx_sims_all.csv produced by compute_same_context_similarity.py and draws
one boxplot per platform showing:
  - human_human_same_ctx    — different users' human responses, same context
  - ai_ai_same_ctx          — different users' AI responses, same context, same model
  - ai_intra_same_user_ctx  — one user's two AI candidates, same context, same model

If --baselines-csv is given (cosine_baselines_all.csv, produced by
compute_cosine_baselines.py), four more distributions are pulled in for context:
  - intra_human  — same user, two DIFFERENT-context messages (self-consistency baseline)
  - human_vs_ai  — same user & context, human ground truth vs AI response (main metric)
  - random_human — unrelated human pairs, cross-user & cross-context (similarity floor)
  - random_ai    — unrelated AI pairs, cross-context & cross-model (similarity floor)

If --intra-ai-all-entries-csv is given (ai_intra_same_user_ctx_all_entries.csv,
produced by compute_intra_ai_all_entries.py), it REPLACES ai_intra_same_user_ctx
for every platform with the version computed over every test-set entry, not just
the subset of contexts with >=2 human repliers. Unlike ai_ai_same_ctx (which
genuinely needs >=2 different users replying to the same post), the intra metric
only needs one user's own >=2 generated candidates, so it's computable everywhere
-- including Bluesky, and with far more samples on Reddit/Twitter.

Usage:
    python analysis/plot_same_context_similarity.py same_context_similarity/same_ctx_sims_all.csv
    python analysis/plot_same_context_similarity.py same_context_similarity/same_ctx_sims_all.csv \\
        --baselines-csv results_PNAS_revision/cosine_baselines/cosine_baselines_all.csv \\
        --intra-ai-all-entries-csv same_context_similarity/ai_intra_same_user_ctx_all_entries.csv
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import with_plot_style, format_dataset_name, get_dataset_color

# Left-to-right column order is defined here, grouped by comparison type. To reorder
# the boxes, reorder the distribution names within a group or reorder the groups
# themselves — everything downstream (positions, colors, labels, group separators/
# headers) is derived from this one structure.
GROUPS_BASE = [
    ("Human–Human", ["human_human_same_ctx"]),
    ("AI–AI",        ["ai_ai_same_ctx", "ai_intra_same_user_ctx"]),
]
GROUPS_WITH_BASELINES = [
    ("Random pairs", ["random_human", "random_ai"]),
    ("Human–Human", ["intra_human", "human_human_same_ctx"]),
    ("Human–AI",     ["human_vs_ai"]),
    ("AI–AI",        ["ai_ai_same_ctx", "ai_intra_same_user_ctx"]),
]

DIST_LABELS = {
    "human_human_same_ctx":   "different users\nsame context",
    "intra_human":            "same user\ndifferent context",
    "human_vs_ai":            "same user\nsame context",
    "ai_ai_same_ctx":         "different users\nsame context\nsame model",
    "ai_intra_same_user_ctx": "same user\nsame context\nsame model",
    "random_human":           "unrelated human\npairs (floor)",
    "random_ai":              "unrelated AI\npairs (floor)",
}
DIST_COLORS = {
    "human_human_same_ctx":   "#55A868",  # green
    "intra_human":            "#8172B3",  # purple
    "human_vs_ai":            "#4C72B0",  # blue
    "ai_ai_same_ctx":         "#C44E52",  # red
    "ai_intra_same_user_ctx": "#DD8452",  # orange
    "random_human":           "#B0B0B0",  # light gray
    "random_ai":              "#707070",  # dark gray
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot same-context human vs AI cosine similarity distributions."
    )
    parser.add_argument("input_csv", help="Path to same_ctx_sims_all.csv")
    parser.add_argument(
        "--baselines-csv", default=None,
        help="Path to cosine_baselines_all.csv; adds intra_human and human_vs_ai to the plot"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save the figure and stats (default: alongside input CSV)"
    )
    parser.add_argument(
        "--presentation", action="store_true",
        help="Transparent background, white text (for slides)"
    )
    parser.add_argument(
        "--include-bluesky", action="store_true",
        help="Keep Bluesky as a third panel. human_human_same_ctx and ai_ai_same_ctx "
             "genuinely need >=2 same-context users, which Bluesky doesn't have, and "
             "are left as empty boxes there rather than dropped."
    )
    parser.add_argument(
        "--intra-ai-all-entries-csv", default=None,
        help="Path to ai_intra_same_user_ctx_all_entries.csv (from "
             "compute_intra_ai_all_entries.py); replaces ai_intra_same_user_ctx for "
             "every platform with the version computed over all test-set entries, "
             "not just the same-context-restricted subset -- this is what fills in "
             "Bluesky, since that metric doesn't actually need multiple repliers."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    groups = GROUPS_BASE

    if args.baselines_csv:
        baselines = pd.read_csv(args.baselines_csv)
        baselines = baselines[
            baselines["distribution"].isin(["intra_human", "human_vs_ai", "random_human", "random_ai"])
        ].copy()
        baselines["model"] = None
        df = pd.concat([df, baselines], ignore_index=True)
        groups = GROUPS_WITH_BASELINES

    if args.intra_ai_all_entries_csv:
        df = df[df["distribution"] != "ai_intra_same_user_ctx"]
        intra_all = pd.read_csv(args.intra_ai_all_entries_csv)
        df = pd.concat([df, intra_all], ignore_index=True)

    if not args.include_bluesky:
        df = df[df["platform"] != "bluesky"]

    # Drop any distribution with no rows at all across every platform (e.g.
    # random_human/random_ai when compute_cosine_baselines.py was run with
    # --skip-random) instead of drawing an empty box everywhere; drop groups that
    # end up with no members left. A distribution missing for only SOME platforms
    # (e.g. the same-context distributions on Bluesky) is kept -- it just renders
    # as an empty box on the platform(s) lacking data.
    present = set(df["distribution"].unique())
    groups = [(gname, [d for d in dists if d in present]) for gname, dists in groups]
    groups = [(gname, dists) for gname, dists in groups if dists]

    dist_order = [d for _, dists in groups for d in dists]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    PLATFORM_ORDER = ["bluesky", "twitter", "reddit"]
    platforms = [p for p in PLATFORM_ORDER if p in df["platform"].unique()]
    x_pos = list(range(len(dist_order)))

    # Group boundaries/centers, derived from `groups`, for header labels + separators
    group_spans = []   # (group_name, first_x, last_x)
    i = 0
    for name, dists in groups:
        group_spans.append((name, i, i + len(dists) - 1))
        i += len(dists)
    separator_xs = [(group_spans[k][2] + group_spans[k + 1][1]) / 2 for k in range(len(group_spans) - 1)]

    stats_path = output_dir / "same_ctx_sims_stats.csv"
    df.groupby(["platform", "distribution"])["similarity"].describe().reset_index().to_csv(
        stats_path, index=False
    )
    print(f"Stats saved to: {stats_path}")

    with with_plot_style(args.presentation):
        width_per_platform = max(4.5, 0.85 * len(dist_order))
        fig, axes = plt.subplots(
            1, len(platforms), figsize=(width_per_platform * len(platforms), 5.5), sharey=True
        )
        if len(platforms) == 1:
            axes = [axes]

        for ax, pname in zip(axes, platforms):
            pdata = df[df["platform"] == pname]
            pcolor = get_dataset_color(pname)

            box_data, box_colors = [], []
            for dist in dist_order:
                vals = pdata.loc[pdata["distribution"] == dist, "similarity"].dropna().values
                box_data.append(vals)
                box_colors.append(DIST_COLORS[dist])

            bp = ax.boxplot(
                box_data, positions=x_pos, widths=0.55, patch_artist=True,
                showfliers=False, medianprops=dict(color="black", linewidth=1.8)
            )
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            for i, color in enumerate(box_colors):
                bp["whiskers"][i * 2].set_color(color)
                bp["whiskers"][i * 2 + 1].set_color(color)
                bp["caps"][i * 2].set_color(color)
                bp["caps"][i * 2 + 1].set_color(color)

            for sep_x in separator_xs:
                ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.9, alpha=0.5, zorder=0)
            for gname, first_x, last_x in group_spans:
                ax.text(
                    (first_x + last_x) / 2, 1.05, gname, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=10, fontweight="bold"
                )

            ax.set_title(format_dataset_name(pname), fontsize=13, color=pcolor, y=1.14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([DIST_LABELS[d] for d in dist_order], fontsize=8.5, rotation=35, ha="right")
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=10)
            if ax is axes[0]:
                ax.set_ylabel("Cosine similarity", fontsize=12)

        dist_to_group = {d: gname for gname, dists in groups for d in dists}
        legend_handles = [
            Patch(
                facecolor=DIST_COLORS[d], alpha=0.75,
                label=f"{dist_to_group[d]} ({DIST_LABELS[d].replace(chr(10), ' ')})"
            )
            for d in dist_order
        ]
        fig.legend(
            handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
            ncol=2, fontsize=9.5, frameon=True
        )
        plt.suptitle("Same-Context Cosine Similarity: Human vs AI Diversity", fontsize=13, y=1.06)
        plt.tight_layout()
        output_path = output_dir / "same_ctx_sims_boxplot.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=args.presentation)
        plt.close(fig)
        print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

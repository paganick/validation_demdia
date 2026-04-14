"""
Compute and plot the raw difference (AI mean − human mean) for key text features,
across selected configurations and models (Twitter, random reply type).

Label convention (from build_validation_data.py): 0 = AI, 1 = human.

Usage:
    python analyze_feature_differences.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
REF_DIR  = BASE_DIR / "results_2026_03_17" / "twitter"
PP_DIR   = BASE_DIR / "results_ft_hyperparam_pp" / "twitter"

CONFIGS = [
    # (model_label, col_label, base_path_stem)
    ("Llama-3.1-8B",    "No FT\n(best)",     REF_DIR / "meta-llama/Llama-3.1-8B__noft__ctx1__style10___random_validation_data"),
    ("Llama-3.1-8B",    "FT default",         REF_DIR / "meta-llama/Llama-3.1-8B__ft__ctx1__style10___random_validation_data"),
    ("Llama-3.1-8B",    "FT low rank",        PP_DIR  / "meta-llama/Llama-3.1-8B__ft__ctx1__style10__ft_lorank___random_validation_data"),
    ("Llama-3.1-8B",    "FT conserv.",        PP_DIR  / "meta-llama/Llama-3.1-8B__ft__ctx1__style10__ft_conservative___random_validation_data"),
    ("Llama-3.1-8B",    "FT high reg.",       PP_DIR  / "meta-llama/Llama-3.1-8B__ft__ctx1__style10__ft_highreg___random_validation_data"),
    ("Mistral-7B-v0.1", "No FT\n(best)",     REF_DIR / "mistralai/Mistral-7B-v0.1__noft__ctx0__style0__no_persona___random_validation_data"),
    ("Mistral-7B-v0.1", "FT default",         REF_DIR / "mistralai/Mistral-7B-v0.1__ft__ctx1__style10___random_validation_data"),
    ("Mistral-7B-v0.1", "FT low rank",        PP_DIR  / "mistralai/Mistral-7B-v0.1__ft__ctx1__style10__ft_lorank___random_validation_data"),
    ("Mistral-7B-v0.1", "FT conserv.",        PP_DIR  / "mistralai/Mistral-7B-v0.1__ft__ctx1__style10__ft_conservative___random_validation_data"),
    ("Mistral-7B-v0.1", "FT high reg.",       PP_DIR  / "mistralai/Mistral-7B-v0.1__ft__ctx1__style10__ft_highreg___random_validation_data"),
]

# Features to show (ordered by overall mean importance)
FEATURES = [
    ("uppercase_ratio",          "Uppercase ratio"),
    ("emojis_count",             "Emoji count"),
    ("avg_word_length",          "Avg. word length"),
    ("toxicity_score",           "Toxicity score"),
    ("perplexity_proxy",         "Perplexity proxy"),
    ("type_token_ratio",         "Type-token ratio"),
    ("word_count",               "Word count"),
    ("sentiment",                "Sentiment"),
    ("punctuation_count",        "Punctuation count"),
    ("sentence_length_variance", "Sentence length var."),
]
FEAT_KEYS   = [f for f, _ in FEATURES]
FEAT_LABELS = [l for _, l in FEATURES]

# ── Data loading ─────────────────────────────────────────────────────────────

def load_diff(base: Path) -> pd.Series | None:
    """Return AI_mean - human_mean for each feature."""
    lab_path  = Path(str(base) + "_labelled.csv")
    feat_path = Path(str(base) + "_features.csv")
    if not lab_path.exists() or not feat_path.exists():
        print(f"  WARNING: missing files for {base.name}")
        return None
    lab  = pd.read_csv(lab_path)
    feat = pd.read_csv(feat_path)
    df = pd.concat([lab[["labels"]], feat[FEAT_KEYS]], axis=1)
    ai    = df[df.labels == 0][FEAT_KEYS].mean()
    human = df[df.labels == 1][FEAT_KEYS].mean()
    return ai - human


# ── Printing ─────────────────────────────────────────────────────────────────

def print_table(matrix: pd.DataFrame):
    print("\n" + "=" * 80)
    print("AI − human mean feature difference (random reply type, Twitter)")
    print("Positive = AI higher than human; negative = AI lower")
    print("=" * 80)
    print(f"\n{'Feature':<30}", end="")
    for col in matrix.columns:
        print(f" {col.replace(chr(10),' '):>14}", end="")
    print()
    print("-" * 80)
    for feat_key, feat_label in FEATURES:
        print(f"{feat_label:<30}", end="")
        for col in matrix.columns:
            val = matrix.loc[feat_key, col]
            print(f" {val:>14.4f}", end="")
        print()


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_diffs(matrix: pd.DataFrame, out_path: Path):
    """
    Heatmap: rows = features, columns = configs.
    Color encodes AI − human difference, normalized per feature (so all features
    are on the same scale). Raw values are annotated in each cell.
    """
    n_feat = len(FEAT_KEYS)
    n_cols = len(matrix.columns)

    # Normalize each row by its max absolute value so colours are comparable
    abs_max = matrix.abs().max(axis=1).replace(0, 1)
    normed  = matrix.div(abs_max, axis=0)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(
        "AI − human mean feature difference (Twitter, random reply type)\n"
        "Colour normalized per feature; values show raw difference",
        fontsize=11
    )

    cmap = plt.get_cmap("RdBu_r")   # red = AI higher, blue = AI lower
    im = ax.imshow(normed.values, aspect="auto", cmap=cmap,
                   vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Normalized AI − human diff.", shrink=0.8,
                 ticks=[-1, -0.5, 0, 0.5, 1])

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix.columns.tolist(), fontsize=9)
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels(FEAT_LABELS, fontsize=9)

    # Annotate with raw values
    for i in range(n_feat):
        for j in range(n_cols):
            val  = matrix.values[i, j]
            norm = normed.values[i, j]
            color = "white" if abs(norm) > 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Separator between Llama and Mistral groups
    n_llama = sum(1 for m, _, __ in CONFIGS if m == "Llama-3.1-8B")
    ax.axvline(n_llama - 0.5, color="black", linewidth=2, zorder=5)

    # Separator between noft and ft within each group
    ax.axvline(0.5, color="gray", linewidth=1, linestyle="--", alpha=0.7, zorder=5)
    ax.axvline(n_llama + 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.7, zorder=5)

    # Model labels above column groups
    for label, x_center in [
        ("Llama-3.1-8B",    (n_llama - 1) / 2),
        ("Mistral-7B-v0.1", n_llama + (n_cols - n_llama - 1) / 2),
    ]:
        ax.text(x_center, -1.7, label, ha="center", va="center",
                fontsize=10, transform=ax.transData)

    ax.tick_params(top=False, bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ── Raw value plot ───────────────────────────────────────────────────────────

CONFIG_COLORS = {
    "No FT\n(best)": "#9C27B0",
    "FT default":    "#555555",
    "FT low rank":   "#2196F3",
    "FT conserv.":   "#4CAF50",
    "FT high reg.":  "#FF9800",
}
CONFIG_ORDER = list(CONFIG_COLORS.keys())


def load_raw(base: Path) -> dict | None:
    """Return mean per label: {"human": Series, "ai": Series}."""
    lab_path  = Path(str(base) + "_labelled.csv")
    feat_path = Path(str(base) + "_features.csv")
    if not lab_path.exists() or not feat_path.exists():
        return None
    lab  = pd.read_csv(lab_path)
    feat = pd.read_csv(feat_path)
    df = pd.concat([lab[["labels"]], feat[FEAT_KEYS]], axis=1)
    return {
        "human": df[df.labels == 1][FEAT_KEYS].mean(),
        "ai":    df[df.labels == 0][FEAT_KEYS].mean(),
    }


def plot_raw_values(out_path: Path, top_n: int = 8):
    # Load all configs
    data = {}   # (model, col_label) -> {"human": Series, "ai": Series}
    for model, col_label, base in CONFIGS:
        raw = load_raw(base)
        if raw is not None:
            data[(model, col_label)] = raw

    models    = ["Llama-3.1-8B", "Mistral-7B-v0.1"]
    features  = FEATURES[:top_n]
    n_feat    = len(features)
    n_models  = len(models)

    fig, axes = plt.subplots(n_feat, n_models, figsize=(11, 2.2 * n_feat),
                             sharex="col")
    fig.suptitle(
        "Mean feature values: human vs AI by configuration (Twitter, random reply type)",
        fontsize=11, y=1.01
    )

    x      = np.arange(len(CONFIG_ORDER))
    width  = 0.45

    for col_idx, model in enumerate(models):
        configs_for_model = [(col_label, base)
                             for m, col_label, base in CONFIGS if m == model]

        for row_idx, (feat_key, feat_label) in enumerate(features):
            ax = axes[row_idx, col_idx]

            ai_vals    = []
            human_val  = None

            for col_label, _ in configs_for_model:
                entry = data.get((model, col_label))
                if entry is None:
                    ai_vals.append(np.nan)
                else:
                    ai_vals.append(entry["ai"][feat_key])
                    if human_val is None:
                        human_val = entry["human"][feat_key]

            # AI bars
            bars = ax.bar(x, ai_vals, width=width,
                          color=[CONFIG_COLORS[c] for c, _ in configs_for_model],
                          alpha=0.82, zorder=2)

            # Human reference line
            if human_val is not None:
                ax.axhline(human_val, color="black", linewidth=1.5,
                           linestyle="--", zorder=3, label="Human")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [c for c, _ in configs_for_model],
                fontsize=7.5, rotation=15, ha="right"
            )
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=8)

            # Row label (feature name) on left panel only
            if col_idx == 0:
                ax.set_ylabel(feat_label, fontsize=9)

            # Model title on top row only
            if row_idx == 0:
                ax.set_title(model, fontsize=10)

            # Human legend on top-right panel only
            if row_idx == 0 and col_idx == n_models - 1:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    col_tuples, series_list = [], []

    for model, col_label, base in CONFIGS:
        diff = load_diff(base)
        if diff is not None:
            col_tuples.append((model, col_label))
            series_list.append(diff)

    matrix = pd.DataFrame(
        {col: s for col, s in zip(col_tuples, series_list)},
        index=FEAT_KEYS
    )
    matrix.columns = pd.MultiIndex.from_tuples(col_tuples)

    # Flat version for printing/plotting (use short col labels, duplicates OK there)
    flat_labels = [f"{m.split('-')[0]}\n{c}" for m, c in col_tuples]
    matrix_flat = matrix.copy()
    matrix_flat.columns = flat_labels

    print_table(matrix_flat)

    out_path = BASE_DIR / "results_ft_hyperparam_pp" / "ft_hyperparam_feature_diffs.png"
    plot_diffs(matrix_flat, out_path)

    out_path2 = BASE_DIR / "results_ft_hyperparam_pp" / "ft_hyperparam_feature_raw.png"
    plot_raw_values(out_path2)

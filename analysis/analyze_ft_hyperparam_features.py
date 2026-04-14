"""
Analyze random forest feature importances for fine-tuned model detectability.

Loads feature_importance_stats.csv files for both the default ft config
(from results_2026_03_17) and the hyperparameter variants (from results_ft_hyperparam_pp),
random reply type only. Plots mean importance per feature per model, with individual
variant values overlaid as dots to show consistency across hyperparameter choices.

Usage:
    python analyze_ft_hyperparam_features.py
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
PP_DIR   = BASE_DIR / "results_ft_hyperparam_pp"
REF_DIR  = BASE_DIR / "results_2026_03_17"

META_COLS = {"model", "ft", "context", "style", "persona",
             "label_source", "source_type", "auc"}

MODELS = {
    "Llama-3.1-8B":   ("meta-llama",  "Llama-3.1-8B"),
    "Mistral-7B-v0.1": ("mistralai", "Mistral-7B-v0.1"),
}

# Best noft config per model (lowest BERT accuracy on random reply type)
BEST_NOFT = {
    "Llama-3.1-8B":    "Llama-3.1-8B__noft__ctx1__style10___random_validation_data_labelled_from_labels_feature_importance_stats.csv",
    "Mistral-7B-v0.1": "Mistral-7B-v0.1__noft__ctx0__style0__no_persona___random_validation_data_labelled_from_labels_feature_importance_stats.csv",
}

FEATURE_LABELS = {
    "uppercase_ratio":           "Uppercase ratio",
    "avg_word_length":           "Avg. word length",
    "toxicity_score":            "Toxicity score",
    "perplexity_proxy":          "Perplexity proxy",
    "type_token_ratio":          "Type-token ratio",
    "word_count":                "Word count",
    "sentiment":                 "Sentiment",
    "emojis_count":              "Emoji count",
    "punctuation_count":         "Punctuation count",
    "sentence_count":            "Sentence count",
    "sentence_length_variance":  "Sentence length variance",
    "syntactic_complexity":      "Syntactic complexity",
    "has_exclamation_mark":      "Has exclamation mark",
    "links_count":               "Link count",
    "repetitive_patterns":       "Repetitive patterns",
    "has_question_mark":         "Has question mark",
    "hashtags_count":            "Hashtag count",
    "hedge_words":               "Hedge words",
    "superlatives":              "Superlatives",
    "has_quotes":                "Has quotes",
    "abstract_concrete_ratio":   "Abstract/concrete ratio",
    "transition_words":          "Transition words",
    "mentions_count":            "Mention count",
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_series(path: Path) -> pd.Series | None:
    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None
    df = pd.read_csv(path)
    row = df.iloc[0]
    return pd.Series({k: row[k] for k in df.columns if k not in META_COLS})


def load_all() -> dict:
    """
    Returns:
        data[model_name][variant] = Series of feature importances
        variant is one of: "noft", "default", "ft_lorank", "ft_conservative", "ft_highreg"
    """
    data = {m: {} for m in MODELS}

    for model_name, (vendor, slug) in MODELS.items():
        # Best noft config
        noft_path = REF_DIR / "twitter" / vendor / BEST_NOFT[model_name]
        s = load_series(noft_path)
        if s is not None:
            data[model_name]["noft"] = s

        # Default ft config
        ft_path = REF_DIR / "twitter" / vendor / \
                  f"{slug}__ft__ctx1__style10___random_validation_data_labelled_from_labels_feature_importance_stats.csv"
        s = load_series(ft_path)
        if s is not None:
            data[model_name]["default"] = s

        # Hyperparameter variants
        pattern = str(PP_DIR / "twitter" / vendor /
                      f"{slug}__ft__ctx1__style10__ft_*___random_validation_data_labelled_from_labels_feature_importance_stats.csv")
        for f in sorted(glob.glob(pattern)):
            variant = Path(f).name.split("__ft__ctx1__style10__")[1].split("___")[0]
            s = load_series(Path(f))
            if s is not None:
                data[model_name][variant] = s

    return data


# ── Printing ─────────────────────────────────────────────────────────────────

def print_table(data: dict):
    print("\n" + "=" * 65)
    print("Random forest feature importances (random reply type)")
    print("Mean across all variants | lower = less important for detection")
    print("=" * 65)
    for model_name, variants in data.items():
        if not variants:
            continue
        df = pd.DataFrame(variants).T
        mean = df.mean().sort_values(ascending=False)
        print(f"\n── {model_name} ──")
        for feat, val in mean.items():
            std = df[feat].std()
            label = FEATURE_LABELS.get(feat, feat)
            print(f"  {label:<35} {val:.4f}  (±{std:.4f})")


# ── Plotting ─────────────────────────────────────────────────────────────────

VARIANT_LABELS = {
    "noft":            "No FT\n(best)",
    "default":         "FT default",
    "ft_lorank":       "FT low rank",
    "ft_conservative": "FT conserv.",
    "ft_highreg":      "FT high reg.",
}

VARIANT_ORDER = ["noft", "default", "ft_lorank", "ft_conservative", "ft_highreg"]

def plot_feature_importance(data: dict, out_path: Path, top_n: int = 12):
    # Build a combined DataFrame: columns = (model, variant), rows = features
    cols, series_list = [], []
    for model_name in MODELS:
        for variant in VARIANT_ORDER:
            if variant in data[model_name]:
                cols.append((model_name, variant))
                series_list.append(data[model_name][variant])

    matrix = pd.DataFrame(series_list, index=pd.MultiIndex.from_tuples(cols)).T
    # Sort features by mean importance across all configs
    mean_imp = matrix.mean(axis=1).sort_values(ascending=False)
    matrix = matrix.loc[mean_imp.index[:top_n]]

    # Row labels
    row_labels = [FEATURE_LABELS.get(f, f) for f in matrix.index]

    # Column labels: two-line (model short name + variant)
    model_short = {"Llama-3.1-8B": "Llama-3.1-8B", "Mistral-7B-v0.1": "Mistral-7B"}
    col_labels = [f"{model_short[m]}\n{VARIANT_LABELS[v]}" for m, v in cols]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Random forest feature importances for fine-tuned model detectability"
        " (Twitter, random reply type)",
        fontsize=11
    )

    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Feature importance", shrink=0.8)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(top_n):
        for j in range(len(cols)):
            val = matrix.values[i, j]
            color = "white" if val > matrix.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    # Vertical separators: between models, and between noft and ft within each model
    n_llama = sum(1 for m, _ in cols if m == "Llama-3.1-8B")
    ax.axvline(n_llama - 0.5, color="white", linewidth=3)

    # Thin separator after noft within each model group
    for offset in [0, n_llama]:
        # noft is the first column in each group
        ax.axvline(offset + 0.5, color="white", linewidth=1.2, linestyle="--", alpha=0.7)

    # Model labels above columns
    for model_name, x_center in [
        ("Llama-3.1-8B",    (n_llama - 1) / 2),
        ("Mistral-7B-v0.1", n_llama + (len(cols) - n_llama - 1) / 2),
    ]:
        ax.text(x_center, -1.6, model_name, ha="center", va="center",
                fontsize=10, transform=ax.transData)

    ax.tick_params(top=False, bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all()
    print_table(data)
    out_path = PP_DIR / "ft_hyperparam_feature_importance.png"
    plot_feature_importance(data, out_path)

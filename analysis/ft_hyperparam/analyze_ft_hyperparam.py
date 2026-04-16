"""
Analyze and plot BERT classifier accuracy for the fine-tuning hyperparameter sweep.

Compares default ft config vs three new variants (ft_lorank, ft_conservative, ft_highreg)
for Llama-3.1-8B and Mistral-7B-v0.1 on Twitter. Only the "random" reply type is plotted.
Confidence bars come from the 3 repeated runs in trainer_results.json.

Lower accuracy = better (harder for classifier to detect AI-generated text).

Usage:
    python analyze_ft_hyperparam.py --sweep-dir <postprocessed_sweep_dir> --ref-dir <postprocessed_ref_dir>

    --sweep-dir   Postprocessed results from the ft hyperparameter sweep
                  (contains twitter/<vendor>/ subfolders with trainer_results.json files)
    --ref-dir     Postprocessed results from the main reference run
                  (used for the no-FT baseline and the default-FT comparison)
"""

import argparse
import json
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS = {
    "Llama-3.1-8B": {
        "vendor": "meta-llama",
        "slug":   "Llama-3.1-8B",
    },
    "Mistral-7B-v0.1": {
        "vendor": "mistralai",
        "slug":   "Mistral-7B-v0.1",
    },
}

VARIANTS = {
    "noft":            {"label": "No FT\n(best config)",          "color": "#9C27B0"},
    "default":         {"label": "FT default\n(r=16, lr=1e-4)",   "color": "#555555"},
    "ft_lorank":       {"label": "FT low rank\n(r=4, lr=1e-4)",   "color": "#2196F3"},
    "ft_conservative": {"label": "FT conserv.\n(r=8, lr=5e-5)",   "color": "#4CAF50"},
    "ft_highreg":      {"label": "FT high reg.\n(r=8, drop=0.2)", "color": "#FF9800"},
}

# Best noft config trainer_results per model
BEST_NOFT_TRAINER = {
    "Llama-3.1-8B":    "twitter/meta-llama/Llama-3.1-8B__noft__ctx1__style10___random_validation_data_trainer_results.json",
    "Mistral-7B-v0.1": "twitter/mistralai/Mistral-7B-v0.1__noft__ctx0__style0__no_persona___random_validation_data_trainer_results.json",
}

RTYPE = "random"

# ── Data loading ─────────────────────────────────────────────────────────────

def load_runs(trainer_results_path: Path) -> list[float]:
    """Load per-run accuracies from trainer_results.json."""
    try:
        runs = json.load(open(trainer_results_path))
        return [r["accuracy"] for r in runs if "accuracy" in r]
    except Exception:
        return []


def find_trainer_results(directory: Path, slug: str, variant_slug: str) -> Path | None:
    """Find trainer_results.json for a given config and reply type."""
    if variant_slug == "default":
        pattern = str(directory / f"{slug}__ft__ctx1__style10___{RTYPE}_validation_data_trainer_results.json")
    else:
        pattern = str(directory / f"{slug}__ft__ctx1__style10__{variant_slug}___{RTYPE}_validation_data_trainer_results.json")
    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None


def collect_results(pp_dir: Path, ref_dir: Path) -> dict:
    """
    Returns:
        results[model_name][variant] = {"mean": float, "std": float, "runs": list[float]}
    """
    results = {m: {} for m in MODELS}

    for model_name, info in MODELS.items():
        vendor = info["vendor"]
        slug   = info["slug"]

        for variant in VARIANTS:
            if variant == "noft":
                path = ref_dir / BEST_NOFT_TRAINER[model_name]
            elif variant == "default":
                search_dir = ref_dir / "twitter" / vendor
                path = find_trainer_results(search_dir, slug, variant)
            else:
                search_dir = pp_dir / "twitter" / vendor
                path = find_trainer_results(search_dir, slug, variant)
            if path is None:
                print(f"  WARNING: trainer_results not found for {model_name} / {variant}")
                results[model_name][variant] = {"mean": np.nan, "std": 0.0, "runs": []}
                continue

            runs = load_runs(path)
            results[model_name][variant] = {
                "mean": np.mean(runs) if runs else np.nan,
                "std":  np.std(runs)  if len(runs) > 1 else 0.0,
                "runs": runs,
            }

    return results


# ── Printing ─────────────────────────────────────────────────────────────────

def print_table(results: dict):
    print("\n" + "=" * 65)
    print("BERT Classifier Accuracy — ft hyperparameter sweep (Twitter)")
    print(f"Reply type: {RTYPE} | mean ± std over {3} runs")
    print("(Lower = better)")
    print("=" * 65)
    for model_name in MODELS:
        print(f"\n── {model_name} ──")
        print(f"{'Variant':<25} {'Mean':>7} {'Std':>7}  Runs")
        print("-" * 60)
        for variant, vinfo in VARIANTS.items():
            r = results[model_name][variant]
            label = vinfo["label"].replace("\n", " ")
            runs_str = "  ".join(f"{v:.4f}" for v in r["runs"])
            print(f"{label:<25} {r['mean']:.4f}  {r['std']:.4f}  [{runs_str}]")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: Path):
    n_variants = len(VARIANTS)
    x = np.arange(n_variants)
    bar_width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.suptitle(
        "BERT classifier accuracy — fine-tuning hyperparameter sweep (Twitter, random reply type)",
        fontsize=11
    )

    variant_names  = list(VARIANTS.keys())
    variant_colors = [VARIANTS[v]["color"] for v in variant_names]
    variant_labels = [VARIANTS[v]["label"] for v in variant_names]

    for ax, model_name in zip(axes, MODELS):
        means = [results[model_name][v]["mean"] for v in variant_names]
        stds  = [results[model_name][v]["std"]  for v in variant_names]

        # Bars
        bars = ax.bar(x, means, width=bar_width, color=variant_colors,
                      alpha=0.82, edgecolor="white", linewidth=0.8, zorder=2)

        # Error bars
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                    capsize=5, capthick=1.2, elinewidth=1.2, zorder=3)

        # Individual run dots
        for i, variant in enumerate(variant_names):
            for run_val in results[model_name][variant]["runs"]:
                ax.scatter(i, run_val, color="black", s=25, zorder=4, alpha=0.6)

        # Value labels
        for bar, mean, std in zip(bars, means, stds):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.002,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="normal")

        # Reference line at default ft accuracy for this model
        default_mean = results[model_name]["default"]["mean"]
        if not np.isnan(default_mean):
            ax.axhline(default_mean, color="red", linestyle="--", linewidth=1.2,
                       alpha=0.7, label=f"FT default ({default_mean:.3f})")
            ax.legend(fontsize=8, loc="upper right")

        # Vertical separator between noft and ft variants
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

        ax.set_title(model_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels, fontsize=8.5)
        ax.set_ylabel("BERT accuracy", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # Shared y-axis range across both panels
    all_vals_global = [
        v for model_name in MODELS
        for variant in variant_names
        for v in results[model_name][variant]["runs"] if v
    ]
    if all_vals_global:
        lo = min(all_vals_global) - 0.02
        hi = max(all_vals_global) + 0.025
        axes[0].set_ylim(max(0.0, lo), min(1.01, hi))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot BERT accuracy for ft hyperparameter sweep.")
    parser.add_argument("--sweep-dir", required=True,
                        help="Postprocessed results from the ft hyperparameter sweep "
                             "(contains twitter/<vendor>/ subfolders)")
    parser.add_argument("--ref-dir", required=True,
                        help="Postprocessed results from the main reference run "
                             "(used for no-FT baseline and default-FT comparison)")
    args = parser.parse_args()

    pp_dir  = Path(args.sweep_dir)
    ref_dir = Path(args.ref_dir)

    results = collect_results(pp_dir, ref_dir)
    print_table(results)
    out_path = pp_dir / "ft_hyperparam_bert_accuracy.png"
    plot_results(results, out_path)

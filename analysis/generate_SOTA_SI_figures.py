#!/usr/bin/env python3
"""
Reference-configuration ("SOTA": baseline + persona, no style/context/fine-tuning)
figures that appear only in the Supplementary Information:
  - Cosine similarity by model and platform
  - Raw feature bias: AI vs human distributions (top-N features per platform)
  - Feature importance by model (heatmap, averaged across platforms)
  - MDI vs. permutation feature importance comparison

For the main-text reference-configuration figures (accuracy, cosine similarity,
ML explainability heatmap, Empath significance), see generate_SOTA_main_figures.py.

Usage:
    python analysis/generate_SOTA_SI_figures.py results_cleaned/
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulation.src.plotting_utils import (
    MODEL_PALETTE, get_ordered_models, format_dataset_name, with_plot_style,
    normalize_dataset_name, get_dataset_color, parse_filename, filter_for_baseline_persona,
)
from analysis._sota_common import (
    _save_stats, _get_candidate_features, _get_top_features, _load_feature_distributions,
    HUMAN_COLOR, HUMAN_LABEL,
)

# === Configuration ===
DATASETS = ['results_cleaned_20260309_095640/bluesky']
OUTPUT_DIR = "results_revision/SOTA_figures"
PRESENTATION_MODE = False
SAVE_FORMAT = 'png'
NORMALIZED_DATASETS = []  # populated in main()


# === Cosine Similarity by Model and Platform ===
def generate_cosine_similarity_by_model():
    """
    3-row figure (one per platform). Each row shows one boxplot per LLM,
    colored by model name. A single shared x-axis label appears at the bottom.
    """
    print("\n=== Generating Figure: Cosine Similarity by Model and Platform ===")

    rows = []

    for dataset in DATASETS:
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue

        print(f"Processing {dataset}...")

        response_files = list(dataset_path.rglob("*_optimal_response.json"))
        filtered_files = [f for f in response_files
                           if '__noft__ctx0__style0__optimal_response.json' in str(f)]

        print(f"  Found {len(filtered_files)} matching files")

        for file_path in filtered_files:
            try:
                model = os.path.basename(str(file_path)).split('__')[0]

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entries = data if isinstance(data, list) else [data]

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    for key in ['all_valid_responses', 'valid_responses', 'responses']:
                        if key in entry and entry[key]:
                            first = entry[key][0]
                            if isinstance(first, dict):
                                sim = first.get("cosine_similarity") or first.get("similarity")
                                if sim is not None:
                                    rows.append({
                                        "dataset": normalize_dataset_name(dataset),
                                        "model": model,
                                        "similarity": sim,
                                    })
                            break

            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")

    if not rows:
        print("No data found for cosine similarity by model plot!")
        return

    df = pd.DataFrame(rows)
    datasets = NORMALIZED_DATASETS
    model_order = get_ordered_models(df['model'].unique())

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(
            len(datasets), 1,
            figsize=(14, 5 * len(datasets)),
            sharex=True,
        )
        if len(datasets) == 1:
            axes = [axes]

        for row_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
            plot_data = [
                df[(df['dataset'] == dataset) & (df['model'] == m)]['similarity'].values
                for m in model_order
            ]
            colors = [MODEL_PALETTE.get(m, '#888888') for m in model_order]

            bp = ax.boxplot(
                plot_data,
                positions=range(len(model_order)),
                widths=0.55,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4),
            )

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(1.0)

            for i, color in enumerate(colors):
                for j in [i * 2, i * 2 + 1]:
                    bp['whiskers'][j].set_color(color)
                    bp['caps'][j].set_color(color)

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylim([-1, 1])
            ax.set_ylabel("Cosine Similarity", fontsize=14)
            ax.tick_params(axis='y', labelsize=13)
            ax.grid(True, alpha=0.3, axis='y')

            dataset_color = get_dataset_color(dataset)
            ax.set_title(format_dataset_name(dataset), fontsize=15, loc='right',
                         color=dataset_color, pad=6)

            ax.set_xticks(range(len(model_order)))
            if row_idx == len(datasets) - 1:
                ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=13)
                for tick, model in zip(ax.get_xticklabels(), model_order):
                    tick.set_color('black')
            else:
                ax.set_xticklabels([])

        stats_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity_by_model_stats.csv')
        _save_stats(df, ['dataset', 'model'], 'similarity', stats_path)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_cosine_similarity_by_model.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=600, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Raw Feature Bias (AI vs Human) ===
def generate_feature_bias(top_n: int = 10):
    """
    3 rows (platforms) x top_n columns (features).
    Each subplot: one boxplot per model (MODEL_PALETTE) + one Human boxplot (gray).
    """
    print(f"\n=== Generating Figure: Raw Feature Bias (AI vs Human) ===")

    candidate_features, all_models = _get_candidate_features(DATASETS)

    platform_features = {}
    platform_data = {}

    for dataset in DATASETS:
        pname = normalize_dataset_name(dataset)
        top_feats = _get_top_features(Path(dataset), top_n, candidate_features, all_models)
        model_data = _load_feature_distributions(Path(dataset))
        if not top_feats or not model_data:
            print(f"  Skipping {pname}: missing data")
            continue
        print(f"  {pname}: top features = {top_feats}")
        platform_features[pname] = top_feats
        platform_data[pname] = model_data

    if not platform_data:
        print("  No data found.")
        return

    pnames = list(platform_data.keys())
    model_order = get_ordered_models({m for pd_ in platform_data.values() for m in pd_})
    x_labels = model_order + [HUMAN_LABEL]
    x_positions = list(range(len(x_labels)))
    n_rows, n_cols = len(pnames), top_n

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3.5 * n_rows), sharey=False)
        if n_rows == 1: axes = axes[np.newaxis, :]
        if n_cols == 1: axes = axes[:, np.newaxis]

        for row_idx, pname in enumerate(pnames):
            pdata = platform_data[pname]
            features = platform_features[pname]
            pcolor = get_dataset_color(pname)
            first_df = next(iter(pdata.values()), None)

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(features):
                    ax.set_visible(False)
                    continue
                feat = features[col_idx]

                box_data, box_colors = [], []
                for model in model_order:
                    df = pdata.get(model)
                    vals = df.loc[df['label'] == 0, feat].dropna().values if (df is not None and feat in df.columns) else np.array([])
                    box_data.append(vals)
                    box_colors.append(MODEL_PALETTE.get(model, '#888888'))

                human_vals = first_df.loc[first_df['label'] == 1, feat].dropna().values if (first_df is not None and feat in first_df.columns) else np.array([])
                human_median = float(np.median(human_vals)) if len(human_vals) else 0.0
                box_data.append(human_vals)
                box_colors.append(HUMAN_COLOR)

                bp = ax.boxplot(box_data, positions=x_positions, widths=0.55, patch_artist=True,
                                 showfliers=False, medianprops=dict(color='black', linewidth=1.5))
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color); patch.set_alpha(1.0)
                for i, color in enumerate(box_colors):
                    bp['whiskers'][i*2].set_color(color);   bp['whiskers'][i*2+1].set_color(color)
                    bp['caps'][i*2].set_color(color);       bp['caps'][i*2+1].set_color(color)

                rng = np.random.default_rng(42)
                max_pts = 80
                for pos, vals, color in zip(x_positions, box_data, box_colors):
                    if len(vals) == 0:
                        continue
                    sample = vals if len(vals) <= max_pts else rng.choice(vals, max_pts, replace=False)
                    jitter = rng.uniform(-0.18, 0.18, size=len(sample))
                    ax.scatter(pos + jitter, sample, s=4, color=color, alpha=0.45, zorder=3, linewidths=0)

                all_vals = np.concatenate([v for v in box_data if len(v) > 0])
                if len(all_vals) > 0:
                    lo, hi = np.percentile(all_vals, 2), np.percentile(all_vals, 98)
                    pad = (hi - lo) * 0.1 if hi > lo else 0.1
                    ax.set_ylim(lo - pad, hi + pad)

                ax.axhline(human_median, color=HUMAN_COLOR, linestyle='-', linewidth=2.5, alpha=1.0, zorder=4)
                ax.set_title(feat.replace('_', ' '), fontsize=16, pad=4)
                if col_idx == 0:
                    ax.set_ylabel(format_dataset_name(pname), fontsize=18, color=pcolor, labelpad=6)
                ax.tick_params(axis='y', labelsize=9)
                ax.tick_params(axis='x', bottom=False)
                ax.set_xticks([])
                ax.grid(True, alpha=0.3, axis='y')

        legend_handles = [mpatches.Patch(facecolor=MODEL_PALETTE.get(m, '#888888'), alpha=1.0, label=m) for m in model_order]
        legend_handles.append(mpatches.Patch(facecolor=HUMAN_COLOR, alpha=1.0, label=HUMAN_LABEL))
        legend = fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.01),
                             ncol=min(len(legend_handles), 5), fontsize=16, frameon=True)
        for text, model in zip(legend.get_texts(), model_order):
            text.set_color('black')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_feature_bias.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === Feature Importance by Model (heatmap, aggregated across platforms) ===
def generate_feature_importance_by_model():
    """
    Heatmap: rows = models, columns = features (sorted by mean importance).
    Values are averaged across all SOTA-config files across all platforms.
    Highlights per-model outliers (e.g. Qwen's emojis, Mistral-Instruct's mentions).
    """
    print(f"\n=== Generating Figure: Feature Importance by Model (heatmap) ===")

    rows = []
    for dataset in DATASETS:
        dataset_path = Path(dataset)
        files = glob.glob(str(dataset_path / '**' / '*feature_correlation_stats.csv'), recursive=True)
        files = [f for f in files if filter_for_baseline_persona(f) and 'random_validation' in f]
        for fpath in files:
            try:
                model, ft, context, style, persona = parse_filename(fpath)
                if model is None:
                    continue
                df = pd.read_csv(fpath)
                if 'feature' not in df.columns or 'importance' not in df.columns:
                    continue
                df['model'] = model
                df['platform'] = normalize_dataset_name(str(dataset))
                rows.append(df[['model', 'platform', 'feature', 'importance']])
            except Exception:
                continue

    if not rows:
        print("  No data found.")
        return

    combined = pd.concat(rows, ignore_index=True)

    pivot = (combined.groupby(['model', 'feature'])['importance']
                     .mean()
                     .unstack(fill_value=0))

    feat_order = pivot.mean(axis=0).sort_values(ascending=False).index.tolist()
    model_order = get_ordered_models(pivot.index.tolist())
    pivot = pivot.loc[model_order, feat_order]

    stats_path = os.path.join(OUTPUT_DIR, 'SOTA_feature_importance_by_model.csv')
    pivot.reset_index().to_csv(stats_path, index=False)
    print(f"  Stats saved to: {stats_path}")

    feat_labels = [f.replace('_', ' ') for f in feat_order]

    with with_plot_style(PRESENTATION_MODE):
        fig, ax = plt.subplots(figsize=(max(14, len(feat_order) * 0.7), len(model_order) * 0.7 + 1.5))

        im = ax.imshow(pivot.values, aspect='auto', cmap='Blues',
                        vmin=0, vmax=pivot.values.max())

        ax.set_xticks(range(len(feat_order)))
        ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels(model_order, fontsize=11)
        for tick, model in zip(ax.get_yticklabels(), model_order):
            tick.set_color('black')

        for r, model in enumerate(model_order):
            for c, feat in enumerate(feat_order):
                val = pivot.loc[model, feat]
                text_color = 'white' if val > pivot.values.max() * 0.65 else 'black'
                ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=text_color)

        cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
        cbar.set_label('Mean RF Importance\n(avg across platforms)', fontsize=10)

        ax.set_title('Feature Importance by Model (reference configuration, averaged across platforms)',
                     fontsize=13, pad=10)
        plt.tight_layout()

        output_path = os.path.join(OUTPUT_DIR, f'SOTA_feature_importance_by_model.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


# === MDI vs Permutation Feature Importance comparison ===
def generate_feature_importance_comparison(n_repeats: int = 20, top_n: int = 10):
    """
    For each platform, train a Random Forest on SOTA-config validation data
    and compute both Mean Decrease in Impurity (MDI) and Permutation Feature
    Importance (PFI) on the held-out test set.  Importances are averaged across
    models and shown as side-by-side horizontal bar charts (one row per platform).

    MDI is known to be biased toward high-cardinality continuous features and is
    computed on training data.  PFI measures the drop in AUC when each feature is
    shuffled on the test set, giving an unbiased out-of-sample estimate.
    """
    print(f"\n=== Generating Figure: MDI vs Permutation Feature Importance ===")

    EXCLUDE = {'spelling_grammar_errors', 'has_emoji', 'has_mention', 'has_link'}

    platform_results = {}

    for dataset in DATASETS:
        pname = normalize_dataset_name(dataset)
        model_data = _load_feature_distributions(Path(dataset))
        if not model_data:
            print(f"  Skipping {pname}: no data")
            continue

        platform_results[pname] = []

        for model, df in model_data.items():
            feat_cols = [c for c in df.columns if c not in EXCLUDE | {'label'}]
            feat_cols = [c for c in feat_cols if df[c].dtype in (np.float64, np.int64, float, int)]
            X = df[feat_cols].fillna(0)
            y = df['label']

            if y.nunique() < 2 or len(X) < 20:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, random_state=42, test_size=0.25
            )

            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)

            mdi = pd.Series(clf.feature_importances_, index=feat_cols)

            pfi_result = permutation_importance(
                clf, X_test, y_test,
                n_repeats=n_repeats,
                random_state=42,
                scoring='roc_auc',
                n_jobs=-1,
            )
            pfi = pd.Series(pfi_result.importances_mean, index=feat_cols)

            platform_results[pname].append((mdi, pfi))
            print(f"    {pname} / {model}: done")

    if not platform_results:
        print("  No data found.")
        return

    pnames = list(platform_results.keys())
    n_rows = len(pnames)

    with with_plot_style(PRESENTATION_MODE):
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for row_idx, pname in enumerate(pnames):
            pairs = platform_results[pname]
            if not pairs:
                continue

            all_mdi = pd.concat([p[0] for p in pairs], axis=1).mean(axis=1)
            all_pfi = pd.concat([p[1] for p in pairs], axis=1).mean(axis=1)

            all_mdi = all_mdi / all_mdi.sum() if all_mdi.sum() > 0 else all_mdi
            all_pfi_pos = all_pfi.clip(lower=0)
            all_pfi_pos = all_pfi_pos / all_pfi_pos.sum() if all_pfi_pos.sum() > 0 else all_pfi_pos

            top_mdi_feats = set(all_mdi.nlargest(top_n).index)
            top_pfi_feats = set(all_pfi_pos.nlargest(top_n).index)
            top_feats = list(top_mdi_feats | top_pfi_feats)
            avg_rank = (all_mdi[top_feats].rank(ascending=False) +
                        all_pfi_pos[top_feats].rank(ascending=False)) / 2
            top_feats = avg_rank.sort_values().index.tolist()[:top_n]

            pcolor = get_dataset_color(pname)

            for col_idx, (values, title) in enumerate([
                (all_mdi[top_feats].sort_values(), 'MDI (Gini impurity, train)'),
                (all_pfi[top_feats].sort_values(), 'Permutation (AUC drop, test)'),
            ]):
                ax = axes[row_idx, col_idx]
                colors = ['#d32f2f' if v < 0 else pcolor for v in values]
                ax.barh(range(len(values)), values.values, color=colors, alpha=0.8)
                ax.set_yticks(range(len(values)))
                ax.set_yticklabels([f.replace('_', ' ') for f in values.index], fontsize=10)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_title(title, fontsize=12)
                ax.grid(True, axis='x', alpha=0.3)
                if col_idx == 0:
                    ax.set_ylabel(format_dataset_name(pname), fontsize=13,
                                  color=pcolor, labelpad=6)

        plt.suptitle('Feature importance: MDI vs Permutation (reference configuration, averaged across models)',
                     fontsize=13, y=1.01)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'SOTA_feature_importance_comparison.{SAVE_FORMAT}')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=PRESENTATION_MODE)
        plt.close(fig)
        print(f"  Saved to: {output_path}")


def main():
    global DATASETS, NORMALIZED_DATASETS, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Generate SI-only reference-configuration (SOTA) figures.")
    parser.add_argument("results_folder", nargs="?", default=None,
                         help="Base results folder (e.g. results_2026_03_17). "
                              "Platform subfolders (bluesky, reddit, twitter) are detected automatically.")
    args = parser.parse_args()

    if args.results_folder is not None:
        base = Path(args.results_folder)
        EXCLUDE_DIRS = {'cosine_baselines', 'configuration_optimization_figures', 'figures'}
        PLATFORM_ORDER = ['bluesky', 'twitter', 'reddit']
        all_dirs = {p.name: str(p) for p in base.iterdir() if p.is_dir() and not p.name.endswith('_figures') and p.name not in EXCLUDE_DIRS}
        found = [all_dirs[k] for k in PLATFORM_ORDER if k in all_dirs] + \
                [v for k, v in sorted(all_dirs.items()) if k not in PLATFORM_ORDER]
        if not found:
            print(f"Error: no subdirectories found in {base}", file=sys.stderr)
            sys.exit(1)
        DATASETS = found
        NORMALIZED_DATASETS = [normalize_dataset_name(d) for d in DATASETS]
        OUTPUT_DIR = str(base / "figures" / "SI_figures")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SOTA SI-Only Figure Generation")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Datasets: {DATASETS}")
    print("Configuration filter: baseline + persona only (no ft, no style, no context)")

    generate_cosine_similarity_by_model()
    generate_feature_bias()
    generate_feature_importance_by_model()
    generate_feature_importance_comparison()

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

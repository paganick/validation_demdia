"""
Shared helpers for the reference-configuration ("SOTA") figure scripts:
generate_SOTA_main_figures.py and generate_SOTA_SI_figures.py.
"""

import os
import glob
from pathlib import Path

import pandas as pd

from simulation.src.plotting_utils import parse_filename, filter_for_baseline_persona

HUMAN_COLOR = '#888888'
HUMAN_LABEL = 'Human'


def _save_stats(df, groupby_cols, value_cols, output_path):
    """Save summary statistics (count, mean, std, median, Q1, Q3) for a figure's data."""
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    agg = df.groupby(groupby_cols)[value_cols].describe(percentiles=[.25, .5, .75])
    agg.columns = ['_'.join(c).strip('_') for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(output_path, index=False)
    print(f"  Stats saved to: {output_path}")


def _get_candidate_features(datasets) -> tuple:
    """Compute global candidate features using the same strategy as the SOTA_ML heatmap.

    Returns (candidate_features: set, all_models: list).
    candidate_features = global top-10 by mean importance ∪ top-2 per model (globally).
    """
    all_data = []
    for dataset in datasets:
        files = glob.glob(str(Path(dataset)) + '/**/*feature_correlation_stats.csv', recursive=True)
        files = [f for f in files if filter_for_baseline_persona(f)]
        for fpath in files:
            try:
                df = pd.read_csv(fpath)
                if 'feature' not in df.columns or 'importance' not in df.columns:
                    continue
                model, _, _, _, _ = parse_filename(fpath)
                if model:
                    df = df[['feature', 'importance']].copy()
                    df['model'] = model
                    all_data.append(df)
            except Exception:
                continue
    if not all_data:
        return set(), []
    combined = pd.concat(all_data, ignore_index=True)
    top_overall = set(
        combined.groupby('feature')['importance'].mean()
                .sort_values(ascending=False).head(10).index
    )
    top_per_model = set()
    for model in combined['model'].unique():
        mi = (combined[combined['model'] == model]
              .groupby('feature')['importance'].mean()
              .sort_values(ascending=False))
        top_per_model.update(mi.head(2).index)
    return top_overall.union(top_per_model), combined['model'].unique().tolist()


def _get_top_features(dataset_path: Path, top_n: int = 5,
                       candidate_features: set = None, all_models: list = None) -> list:
    """Return top-N features ranked by mean RF importance across SOTA-config models.

    When candidate_features and all_models are supplied the same intersection
    strategy used by the SOTA_ML heatmap is applied, keeping the two plots in sync.
    """
    files = glob.glob(str(dataset_path / '**' / '*feature_correlation_stats.csv'), recursive=True)
    files = [f for f in files if filter_for_baseline_persona(f)]
    dfs = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            if 'feature' not in df.columns or 'importance' not in df.columns:
                continue
            if candidate_features is not None:
                model, _, _, _, _ = parse_filename(fpath)
                if model:
                    df = df[['feature', 'importance']].copy()
                    df['model'] = model
            else:
                df = df[['feature', 'importance']]
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return []
    combined = pd.concat(dfs, ignore_index=True)
    di = combined.groupby('feature')['importance'].mean().sort_values(ascending=False)

    if candidate_features is None:
        return di.head(top_n).index.tolist()

    # Mirror heatmap logic: per-dataset top-10 ∪ per-dataset top-2-per-model,
    # intersected with globally-derived candidate_features.
    dataset_top = set(di.head(10).index)
    dataset_top_per_model = set()
    if all_models is not None and 'model' in combined.columns:
        for model in all_models:
            mi = (combined[combined['model'] == model]
                  .groupby('feature')['importance'].mean()
                  .sort_values(ascending=False))
            dataset_top_per_model.update(mi.head(2).index)
    return sorted(
        dataset_top.union(dataset_top_per_model).intersection(candidate_features),
        key=lambda x: float(di.get(x, 0)),
        reverse=True
    )[:top_n]


def _load_feature_distributions(dataset_path: Path) -> dict:
    """Load raw feature values + label (0=AI, 1=human) for each SOTA-config model."""
    label_files = glob.glob(str(dataset_path / '**' / '*_random_validation_data.csv'), recursive=True)
    label_files = [f for f in label_files if filter_for_baseline_persona(f) and '_features' not in f]
    model_data = {}
    for lpath in label_files:
        try:
            model, ft, context, style, persona = parse_filename(lpath)
            if model is None:
                continue
            feat_path = lpath.replace('_random_validation_data.csv', '_random_validation_data_features.csv')
            if not os.path.exists(feat_path):
                continue
            labels_df = pd.read_csv(lpath)
            feats_df = pd.read_csv(feat_path)
            if len(labels_df) != len(feats_df):
                continue
            feats_df['label'] = labels_df['labels'].values
            model_data[model] = feats_df
        except Exception:
            continue
    return model_data

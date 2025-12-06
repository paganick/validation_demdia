"""Consolidated output formats for postprocessing results.

This module defines consolidated output structures that replace the
44+ separate files per configuration with 4 consolidated files:
  - responses.json (already handled by data_formats.py)
  - validation_results.json
  - features.parquet
  - statistics.json
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ValidationResult:
    """Validation results for a single selection method.

    Attributes:
        selection_method: Which method was used (random, ml, cosine)
        bert_metrics: BERT classification metrics (accuracy, F1, etc.)
        confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        predictions: Optional list of individual predictions
        shap_values: Optional SHAP analysis results
    """
    selection_method: str
    bert_metrics: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    predictions: Optional[List[Dict]] = None
    shap_values: Optional[Dict] = None


@dataclass
class ConsolidatedValidation:
    """All validation results for a configuration.

    Consolidates what was previously spread across:
      - {method}_validation_data_bert_report.json
      - {method}_validation_data_confusion_matrix.csv
      - {method}_validation_data_bert_shap_analysis.json
      - {method}_validation_data_trainer_results.json
    Into a single structure.

    Attributes:
        config_name: Configuration identifier
        model: Model name
        results: Validation results per selection method
        metadata: Additional configuration metadata
    """
    config_name: str
    model: str
    results: Dict[str, ValidationResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, filepath: str):
        """Save validation results to JSON."""
        data = {
            'config_name': self.config_name,
            'model': self.model,
            'metadata': self.metadata,
            'results': {
                method: asdict(result)
                for method, result in self.results.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'ConsolidatedValidation':
        """Load validation results from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = {
            method: ValidationResult(**result_data)
            for method, result_data in data.get('results', {}).items()
        }

        return cls(
            config_name=data['config_name'],
            model=data['model'],
            metadata=data.get('metadata', {}),
            results=results
        )


@dataclass
class FeatureStatistics:
    """Feature statistics for a selection method.

    Attributes:
        selection_method: Which method was used
        correlation_stats: Feature correlation statistics
        importance_stats: Feature importance statistics
        empath_significant: Significant Empath features
        agreement_stats: Agreement statistics from labels
    """
    selection_method: str
    correlation_stats: Optional[pd.DataFrame] = None
    importance_stats: Optional[pd.DataFrame] = None
    empath_significant: Optional[pd.DataFrame] = None
    agreement_stats: Optional[pd.DataFrame] = None


@dataclass
class ConsolidatedStatistics:
    """All statistical analyses for a configuration.

    Consolidates what was previously spread across:
      - {method}_validation_data_labelled_from_labels_feature_correlation_stats.csv
      - {method}_validation_data_labelled_from_labels_feature_importance_stats.csv
      - {method}_validation_data_empath_significant_features.csv
      - {method}_validation_data_from_labels_median_run_agreement.csv
      - {method}_validation_data_from_labels_median_run_results.csv

    Attributes:
        config_name: Configuration identifier
        model: Model name
        statistics: Statistics per selection method
        metadata: Additional configuration metadata
    """
    config_name: str
    model: str
    statistics: Dict[str, FeatureStatistics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, filepath: str):
        """Save statistics to JSON."""
        data = {
            'config_name': self.config_name,
            'model': self.model,
            'metadata': self.metadata,
            'statistics': {}
        }

        for method, stats in self.statistics.items():
            data['statistics'][method] = {
                'selection_method': stats.selection_method,
                'correlation_stats': stats.correlation_stats.to_dict('records') if stats.correlation_stats is not None else None,
                'importance_stats': stats.importance_stats.to_dict('records') if stats.importance_stats is not None else None,
                'empath_significant': stats.empath_significant.to_dict('records') if stats.empath_significant is not None else None,
                'agreement_stats': stats.agreement_stats.to_dict('records') if stats.agreement_stats is not None else None,
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'ConsolidatedStatistics':
        """Load statistics from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        statistics = {}
        for method, stats_data in data.get('statistics', {}).items():
            statistics[method] = FeatureStatistics(
                selection_method=stats_data['selection_method'],
                correlation_stats=pd.DataFrame(stats_data['correlation_stats']) if stats_data.get('correlation_stats') else None,
                importance_stats=pd.DataFrame(stats_data['importance_stats']) if stats_data.get('importance_stats') else None,
                empath_significant=pd.DataFrame(stats_data['empath_significant']) if stats_data.get('empath_significant') else None,
                agreement_stats=pd.DataFrame(stats_data['agreement_stats']) if stats_data.get('agreement_stats') else None,
            )

        return cls(
            config_name=data['config_name'],
            model=data['model'],
            metadata=data.get('metadata', {}),
            statistics=statistics
        )


def save_consolidated_features(features_dict: Dict[str, pd.DataFrame], output_path: str):
    """Save features for all selection methods to a single Parquet file.

    Consolidates what was previously spread across:
      - {method}_validation_data_features.csv
      - {method}_validation_data_features_with_text_and_label.csv
      - {method}_validation_data_labelled.csv

    Args:
        features_dict: Dictionary mapping selection_method -> DataFrame with features
        output_path: Path to output Parquet file
    """
    # Combine all method dataframes with a 'selection_method' column
    combined_features = []

    for method, df in features_dict.items():
        df_copy = df.copy()
        df_copy['selection_method'] = method
        combined_features.append(df_copy)

    if combined_features:
        final_df = pd.concat(combined_features, ignore_index=True)
        final_df.to_parquet(output_path, engine='pyarrow', compression='snappy')


def load_consolidated_features(input_path: str, selection_method: Optional[str] = None) -> pd.DataFrame:
    """Load features from consolidated Parquet file.

    Args:
        input_path: Path to Parquet file
        selection_method: If specified, filter to only this method

    Returns:
        DataFrame with features (optionally filtered by method)
    """
    df = pd.read_parquet(input_path)

    if selection_method:
        df = df[df['selection_method'] == selection_method].copy()
        df = df.drop('selection_method', axis=1)

    return df


class ConsolidatedOutputWriter:
    """Writer for consolidated postprocessing outputs.

    Manages writing the 4 consolidated files per configuration:
      1. responses.json (from data_formats.py)
      2. validation_results.json
      3. features.parquet
      4. statistics.json
    """

    def __init__(self, config_dir: Path):
        """Initialize writer for a configuration directory.

        Args:
            config_dir: Directory for this configuration's outputs
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def write_validation(self, validation: ConsolidatedValidation):
        """Write validation results."""
        output_path = self.config_dir / 'validation_results.json'
        validation.save(str(output_path))
        return output_path

    def write_features(self, features_dict: Dict[str, pd.DataFrame]):
        """Write features for all methods."""
        output_path = self.config_dir / 'features.parquet'
        save_consolidated_features(features_dict, str(output_path))
        return output_path

    def write_statistics(self, statistics: ConsolidatedStatistics):
        """Write statistics."""
        output_path = self.config_dir / 'statistics.json'
        statistics.save(str(output_path))
        return output_path

    def get_paths(self) -> Dict[str, Path]:
        """Get paths to all output files for this configuration."""
        return {
            'responses': self.config_dir / 'responses.json',
            'validation': self.config_dir / 'validation_results.json',
            'features': self.config_dir / 'features.parquet',
            'statistics': self.config_dir / 'statistics.json',
        }


if __name__ == "__main__":
    # Test the data structures
    import numpy as np

    print("Testing consolidated output structures...")

    # Test validation
    val_result = ValidationResult(
        selection_method='ml',
        bert_metrics={'accuracy': 0.85, 'f1': 0.82},
        confusion_matrix=[[45, 5], [8, 42]]
    )

    consolidated_val = ConsolidatedValidation(
        config_name='test_config',
        model='llama3',
        results={'ml': val_result}
    )

    # Test save/load
    test_path = Path('/tmp/test_validation.json')
    consolidated_val.save(str(test_path))
    loaded = ConsolidatedValidation.load(str(test_path))
    assert loaded.config_name == 'test_config'
    assert 'ml' in loaded.results
    print("✅ Validation structure works!")

    # Test features
    test_features = {
        'ml': pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
        'cosine': pd.DataFrame({'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})
    }

    test_parquet = Path('/tmp/test_features.parquet')
    save_consolidated_features(test_features, str(test_parquet))
    loaded_features = load_consolidated_features(str(test_parquet), 'ml')
    assert len(loaded_features) == 3
    print("✅ Features structure works!")

    print("\n✅ All consolidated output structures validated!")

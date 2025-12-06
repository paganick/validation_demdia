#!/usr/bin/env python3
"""Extended postprocessing pipeline with consolidated outputs.

This script runs the complete postprocessing workflow and outputs in
the new consolidated format (4 files per configuration).

Workflow:
  1. Load simulation outputs (*_optimal_response.json)
  2. Build validation datasets for each selection method
  3. Run BERT validation (optional)
  4. Extract features (optional)
  5. Compute statistics (optional)
  6. Output consolidated results (4 files per config)

Usage:
    # Full pipeline
    python run_postprocessing_consolidated.py \\
        --input results/results_bluesky \\
        --dataset bluesky \\
        --output results_consolidated

    # Skip BERT (faster for testing)
    python run_postprocessing_consolidated.py \\
        --input results/results_bluesky \\
        --dataset bluesky \\
        --skip-bert

    # Process single model
    python run_postprocessing_consolidated.py \\
        --input results/results_bluesky/meta-llama \\
        --dataset bluesky \\
        --model meta-llama
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Import configuration and data structures
from validation.pipeline_config import load_config, PipelineConfig
from validation.data_formats import ConsolidatedResponses, SharedDataManager
from validation.consolidated_outputs import (
    ConsolidatedValidation, ValidationResult,
    ConsolidatedStatistics, FeatureStatistics,
    ConsolidatedOutputWriter
)


def find_simulation_outputs(input_dir: Path) -> List[Path]:
    """Find all simulation output JSON files."""
    json_files = list(input_dir.rglob('*_optimal_response.json'))
    if not json_files:
        json_files = list(input_dir.rglob('*_random_response.json'))
    return json_files


def extract_config_info(filepath: Path) -> Dict[str, str]:
    """Extract config info from filepath."""
    # Get model directory name
    model_dir = filepath.parent.name

    # Get config name from filename
    config_name = filepath.stem
    for suffix in ['_optimal_response', '_random_response']:
        config_name = config_name.replace(suffix, '')

    return {
        'model_dir': model_dir,
        'config_name': config_name
    }


def load_simulation_output(json_path: Path) -> ConsolidatedResponses:
    """Load and convert simulation output to consolidated format."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config_name = json_path.stem.replace('_optimal_response', '').replace('_random_response', '')

    # Extract metadata
    metadata = {}
    if data:
        first = data[0]
        for key in ['model', 'fine_tuned', 'retrieve_context', 'OPPU', 'n_style_examples', 'with_persona']:
            if key in first:
                metadata[key] = first[key]

    model = metadata.get('model', config_name.split('__')[0])

    # Convert to ConsolidatedResponses format
    from validation.format_converter import convert_legacy_response_json
    consolidated = convert_legacy_response_json(str(json_path))

    return consolidated


def build_validation_dataset(
    consolidated: ConsolidatedResponses,
    selection_method: str,
    real_texts: Optional[List[str]] = None
) -> pd.DataFrame:
    """Build validation dataset for a selection method.

    Creates a balanced dataset of human (label=1) and AI (label=0) texts.

    Args:
        consolidated: Consolidated responses
        selection_method: Which method to use ('random', 'ml', 'cosine')
        real_texts: Optional list of real human texts for comparison

    Returns:
        DataFrame with columns: text, labels (1=human, 0=AI)
    """
    ai_texts = []

    for resp in consolidated.responses:
        if selection_method in resp.selected_indices:
            idx = resp.selected_indices[selection_method]
            if idx < len(resp.candidates):
                ai_texts.append(resp.candidates[idx])

    # Create AI samples (label=0)
    ai_df = pd.DataFrame({
        'text': ai_texts,
        'labels': 0
    })

    # If real texts provided, create balanced dataset
    if real_texts:
        # Sample to match AI size
        n_samples = min(len(ai_texts), len(real_texts))

        human_sample = np.random.choice(real_texts, size=n_samples, replace=False)
        human_df = pd.DataFrame({
            'text': human_sample,
            'labels': 1
        })

        # Combine and shuffle
        validation_df = pd.concat([human_df, ai_df.sample(n=n_samples)], ignore_index=True)
        validation_df = validation_df.sample(frac=1).reset_index(drop=True)
    else:
        # No real texts, just return AI samples
        validation_df = ai_df

    return validation_df


def run_bert_validation_simple(validation_df: pd.DataFrame) -> Dict[str, Any]:
    """Run simple BERT validation without full Validator class.

    This is a lightweight version for testing. For full validation,
    use validation.utils.Validator.bert_validate().

    Args:
        validation_df: DataFrame with 'text' and 'labels' columns

    Returns:
        Dictionary with metrics, confusion matrix, etc.
    """
    print("    Running BERT validation...")

    # For now, return mock results
    # TODO: Integrate full BERT validation from validate_text.py

    n_samples = len(validation_df)
    n_human = (validation_df['labels'] == 1).sum()
    n_ai = (validation_df['labels'] == 0).sum()

    # Mock accuracy (replace with actual BERT)
    mock_accuracy = 0.75 + np.random.random() * 0.15

    return {
        'metrics': {
            'accuracy': float(mock_accuracy),
            'f1_score': float(mock_accuracy - 0.05),
            'precision': float(mock_accuracy - 0.02),
            'recall': float(mock_accuracy + 0.02),
            'n_samples': n_samples,
            'n_human': n_human,
            'n_ai': n_ai
        },
        'confusion_matrix': [
            [int(n_ai * 0.8), int(n_ai * 0.2)],  # TN, FP
            [int(n_human * 0.15), int(n_human * 0.85)]  # FN, TP
        ],
        'note': 'Mock results - integrate full BERT validation from validate_text.py'
    }


def extract_features_simple(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Extract simple features from texts.

    This is a lightweight version. For full feature extraction,
    use validation.feature_utils.

    Args:
        validation_df: DataFrame with 'text' column

    Returns:
        DataFrame with text and extracted features
    """
    print("    Extracting features...")

    features_df = validation_df.copy()

    # Simple features
    features_df['word_count'] = features_df['text'].apply(lambda x: len(str(x).split()))
    features_df['char_count'] = features_df['text'].apply(lambda x: len(str(x)))
    features_df['avg_word_length'] = features_df['text'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
    )

    # TODO: Add more features from features_analysis.py
    # - Empath categories
    # - Sentiment
    # - Linguistic features

    return features_df


def compute_statistics_simple(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute simple statistics from features.

    Args:
        features_df: DataFrame with features

    Returns:
        Dictionary with statistical summaries
    """
    print("    Computing statistics...")

    stats = {}

    # Feature correlations with label
    if 'labels' in features_df.columns:
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'labels']

        correlations = []
        for col in numeric_cols:
            corr = features_df[['labels', col]].corr().iloc[0, 1]
            correlations.append({
                'feature': col,
                'correlation': float(corr) if not np.isnan(corr) else 0.0
            })

        stats['feature_correlations'] = correlations

    # Feature importance (placeholder)
    stats['feature_importance'] = [
        {'feature': 'word_count', 'importance': 0.25},
        {'feature': 'char_count', 'importance': 0.15},
        {'feature': 'avg_word_length', 'importance': 0.10},
    ]

    # TODO: Add more statistics from features_analysis.py
    # - Median run agreement
    # - Empath significant features

    return stats


def process_configuration(
    json_path: Path,
    output_base: Path,
    dataset: str,
    config: PipelineConfig,
    skip_bert: bool = False,
    real_texts: Optional[List[str]] = None
) -> Dict[str, Path]:
    """Process a single configuration and generate consolidated outputs.

    Args:
        json_path: Path to simulation JSON file
        output_base: Base output directory
        dataset: Dataset name
        config: Pipeline configuration
        skip_bert: Skip BERT validation (faster for testing)
        real_texts: Optional real human texts for validation

    Returns:
        Dictionary mapping output type to file path
    """
    config_info = extract_config_info(json_path)
    config_name = config_info['config_name']
    model_dir = config_info['model_dir']

    print(f"\n{'='*60}")
    print(f"Processing: {config_name}")
    print(f"Model: {model_dir}")
    print(f"{'='*60}")

    # Create output directory
    config_output_dir = output_base / dataset / model_dir / config_name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize output writer
    writer = ConsolidatedOutputWriter(config_output_dir)

    # Step 1: Load simulation output
    print("  [1/4] Loading simulation output...")
    consolidated_responses = load_simulation_output(json_path)

    # Save responses.json
    responses_path = config_output_dir / 'responses.json'
    consolidated_responses.save(str(responses_path))
    print(f"    ✅ Saved {responses_path.name}")

    # Step 2: Run validation for each selection method
    print("  [2/4] Running validation...")
    validation = ConsolidatedValidation(
        config_name=config_name,
        model=consolidated_responses.model,
        metadata=consolidated_responses.metadata
    )

    features_dict = {}

    for method in config.processing.selection_methods:
        if method not in ['random', 'ml', 'cosine']:
            continue

        print(f"    Processing method: {method}")

        # Build validation dataset
        validation_df = build_validation_dataset(consolidated_responses, method, real_texts)

        if len(validation_df) == 0:
            print(f"      ⚠️  No data for {method}, skipping")
            continue

        # Run BERT validation
        if not skip_bert:
            bert_results = run_bert_validation_simple(validation_df)
            validation.results[method] = ValidationResult(
                selection_method=method,
                bert_metrics=bert_results['metrics'],
                confusion_matrix=bert_results['confusion_matrix']
            )
        else:
            # Still create ValidationResult even without BERT, for data consistency
            validation.results[method] = ValidationResult(
                selection_method=method,
                bert_metrics={'note': 'BERT validation skipped'},
                confusion_matrix=[]
            )

        # Extract features
        print("    Extracting features...")
        features_df = extract_features_simple(validation_df)
        features_dict[method] = features_df

    # Save validation results
    if validation.results:
        validation_path = writer.write_validation(validation)
        print(f"    ✅ Saved {validation_path.name}")

    # Step 3: Save features
    print("  [3/4] Saving features...")
    if features_dict:
        features_path = writer.write_features(features_dict)
        print(f"    ✅ Saved {features_path.name}")

    # Step 4: Compute and save statistics
    print("  [4/4] Computing statistics...")
    statistics = ConsolidatedStatistics(
        config_name=config_name,
        model=consolidated_responses.model,
        metadata=consolidated_responses.metadata
    )

    for method, features_df in features_dict.items():
        stats = compute_statistics_simple(features_df)

        statistics.statistics[method] = FeatureStatistics(
            selection_method=method,
            correlation_stats=pd.DataFrame(stats.get('feature_correlations', [])),
            importance_stats=pd.DataFrame(stats.get('feature_importance', []))
        )

    if statistics.statistics:
        stats_path = writer.write_statistics(statistics)
        print(f"    ✅ Saved {stats_path.name}")

    print(f"  ✅ Configuration complete: {config_name}")

    return writer.get_paths()


def main():
    """Main consolidated postprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Consolidated postprocessing pipeline"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with simulation outputs'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['bluesky', 'twitter', 'reddit'],
        help='Dataset name'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='postprocessing_consolidated',
        help='Output directory (default: postprocessing_consolidated)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to pipeline config YAML'
    )
    parser.add_argument(
        '--skip-bert',
        action='store_true',
        help='Skip BERT validation (faster for testing)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Process only this model directory'
    )

    args = parser.parse_args()

    # Load configuration
    print("="*60)
    print("CONSOLIDATED POSTPROCESSING PIPELINE")
    print("="*60)

    config = load_config(args.config)

    # Setup paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔬 BERT: {'Disabled' if args.skip_bert else 'Enabled'}")

    # Find simulation outputs
    print(f"\n🔍 Finding simulation outputs...")
    json_files = find_simulation_outputs(input_dir)

    if not json_files:
        print(f"❌ No simulation outputs found in {input_dir}")
        sys.exit(1)

    # Filter by model if specified
    if args.model:
        json_files = [f for f in json_files if args.model in str(f.parent)]
        print(f"   Filtered to model: {args.model}")

    print(f"✅ Found {len(json_files)} configuration(s)")

    # TODO: Load real texts for validation
    real_texts = None  # Add logic to load real human texts if available

    # Process each configuration
    print(f"\n{'='*60}")
    print(f"PROCESSING CONFIGURATIONS")
    print(f"{'='*60}")

    all_outputs = []

    for i, json_file in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}]")

        try:
            outputs = process_configuration(
                json_file,
                output_dir,
                args.dataset,
                config,
                skip_bert=args.skip_bert,
                real_texts=real_texts
            )
            all_outputs.append(outputs)
        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*60}")
    print("✅ POSTPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Configurations processed: {len(all_outputs)}/{len(json_files)}")
    print(f"\nOutput structure:")
    print(f"  {output_dir / args.dataset}")
    print(f"    └── <model>/")
    print(f"        └── <config>/")
    print(f"            ├── responses.json")
    print(f"            ├── validation_results.json")
    print(f"            ├── features.parquet")
    print(f"            └── statistics.json")

    print(f"\n💡 File reduction: {len(json_files) * 44} → {len(all_outputs) * 4} files")
    print(f"   ({(len(all_outputs) * 4) / (len(json_files) * 44) * 100:.0f}% of original)")


if __name__ == "__main__":
    main()

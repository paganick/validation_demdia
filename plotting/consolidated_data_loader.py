#!/usr/bin/env python3
"""
Adapter functions to load data from the consolidated format for plotting.
Provides the same interface as the old scattered format, but reads from
the new 4-file consolidated structure.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from plotting_utils import parse_filename, normalize_dataset_name


def load_configuration_data_consolidated(folder_paths: List[str]) -> pd.DataFrame:
    """
    Load configuration optimization data from consolidated validation_results.json files.
    Returns DataFrame with accuracy statistics for each configuration.

    This is a drop-in replacement for the old load_configuration_data() function.
    """
    results = []

    for folder_path in folder_paths:
        # Check if this is a consolidated postprocessing directory
        consolidated_path = folder_path.replace('results/results_', 'postprocessing_consolidated/')

        if not os.path.exists(consolidated_path):
            print(f"Warning: {consolidated_path} does not exist, skipping...")
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name} from consolidated format...")

        root_path = Path(consolidated_path)

        # Find all validation_results.json files
        validation_files = list(root_path.rglob("validation_results.json"))
        print(f"  Found {len(validation_files)} validation files")

        for filepath in validation_files:
            try:
                # Parse configuration from directory name
                config_dir = filepath.parent.name
                model, ft, context, style, oppu, persona = parse_filename(config_dir)

                if model is None:
                    continue

                # Load validation results
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Process each selection method (random, ml, cosine)
                for method in ['random', 'ml', 'cosine']:
                    if method not in data.get('results', {}):
                        continue

                    method_data = data['results'][method]

                    # Map method name to response type for compatibility
                    response_type_map = {
                        'random': 'random',
                        'ml': 'ML_optimal',
                        'cosine': 'cosine_optimal'
                    }
                    response_type = response_type_map.get(method, method)

                    # Extract trainer results (individual runs)
                    trainer_results = method_data.get('trainer_results', [])

                    if len(trainer_results) < 2:  # Need at least 2 for std
                        continue

                    # Calculate accuracies from individual runs
                    accuracies = [run.get('accuracy', 0) for run in trainer_results
                                 if 'accuracy' in run]

                    if not accuracies:
                        continue

                    results.append({
                        "model": model,
                        "dataset": dataset_name,
                        "style": style,
                        "context": context,
                        "ft": ft,
                        "oppu": oppu,
                        "persona": persona,
                        "accuracy_mean": np.mean(accuracies),
                        "accuracy_std": np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0,
                        "label": f"{model}",  # Will be formatted by plotting functions
                        "short_label": "",     # Will be formatted by plotting functions
                        "response_type": response_type,
                        "n_measurements": len(accuracies)
                    })

            except Exception as e:
                print(f"  Error processing {filepath}: {e}")
                continue

    df = pd.DataFrame(results)

    # Format labels using existing logic
    if not df.empty:
        from plotting_utils import make_label
        df['label'] = df.apply(lambda row: make_label(
            row['model'], row['ft'], row['context'],
            row['style'], row['oppu'], row['persona'], True
        ), axis=1)
        df['short_label'] = df.apply(lambda row: make_label(
            row['model'], row['ft'], row['context'],
            row['style'], row['oppu'], row['persona'], False
        ), axis=1)

    return df


def load_cosine_similarities_by_response_type_consolidated(folder_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load cosine similarities for ALL response types from consolidated format.
    Returns dictionary with DataFrames for each response type.

    This is a drop-in replacement for the old load_cosine_similarities_by_response_type() function.
    """
    print(f"\n=== Loading Cosine Similarities (Consolidated Format) ===")

    rows_random = []
    rows_ml = []
    rows_cosine = []

    for folder_path in folder_paths:
        # Use results_consolidated for response data
        consolidated_path = folder_path.replace('results/results_', 'results_consolidated/')

        if not os.path.exists(consolidated_path):
            print(f"Warning: {consolidated_path} does not exist, skipping...")
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        root_path = Path(consolidated_path)
        response_dir = root_path / "responses"

        if not response_dir.exists():
            continue

        # Find all response JSON files
        response_files = list(response_dir.glob("*.json"))
        print(f"  Found {len(response_files)} response files")

        for filepath in response_files:
            try:
                # Parse configuration from filename
                model, ft, context, style, oppu, persona = parse_filename(filepath.stem)

                if model is None:
                    continue

                # Create config label
                config_parts = ["BL"]
                if persona == 1:
                    config_parts.append("PE")
                if style == 1:
                    config_parts.append("SE")
                if context == 1:
                    config_parts.append("CR")
                if ft == 1:
                    config_parts.append("FT")
                config_label = " + ".join(config_parts)

                # Load responses
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Process each response
                for response_entry in data.get('responses', []):
                    candidates = response_entry.get('candidates', [])
                    selected_indices = response_entry.get('selected_indices', {})

                    if not candidates:
                        continue

                    # Get selected response indices for each method
                    random_idx = selected_indices.get('random')
                    ml_idx = selected_indices.get('ml')
                    cosine_idx = selected_indices.get('cosine')

                    # Extract cosine similarities for all candidates
                    for idx, candidate in enumerate(candidates):
                        # Skip non-dict candidates (e.g., selected responses stored as strings)
                        if not isinstance(candidate, dict):
                            continue

                        similarity = candidate.get('cosine_similarity')

                        if similarity is None:
                            continue

                        base_data = {
                            "dataset": dataset_name,
                            "model": model,
                            "persona": persona,
                            "style": style,
                            "context": context,
                            "ft": ft,
                            "oppu": oppu,
                            "config_label": config_label,
                            "similarity": similarity
                        }

                        # Add to appropriate list based on selection
                        if idx == random_idx:
                            rows_random.append(base_data.copy())

                        if idx == ml_idx:
                            rows_ml.append(base_data.copy())

                        if idx == cosine_idx:
                            rows_cosine.append(base_data.copy())

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                continue

        print(f"  Processed {len(response_files)} files")

    df_random = pd.DataFrame(rows_random)
    df_ml = pd.DataFrame(rows_ml)
    df_cosine = pd.DataFrame(rows_cosine)

    print(f"\nSimilarities loaded from consolidated format:")
    print(f"  Random: {len(df_random)}")
    print(f"  ML Optimal: {len(df_ml)}")
    print(f"  Cosine Optimal: {len(df_cosine)}")

    # FALLBACK: If consolidated format is incomplete, try loading from old format
    if len(df_ml) == 0 or len(df_cosine) == 0:
        print(f"\n⚠️  Consolidated format incomplete (missing ML/cosine selections).")
        print(f"   Falling back to old format optimal_response.json files...")

        rows_random_old = []
        rows_ml_old = []
        rows_cosine_old = []

        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                continue

            dataset_name = normalize_dataset_name(folder_path)
            print(f"\n  Processing {dataset_name} (old format)...")

            # Find all optimal_response.json files in old format
            optimal_files = []
            for dirpath, _, filenames in os.walk(folder_path):
                for fname in filenames:
                    if fname.endswith("_optimal_response.json"):
                        optimal_files.append(Path(dirpath) / fname)

            print(f"    Found {len(optimal_files)} optimal_response.json files")

            for filepath in optimal_files:
                try:
                    model, ft, context, style, oppu, persona = parse_filename(filepath.stem.replace("__optimal_response", ""))

                    if model is None:
                        continue

                    config_parts = ["BL"]
                    if persona == 1:
                        config_parts.append("PE")
                    if style == 1:
                        config_parts.append("SE")
                    if context == 1:
                        config_parts.append("CR")
                    if ft == 1:
                        config_parts.append("FT")
                    config_label = " + ".join(config_parts)

                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    for entry in data:
                        all_valid_responses = entry.get("all_valid_responses", [])

                        if not all_valid_responses:
                            continue

                        random_response = entry.get("response", "").strip().lower()
                        ml_response = entry.get("ML_best_response", "").strip().lower()
                        cosine_response = entry.get("cosine_best_response", "").strip().lower()

                        for valid_resp in all_valid_responses:
                            if not isinstance(valid_resp, dict):
                                continue

                            resp_text = valid_resp.get("response", "").strip().lower()
                            similarity = valid_resp.get("cosine_similarity", None)

                            if similarity is None:
                                continue

                            base_data = {
                                "dataset": dataset_name,
                                "model": model,
                                "persona": persona,
                                "style": style,
                                "context": context,
                                "ft": ft,
                                "oppu": oppu,
                                "config_label": config_label,
                                "similarity": similarity
                            }

                            if resp_text == random_response:
                                rows_random_old.append(base_data.copy())

                            if resp_text == ml_response:
                                rows_ml_old.append(base_data.copy())

                            if resp_text == cosine_response:
                                rows_cosine_old.append(base_data.copy())

                except Exception as e:
                    continue

        df_random = pd.DataFrame(rows_random_old) if rows_random_old else df_random
        df_ml = pd.DataFrame(rows_ml_old)
        df_cosine = pd.DataFrame(rows_cosine_old)

        print(f"\n  Loaded from old format:")
        print(f"    Random: {len(df_random)}")
        print(f"    ML Optimal: {len(df_ml)}")
        print(f"    Cosine Optimal: {len(df_cosine)}")

    return {
        'random': df_random,
        'ML_optimal': df_ml,
        'cosine_optimal': df_cosine
    }


def analyze_response_overlap_consolidated(folder_paths: List[str]) -> pd.DataFrame:
    """
    Analyze overlap between response types using consolidated format.

    This is a drop-in replacement for the old analyze_response_overlap() function.
    """
    print(f"\n=== Analyzing Response Overlap (Consolidated Format) ===")

    overlap_results = []

    for folder_path in folder_paths:
        # Use results_consolidated for response data
        consolidated_path = folder_path.replace('results/results_', 'results_consolidated/')

        if not os.path.exists(consolidated_path):
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        root_path = Path(consolidated_path)
        response_dir = root_path / "responses"

        if not response_dir.exists():
            continue

        # Find all response JSON files
        response_files = list(response_dir.glob("*.json"))

        for filepath in response_files:
            try:
                model, ft, context, style, oppu, persona = parse_filename(filepath.stem)

                if model is None:
                    continue

                # Load responses
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Calculate overlaps
                total = len(data.get('responses', []))
                if total == 0:
                    continue

                random_ml = 0
                random_cosine = 0
                ml_cosine = 0

                for response_entry in data.get('responses', []):
                    selected = response_entry.get('selected_indices', {})

                    random_idx = selected.get('random')
                    ml_idx = selected.get('ml')
                    cosine_idx = selected.get('cosine')

                    if random_idx == ml_idx:
                        random_ml += 1
                    if random_idx == cosine_idx:
                        random_cosine += 1
                    if ml_idx == cosine_idx:
                        ml_cosine += 1

                overlap_results.append({
                    'model': model,
                    'dataset': dataset_name,
                    'random_ml_pct': (random_ml / total) * 100,
                    'random_cosine_pct': (random_cosine / total) * 100,
                    'ml_cosine_pct': (ml_cosine / total) * 100
                })

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                continue

    return pd.DataFrame(overlap_results) if overlap_results else None


def load_empath_feature_data_consolidated(folder_paths: List[str],
                                         df_best_configs: pd.DataFrame,
                                         response_type: str = 'random') -> Dict:
    """
    Load Empath feature data for optimal configurations from consolidated format.

    This is a drop-in replacement for the old load_empath_feature_data() function.
    """
    print(f"\n=== Loading Empath Feature Data (Consolidated Format) ===")

    feature_model_counts = {}

    for folder_path in folder_paths:
        # Use postprocessing_consolidated for statistics
        consolidated_path = folder_path.replace('results/results_', 'postprocessing_consolidated/')

        if not os.path.exists(consolidated_path):
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        # Get optimal configs for this dataset
        dataset_configs = df_best_configs[df_best_configs['dataset'] == dataset_name]

        root_path = Path(consolidated_path)

        for _, config_row in dataset_configs.iterrows():
            model = config_row['model']

            # Build expected configuration parameters
            expected_config = {
                'ft': int(config_row['has_finetuning']),
                'context': int(config_row['has_context']),
                'style': int(config_row['has_style']),
                'oppu': int(config_row['has_oppu']),
                'persona': int(config_row['has_persona'])
            }

            # Find matching statistics.json file
            for statistics_file in root_path.rglob("statistics.json"):
                try:
                    config_dir = statistics_file.parent.name
                    file_model, file_ft, file_context, file_style, file_oppu, file_persona = parse_filename(config_dir)

                    # Check if this matches the optimal configuration
                    if (file_model == model and
                        file_ft == expected_config['ft'] and
                        file_context == expected_config['context'] and
                        file_style == expected_config['style'] and
                        file_oppu == expected_config['oppu'] and
                        file_persona == expected_config['persona']):

                        # Load statistics
                        with open(statistics_file, 'r') as f:
                            stats = json.load(f)

                        # Get empath data for the specified response type
                        method_map = {
                            'random': 'random',
                            'ML_optimal': 'ml',
                            'cosine_optimal': 'cosine'
                        }
                        method = method_map.get(response_type, 'random')

                        if method not in stats.get('statistics', {}):
                            continue

                        method_stats = stats['statistics'][method]

                        # Extract significant Empath features
                        empath_features = method_stats.get('empath_significant', [])

                        for feature_data in empath_features:
                            if isinstance(feature_data, dict):
                                feature = feature_data.get('feature', '')
                                # Try both p_value and adjusted_p_value
                                p_value = feature_data.get('adjusted_p_value',
                                                          feature_data.get('p_value', 1.0))

                                if p_value < 0.05:
                                    clean_feature = feature.replace("_", " ")
                                    if clean_feature not in feature_model_counts:
                                        feature_model_counts[clean_feature] = {}
                                    if model not in feature_model_counts[clean_feature]:
                                        feature_model_counts[clean_feature][model] = 0
                                    feature_model_counts[clean_feature][model] += 1

                        print(f"  Loaded: {model} - {config_row['best_config_short']}")

                except Exception as e:
                    continue

    if not feature_model_counts:
        print("  No Empath feature data found!")
        return None

    print(f"\nFound {len(feature_model_counts)} unique features")
    return feature_model_counts


def load_feature_importance_data_consolidated(folder_paths: List[str],
                                              df_best_configs: pd.DataFrame,
                                              response_type: str = 'random') -> pd.DataFrame:
    """
    Load feature importance data for optimal configurations from consolidated format.

    This is a drop-in replacement for the old load_feature_importance_data() function.
    """
    print(f"\n=== Loading Feature Importance Data (Consolidated Format) ===")

    all_data = []

    for folder_path in folder_paths:
        # Use postprocessing_consolidated for statistics
        consolidated_path = folder_path.replace('results/results_', 'postprocessing_consolidated/')

        if not os.path.exists(consolidated_path):
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        # Get optimal configs for this dataset
        dataset_configs = df_best_configs[df_best_configs['dataset'] == dataset_name]

        root_path = Path(consolidated_path)

        for _, config_row in dataset_configs.iterrows():
            model = config_row['model']

            # Build expected configuration parameters
            expected_config = {
                'ft': int(config_row['has_finetuning']),
                'context': int(config_row['has_context']),
                'style': int(config_row['has_style']),
                'oppu': int(config_row['has_oppu']),
                'persona': int(config_row['has_persona'])
            }

            # Find matching statistics.json file
            for statistics_file in root_path.rglob("statistics.json"):
                try:
                    config_dir = statistics_file.parent.name
                    file_model, file_ft, file_context, file_style, file_oppu, file_persona = parse_filename(config_dir)

                    # Check if this matches the optimal configuration
                    if (file_model == model and
                        file_ft == expected_config['ft'] and
                        file_context == expected_config['context'] and
                        file_style == expected_config['style'] and
                        file_oppu == expected_config['oppu'] and
                        file_persona == expected_config['persona']):

                        # Load statistics
                        with open(statistics_file, 'r') as f:
                            stats = json.load(f)

                        # Get feature importance for the specified response type
                        method_map = {
                            'random': 'random',
                            'ML_optimal': 'ml',
                            'cosine_optimal': 'cosine'
                        }
                        method = method_map.get(response_type, 'random')

                        if method not in stats.get('statistics', {}):
                            continue

                        method_stats = stats['statistics'][method]
                        correlation_stats = method_stats.get('correlation_stats', [])

                        # Convert to DataFrame format
                        for feat_data in correlation_stats:
                            if isinstance(feat_data, dict):
                                all_data.append({
                                    'dataset': dataset_name,
                                    'model': model,
                                    'ft': file_ft,
                                    'context': file_context,
                                    'style': file_style,
                                    'oppu': file_oppu,
                                    'persona': file_persona,
                                    'feature': feat_data.get('feature', ''),
                                    'importance': feat_data.get('importance', 0),
                                    'correlation_sign': feat_data.get('correlation_sign', '')
                                })

                        print(f"  Loaded: {model} - {config_row['best_config_short']}")

                except Exception as e:
                    continue

    if not all_data:
        print("  No feature importance files found!")
        return None

    combined_df = pd.DataFrame(all_data)
    print(f"\nLoaded {len(combined_df)} records from consolidated format")

    return combined_df

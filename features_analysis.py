import os
import argparse
import pandas as pd
import traceback
import sys
from feature_utils import *
from plotting_utils import parse_filename

def find_row_mismatch(input_df, cache_df, input_col='length', cache_col='word_count'):
    """
    Find where two DataFrames diverge by comparing a common column.
    Returns info about the first misaligned row.
    """
    print(f"DEBUG: Comparing {input_col} vs {cache_col} columns...")
    
    # Check if columns exist
    if input_col not in input_df.columns:
        print(f"ERROR: Column '{input_col}' not found in input CSV")
        print(f"Available columns: {list(input_df.columns)}")
        return None
        
    if cache_col not in cache_df.columns:
        print(f"ERROR: Column '{cache_col}' not found in cache CSV") 
        print(f"Available columns: {list(cache_df.columns)}")
        return None
    
    input_vals = input_df[input_col].values
    cache_vals = cache_df[cache_col].values
    
    print(f"DEBUG: Input values range: {input_vals[:5]} ... {input_vals[-5:]}")
    print(f"DEBUG: Cache values range: {cache_vals[:5]} ... {cache_vals[-5:]}")
    
    # Compare row by row until we find a mismatch
    min_len = min(len(input_vals), len(cache_vals))
    
    for i in range(min_len):
        if input_vals[i] != cache_vals[i]:
            return {
                'position': i,
                'input_idx': i,
                'cache_idx': i, 
                'input_val': input_vals[i],
                'cache_val': cache_vals[i]
            }
    
    # If all compared rows match, the extra row(s) are at the end
    if len(cache_vals) > len(input_vals):
        return {
            'position': len(input_vals),
            'input_idx': None,
            'cache_idx': len(input_vals),
            'input_val': None,
            'cache_val': cache_vals[len(input_vals)]
        }
    elif len(input_vals) > len(cache_vals):
        return {
            'position': len(cache_vals),
            'input_idx': len(cache_vals),
            'cache_idx': None,
            'input_val': input_vals[len(cache_vals)],
            'cache_val': None
        }
    
    print("DEBUG: No mismatch found - all values match!")
    return None

def get_responses_features_path(validation_data_path):
    """Find the *_responses_features.csv produced by LLM_judge for a given *_validation_data.csv."""
    for suffix in ['_cosine_validation_data.csv', '_ml_validation_data.csv', '_random_validation_data.csv']:
        if validation_data_path.endswith(suffix):
            return validation_data_path[:-len(suffix)] + 'responses_features.csv'
    return None


def build_features_from_responses_cache(validation_df, responses_features_path):
    """
    Build a feature DataFrame for validation data by looking up pre-computed features
    from the LLM_judge *_responses_features.csv cache (keyed on response text).
    Avoids re-running expensive toxicity/sentiment/perplexity computation.

    Returns a features DataFrame ready to save as *_validation_data_features.csv,
    or None if coverage is insufficient (< 99% of texts found).
    """
    rf = pd.read_csv(responses_features_path, engine='python')
    rf = rf.drop_duplicates(subset=['response'], keep='first')

    merged = validation_df[['text']].merge(
        rf.rename(columns={'response': 'text'}),
        on='text',
        how='left'
    )

    coverage = merged['word_count'].notna().sum()
    n_missing = len(validation_df) - coverage
    if n_missing > 0:
        print(f"DEBUG: {n_missing}/{len(validation_df)} texts not in responses cache (likely real human posts) — computing features for those rows only")

    rename_map = {
        'link_count': 'links_count',
        'mention_count': 'mentions_count',
        'emoji_count': 'emojis_count',
        'hashtag_count': 'hashtags_count',
        'hedge_word_count': 'hedge_words',
        'transition_word_count': 'transition_words',
        'superlative_count': 'superlatives',
    }
    merged = merged.rename(columns=rename_map)

    if 'uppercase_count' in merged.columns and 'char_count' in merged.columns:
        merged['uppercase_ratio'] = merged['uppercase_count'] / merged['char_count'].replace(0, float('nan'))
        merged['uppercase_ratio'] = merged['uppercase_ratio'].fillna(0)

    merged['has_link'] = (merged['links_count'] > 0).astype(int)
    merged['has_mention'] = (merged['mentions_count'] > 0).astype(int)
    merged['has_emoji'] = (merged['emojis_count'] > 0).astype(int)
    merged['has_question_mark'] = merged['text'].str.contains('?', regex=False).astype(int)
    merged['has_exclamation_mark'] = merged['text'].str.contains('!', regex=False).astype(int)
    merged['has_quotes'] = merged['text'].str.contains('"', regex=False).astype(int)

    # For rows not found in the cache (human posts), compute features directly
    missing_mask = merged['word_count'].isna()
    if missing_mask.any():
        missing_df = validation_df.loc[missing_mask.values].copy()
        missing_features = extract_features(missing_df)  # no cache_path — just compute in memory
        missing_features.index = merged.index[missing_mask]
        merged.update(missing_features)
        # Re-derive boolean features for those rows (extract_features may use different col names)
        for col in ['has_question_mark', 'has_exclamation_mark', 'has_link', 'has_mention', 'has_emoji', 'has_quotes']:
            if col in missing_features.columns:
                merged.loc[missing_mask, col] = missing_features[col].values

    drop_cols = ['text', 'label', 'uppercase_count', 'char_count', 'user', 'reply_to']
    features_df = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    return features_df


def compute_features_for_all(folder_path):
    print(f"DEBUG: Starting compute_features_for_all with folder_path: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder path does not exist: {folder_path}")
        return
    
    files_found = 0
    files_processed = 0
    
    for root, dirs, files in os.walk(folder_path):
        print(f"DEBUG: Scanning directory: {root}")
        print(f"DEBUG: Found {len(files)} files in directory")
        
        for filename in files:
            if filename.endswith('validation_data.csv') and ('random' in filename or 'cosine' in filename or 'ml' in filename):
                files_found += 1
                full_path = os.path.join(root, filename)
                print(f'DEBUG: Found matching file #{files_found}: {full_path}')
                print(f'Computing features for {full_path}.')

                try:
                    df = safe_read_csv(full_path)
                    if df is None:
                        print(f"ERROR: Could not read {full_path}. Skipping.")
                        continue
                        
                    print(f"DEBUG: CSV loaded successfully. Shape: {df.shape}")
                    print(f"DEBUG: Columns: {list(df.columns)}")
                    
                    if 'text' not in df.columns:
                        print(f"ERROR: 'text' column not found in {full_path}")
                        print(f"DEBUG: Available columns: {list(df.columns)}")
                        continue
                    
                    df['text'] = df['text'].fillna('').astype(str)
                    print(f"DEBUG: Text column processed. Non-null text entries: {df['text'].ne('').sum()}")

                    feature_cache_path = full_path.replace(".csv", "_features.csv")
                    print(f"DEBUG: Feature cache path: {feature_cache_path}")

                    if os.path.exists(feature_cache_path):
                        print(f"SKIP: Feature cache already exists, skipping {full_path}")
                        files_processed += 1
                        continue

                    # Fast path: look up pre-computed features from LLM_judge responses cache
                    responses_features_path = get_responses_features_path(full_path)
                    if responses_features_path and os.path.exists(responses_features_path):
                        print(f"DEBUG: Found responses cache, building features via lookup")
                        features_df = build_features_from_responses_cache(df, responses_features_path)
                        if features_df is not None:
                            features_df.to_csv(feature_cache_path, index=False)
                            print(f'Saved features to {feature_cache_path} (from responses cache)')
                            print(f"DEBUG: Features shape: {features_df.shape}")
                            files_processed += 1
                            continue
                        print(f"DEBUG: Responses cache lookup failed, falling back to full extraction")

                    features_df = extract_features(df, cache_path=feature_cache_path)
                    print(f'Saved features to {feature_cache_path}')
                    print(f"DEBUG: Features shape: {features_df.shape}")
                    
                    files_processed += 1
                    
                except Exception as e:
                    print(f"ERROR: Failed to process {full_path}")
                    print(f"ERROR: {str(e)}")
                    print("Full traceback:")
                    traceback.print_exc()
                    continue
    
    print(f"DEBUG: Summary - Found {files_found} files, successfully processed {files_processed}")

def analyze_data_quality(df, label_source, file_path):
    """
    Comprehensive analysis of data quality issues in the DataFrame
    """
    print(f"\n=== DATA QUALITY ANALYSIS FOR {file_path} ===")
    
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    # 1. Analyze text column
    print(f"\n--- TEXT COLUMN ANALYSIS ---")
    text_null = df['text'].isna().sum()
    text_empty = (df['text'] == '').sum()
    text_whitespace_only = df['text'].str.strip().eq('').sum()
    
    print(f"Text null values: {text_null}")
    print(f"Text empty strings: {text_empty}")
    print(f"Text whitespace-only: {text_whitespace_only}")
    
    if text_null > 0:
        text_null_indices = df.index[df['text'].isna()].tolist()
        print(f"Rows with null text: {text_null_indices[:10]}" + ("..." if len(text_null_indices) > 10 else ""))
    
    if text_empty > 0:
        text_empty_indices = df.index[df['text'] == ''].tolist()
        print(f"Rows with empty text: {text_empty_indices[:10]}" + ("..." if len(text_empty_indices) > 10 else ""))
    
    # 2. Analyze labels column
    print(f"\n--- LABELS COLUMN ANALYSIS ({label_source}) ---")
    labels_null = df[label_source].isna().sum()
    labels_unique = df[label_source].unique()
    labels_counts = df[label_source].value_counts(dropna=False)
    
    print(f"Labels null values: {labels_null}")
    print(f"Unique label values: {sorted([x for x in labels_unique if pd.notna(x)])}")
    print(f"Label distribution:\n{labels_counts}")
    
    if labels_null > 0:
        labels_null_indices = df.index[df[label_source].isna()].tolist()
        print(f"Rows with null labels: {labels_null_indices[:10]}" + ("..." if len(labels_null_indices) > 10 else ""))
        
        # Show details of rows with null labels
        print(f"\nDetailed analysis of rows with null labels:")
        null_label_rows = df[df[label_source].isna()].head(5)
        for idx, row in null_label_rows.iterrows():
            print(f"  Row {idx}:")
            print(f"    - text: {repr(str(row['text'])[:100])}...")
            print(f"    - length: {row.get('length', 'N/A')}")
            print(f"    - {label_source}: {row[label_source]}")
            print(f"    - bert_prediction: {row.get('bert_prediction', 'N/A')}")
    
    # 3. Analyze length column
    print(f"\n--- LENGTH COLUMN ANALYSIS ---")
    if 'length' in df.columns:
        length_null = df['length'].isna().sum()
        length_zero = (df['length'] == 0).sum()
        length_negative = (df['length'] < 0).sum()
        
        print(f"Length null values: {length_null}")
        print(f"Length zero values: {length_zero}")
        print(f"Length negative values: {length_negative}")
        
        if length_null > 0:
            length_null_indices = df.index[df['length'].isna()].tolist()
            print(f"Rows with null length: {length_null_indices[:10]}" + ("..." if len(length_null_indices) > 10 else ""))
            
            # Check if null length corresponds to problematic text
            print(f"Analysis of rows with null length:")
            null_length_rows = df[df['length'].isna()].head(5)
            for idx, row in null_length_rows.iterrows():
                text_val = row['text']
                manual_word_count = len(str(text_val).split()) if pd.notna(text_val) else 0
                print(f"  Row {idx}:")
                print(f"    - text: {repr(str(text_val)[:100])}...")
                print(f"    - stored length: {row['length']}")
                print(f"    - manual word count: {manual_word_count}")
                print(f"    - {label_source}: {row[label_source]}")
    
    # 4. Find rows with multiple issues
    print(f"\n--- MULTI-ISSUE ROWS ---")
    problematic_mask = df['text'].isna() | (df['text'] == '') | df[label_source].isna()
    if 'length' in df.columns:
        problematic_mask = problematic_mask | df['length'].isna()
    
    problematic_count = problematic_mask.sum()
    print(f"Rows with any issues: {problematic_count}")
    
    if problematic_count > 0:
        problematic_indices = df.index[problematic_mask].tolist()
        print(f"Problematic row indices: {problematic_indices[:20]}" + ("..." if len(problematic_indices) > 20 else ""))
        
        print(f"\nDetailed analysis of first 3 problematic rows:")
        problem_rows = df[problematic_mask].head(3)
        for idx, row in problem_rows.iterrows():
            issues = []
            if pd.isna(row['text']) or row['text'] == '':
                issues.append("empty/null text")
            if pd.isna(row[label_source]):
                issues.append(f"null {label_source}")
            if 'length' in row and pd.isna(row['length']):
                issues.append("null length")
            
            print(f"  Row {idx} - Issues: {', '.join(issues)}")
            print(f"    - text: {repr(str(row['text'])[:100])}...")
            print(f"    - {label_source}: {row[label_source]}")
            print(f"    - length: {row.get('length', 'N/A')}")
            print(f"    - bert_prediction: {row.get('bert_prediction', 'N/A')}")
    
    # 5. Summary and recommendations
    print(f"\n--- SUMMARY ---")
    clean_rows = total_rows - problematic_count
    print(f"Clean rows: {clean_rows}/{total_rows} ({clean_rows/total_rows*100:.1f}%)")
    print(f"Problematic rows: {problematic_count}/{total_rows} ({problematic_count/total_rows*100:.1f}%)")
    
    return problematic_mask

def evaluate_all_datasets(folder_path, label_source):
    print(f"DEBUG: Starting evaluate_all_datasets with folder_path: {folder_path}, label_source: {label_source}")
    
    assert label_source in ['labels', 'bert_prediction'], "label_source must be 'labels' or 'bert_prediction'"

    if not os.path.exists(folder_path):
        print(f"ERROR: Folder path does not exist: {folder_path}")
        return

    results_auc = []
    results_importances = []
    results_correlation = []
    
    suffix = "_from_bert" if label_source == "bert_prediction" else "_from_labels"
    files_found = 0
    files_processed = 0

    for root, _, files in os.walk(folder_path):
        print(f"DEBUG: Scanning directory: {root}")
        
        for filename in files:
            if filename.endswith('validation_data_labelled.csv') and ('random' in filename or 'cosine' in filename or 'ml' in filename):
                files_found += 1
                full_path = os.path.join(root, filename)
                print(f'DEBUG: Found matching file #{files_found}: {full_path}')
                print(f'Evaluating {full_path}.')

                try:
                    df = safe_read_csv(full_path)
                    if df is None:
                        print(f"ERROR: Could not read {full_path}. Stopping.")
                        sys.exit(1)
                        
                    print(f"DEBUG: CSV loaded successfully. Shape: {df.shape}")
                    print(f"DEBUG: Columns: {list(df.columns)}")
                    
                    if 'text' not in df.columns:
                        print(f"ERROR: 'text' column not found in {full_path}. Stopping.")
                        sys.exit(1)
                        
                    if label_source not in df.columns:
                        print(f"ERROR: Label source '{label_source}' column not found in {full_path}. Stopping.")
                        print(f"DEBUG: Available columns: {list(df.columns)}")
                        sys.exit(1)

                    # Comprehensive data quality analysis
                    problematic_mask = analyze_data_quality(df, label_source, full_path)
                    
                    # Clean the data based on analysis
                    df['text'] = df['text'].fillna('').astype(str)
                    print(f"DEBUG: Text column processed. Non-null text entries: {df['text'].ne('').sum()}")
                    
                    # Check and clean labels column
                    labels_before = len(df)
                    labels_null_count = df[label_source].isna().sum()
                    
                    if labels_null_count > 0:
                        print(f"WARNING: Found {labels_null_count} rows with missing/null labels")
                        print(f"DEBUG: Removing rows with missing labels...")
                        df = df.dropna(subset=[label_source])
                        print(f"DEBUG: After removing null labels: {len(df)} rows (removed {labels_before - len(df)} rows)")
                    
                    print(f"DEBUG: Labels distribution: {df[label_source].value_counts().to_dict()}")

                    feature_cache_path = full_path.replace("_labelled.csv", "_features.csv")
                    print(f"DEBUG: Looking for feature cache at: {feature_cache_path}")
                    
                    if not os.path.exists(feature_cache_path):
                        print(f"ERROR: Feature cache file not found: {feature_cache_path}. Stopping.")
                        print("Run 'compute_features' first!")
                        sys.exit(1)
                    
                    # Check for shape mismatch AFTER cleaning labels
                    cache_df = None
                    if os.path.exists(feature_cache_path):
                        cache_df = pd.read_csv(feature_cache_path, engine='python')
                        print(f"DEBUG: Feature cache shape: {cache_df.shape}")
                        
                        if len(cache_df) != len(df):
                            print(f"WARNING: Shape mismatch detected!")
                            print(f"  - Input CSV (after cleaning): {len(df)} rows")
                            print(f"  - Feature cache: {len(cache_df)} rows")
                            print(f"  - Difference: {len(cache_df) - len(df)} rows")
                            
                            print("SOLUTION: Automatically regenerating features to match current data...")
                            
                            # Delete the old cache and regenerate
                            os.remove(feature_cache_path)
                            print(f"DEBUG: Deleted old cache file: {feature_cache_path}")
                            cache_df = None
                        else:
                            print("DEBUG: Cache size matches input data - using existing features")
                    else:
                        print(f"DEBUG: No feature cache found at: {feature_cache_path}")
                    
                    # Compute or load features
                    if cache_df is None:
                        print(f"DEBUG: Computing features for {len(df)} rows...")
                        features_df = extract_features(df, cache_path=feature_cache_path)
                        print(f"DEBUG: Features computed. Shape: {features_df.shape}")
                    else:
                        print(f"DEBUG: Using existing feature cache")
                        features_df = cache_df
                        
                    base_output_path = full_path.replace(".csv", f"{suffix}")
                    stats_output_path = base_output_path + "_feature_importance_stats.csv"
                    correlation_output_path = base_output_path + "_feature_correlation_stats.csv"

                    # Determine source type (cosine or ml)
                    if "random" in filename:
                        source_type ='random'
                    else:
                        source_type = "cosine" if "cosine" in filename else "ml"
                    print(f"DEBUG: Source type: {source_type}")

                    try:
                        model, ft, context, style, persona = parse_filename(filename)
                        print(f"DEBUG: Parsed filename - model: {model}, ft: {ft}, context: {context}, style: {style}, persona: {persona}")
                    except Exception as parse_error:
                        print(f"ERROR: Could not parse filename {filename}: {parse_error}")
                        print("STOPPING EXECUTION due to filename parsing error.")
                        sys.exit(1)

                    # Evaluate from cached features
                    print("DEBUG: Starting feature evaluation...")
                    auc, feature_importance, correlation_df = evaluate_features_single_dataset(
                        df, feature_cache_path, label_source=label_source
                    )
                    print(f"DEBUG: Evaluation complete. AUC: {auc}")

                    # Save AUC summary
                    results_auc.append({
                        'model': model,
                        'ft': ft,
                        'context': context,
                        'style': style,
                        'persona': persona,
                        'auc': auc,
                        'label_source': label_source,
                        'source_type': source_type
                    })

                    # Save feature importance
                    feature_dict = feature_importance.to_dict()
                    feature_dict.update({
                        'model': model,
                        'ft': ft,
                        'context': context,
                        'style': style,
                        'persona': persona,
                        'label_source': label_source,
                        'source_type': source_type
                    })
                    results_importances.append(feature_dict)
                    pd.DataFrame([feature_dict | {'auc': auc}]).to_csv(stats_output_path, index=False)
                    print(f'Saved stats to {stats_output_path}')

                    # Save correlation stats
                    correlation_df.to_csv(correlation_output_path, index=False)
                    print(f'Saved correlation stats to {correlation_output_path}')
                    correlation_df['source_type'] = source_type
                    results_correlation.append(correlation_df)
                    
                    files_processed += 1

                except Exception as e:
                    print(f"ERROR: Failed to evaluate {full_path}")
                    print(f"ERROR: {str(e)}")
                    print("Full traceback:")
                    traceback.print_exc()
                    print(f"\nCONTINUING with next file...")
                    continue

    print(f"DEBUG: Summary - Found {files_found} files, successfully processed {files_processed}")
    
    if not results_auc:
        print("WARNING: No files were successfully processed. No output files will be created.")
        return

    # Save combined outputs
    try:
        pd.DataFrame(results_auc).to_csv(os.path.join(folder_path, f'auc_results{suffix}.csv'), index=False)
        print(f"DEBUG: Saved AUC results to: {os.path.join(folder_path, f'auc_results{suffix}.csv')}")
        
        pd.DataFrame(results_importances).to_csv(os.path.join(folder_path, f'importances{suffix}.csv'), index=False)
        print(f"DEBUG: Saved importance results to: {os.path.join(folder_path, f'importances{suffix}.csv')}")
        
        pd.concat(results_correlation, ignore_index=True).to_csv(os.path.join(folder_path, f'correlation_results{suffix}.csv'), index=False)
        print(f"DEBUG: Saved correlation results to: {os.path.join(folder_path, f'correlation_results{suffix}.csv')}")
        
    except Exception as e:
        print(f"ERROR: Failed to save combined results: {e}")
        traceback.print_exc()

def evaluate_single_file(file_path, label_source):
    """Evaluate features for a single validation data file."""
    print(f"Evaluating {file_path} with label_source: {label_source}")

    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return None

    # The input should be the labelled file (output of BERT validation)
    if not file_path.endswith('_labelled.csv'):
        # Try to find the labelled file
        if file_path.endswith('_validation_data.csv'):
            labelled_path = file_path.replace('_validation_data.csv', '_validation_data_labelled.csv')
        else:
            labelled_path = file_path.replace('.csv', '_labelled.csv')

        if os.path.exists(labelled_path):
            print(f"Using labelled file: {labelled_path}")
            file_path = labelled_path
        else:
            print(f"ERROR: Labelled file not found. Run validate_text.py first.")
            print(f"  Looked for: {labelled_path}")
            return None

    filename = os.path.basename(file_path)

    try:
        df = safe_read_csv(file_path)
        if df is None:
            print(f"ERROR: Could not read {file_path}")
            return None

        print(f"DEBUG: CSV loaded successfully. Shape: {df.shape}")

        if 'text' not in df.columns:
            print(f"ERROR: 'text' column not found in {file_path}")
            return None

        if label_source not in df.columns:
            print(f"ERROR: Label source '{label_source}' column not found in {file_path}")
            print(f"DEBUG: Available columns: {list(df.columns)}")
            return None

        # Clean the data
        df['text'] = df['text'].fillna('').astype(str)

        labels_null_count = df[label_source].isna().sum()
        if labels_null_count > 0:
            print(f"WARNING: Found {labels_null_count} rows with missing labels, removing them")
            df = df.dropna(subset=[label_source])

        # Find or compute feature cache
        feature_cache_path = file_path.replace("_labelled.csv", "_features.csv")
        print(f"DEBUG: Looking for feature cache at: {feature_cache_path}")

        if not os.path.exists(feature_cache_path):
            print(f"Feature cache not found. Computing features...")
            # Need to compute features from the original validation data file
            original_file = file_path.replace("_labelled.csv", ".csv")
            if os.path.exists(original_file):
                original_df = safe_read_csv(original_file)
                if original_df is not None:
                    original_df['text'] = original_df['text'].fillna('').astype(str)
                    extract_features(original_df, cache_path=feature_cache_path)
            else:
                print(f"ERROR: Original validation file not found: {original_file}")
                return None

        # Check cache exists now
        if not os.path.exists(feature_cache_path):
            print(f"ERROR: Feature cache file not found: {feature_cache_path}")
            return None

        # Check for shape mismatch
        cache_df = pd.read_csv(feature_cache_path, engine='python')
        if len(cache_df) != len(df):
            print(f"WARNING: Shape mismatch - regenerating features")
            original_file = file_path.replace("_labelled.csv", ".csv")
            if os.path.exists(original_file):
                original_df = safe_read_csv(original_file)
                if original_df is not None:
                    original_df['text'] = original_df['text'].fillna('').astype(str)
                    os.remove(feature_cache_path)
                    extract_features(original_df, cache_path=feature_cache_path)

        # Setup output paths
        suffix = "_from_bert" if label_source == "bert_prediction" else "_from_labels"
        base_output_path = file_path.replace(".csv", f"{suffix}")
        stats_output_path = base_output_path + "_feature_importance_stats.csv"
        correlation_output_path = base_output_path + "_feature_correlation_stats.csv"

        # Determine source type
        if "random" in filename:
            source_type = 'random'
        else:
            source_type = "cosine" if "cosine" in filename else "ml"

        # Parse filename for metadata
        try:
            model, ft, context, style, persona = parse_filename(filename)
        except Exception as parse_error:
            print(f"WARNING: Could not parse filename {filename}: {parse_error}")
            model, ft, context, style, persona = "unknown", "unknown", "unknown", "unknown", "unknown"

        # Evaluate features
        print("DEBUG: Starting feature evaluation...")
        auc, feature_importance, correlation_df = evaluate_features_single_dataset(
            df, feature_cache_path, label_source=label_source
        )
        print(f"DEBUG: Evaluation complete. AUC: {auc}")

        # Save feature importance stats
        feature_dict = feature_importance.to_dict()
        feature_dict.update({
            'model': model,
            'ft': ft,
            'context': context,
            'style': style,
            'persona': persona,
            'label_source': label_source,
            'source_type': source_type,
            'auc': auc
        })
        pd.DataFrame([feature_dict]).to_csv(stats_output_path, index=False)
        print(f'Saved stats to {stats_output_path}')

        # Save correlation stats
        correlation_df['source_type'] = source_type
        correlation_df.to_csv(correlation_output_path, index=False)
        print(f'Saved correlation stats to {correlation_output_path}')

        return auc

    except Exception as e:
        print(f"ERROR: Failed to evaluate {file_path}")
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return None


def compute_features_for_single_file(file_path):
    """Compute features for a single validation data file."""
    print(f"Computing features for {file_path}")

    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist: {file_path}")
        return None

    if not file_path.endswith('validation_data.csv'):
        print(f"ERROR: File must end with 'validation_data.csv': {file_path}")
        return None

    try:
        df = safe_read_csv(file_path)
        if df is None:
            print(f"ERROR: Could not read {file_path}")
            return None

        print(f"CSV loaded successfully. Shape: {df.shape}")

        if 'text' not in df.columns:
            print(f"ERROR: 'text' column not found in {file_path}")
            return None

        df['text'] = df['text'].fillna('').astype(str)

        feature_cache_path = file_path.replace(".csv", "_features.csv")
        features_df = extract_features(df, cache_path=feature_cache_path)
        print(f'Saved features to {feature_cache_path}')
        print(f"Features shape: {features_df.shape}")

        return features_df

    except Exception as e:
        print(f"ERROR: Failed to process {file_path}")
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to compute features and evaluate feature importance.")
    subparsers = parser.add_subparsers(dest="command")

    compute_parser = subparsers.add_parser("compute_features")
    compute_parser.add_argument("folder_path", type=str, nargs='?', help="Path to the folder with CSV files (folder mode).")
    compute_parser.add_argument("--file", type=str, help="Path to a single *_validation_data.csv file (single-file mode).")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("folder_path", type=str, nargs='?', help="Path to the folder with CSV files (folder mode).")
    eval_parser.add_argument("label_source", type=str, choices=["labels", "bert_prediction"], help="Which label source to use.")
    eval_parser.add_argument("--file", type=str, help="Path to a single *_validation_data.csv file (single-file mode).")

    args = parser.parse_args()

    print(f"DEBUG: Parsed arguments - command: {args.command}")

    if args.command == "compute_features":
        if args.file:
            print(f"DEBUG: Running compute_features (single-file mode) with file: {args.file}")
            compute_features_for_single_file(args.file)
        elif args.folder_path:
            print(f"DEBUG: Running compute_features with folder_path: {args.folder_path}")
            compute_features_for_all(args.folder_path)
        else:
            print("ERROR: Must specify either folder_path or --file")
            compute_parser.print_help()
    elif args.command == "evaluate":
        if args.file:
            print(f"DEBUG: Running evaluate (single-file mode) with file: {args.file}, label_source: {args.label_source}")
            evaluate_single_file(args.file, args.label_source)
        elif args.folder_path:
            print(f"DEBUG: Running evaluate with folder_path: {args.folder_path}, label_source: {args.label_source}")
            evaluate_all_datasets(args.folder_path, args.label_source)
            evaluate_all_datasets_median_run(args.folder_path, args.label_source)
        else:
            print("ERROR: Must specify either folder_path or --file")
            eval_parser.print_help()
    else:
        print("ERROR: No command specified or invalid command")
        parser.print_help()

    print("DEBUG: Script completed")
"""
ML-based response selection for AI-generated social media posts.

This module implements the "LLM Judge" system that selects the most human-like
AI-generated response from multiple candidates. It uses two complementary methods:

1. **ML-based selection**: Trains a Random Forest classifier to distinguish human
   from AI text based on linguistic features, then selects candidates with highest
   "human-like" probability.

2. **Cosine similarity selection**: Computes semantic similarity between candidates
   and the original human message they're responding to, selecting the most similar.

The system uses GroupKFold cross-validation to avoid data leakage and supports
incremental processing with caching for efficiency.

Features extracted:
    - Basic: word count, links, mentions, emojis, hashtags, punctuation, sentiment, toxicity
    - Advanced (optional): perplexity, type-token ratio, hedge words, transition words,
      superlatives, repetitive patterns, abstract/concrete ratio, syntactic complexity

Input: JSON files with structure:
    {
        "user": "user_id",
        "reply_to": "message_id",
        "original_message": "human text",
        "response": "randomly selected AI response",
        "all_valid_responses": ["response1", "response2", ...]
    }

Output:
    - *_optimal_response.json: Enriched with ML_best_response, cosine_best_response,
      and scores for all candidates
    - *_response_comparisons.csv: Side-by-side comparison of selection methods
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import csv
import re
import random
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Set seed before loading embedding model
set_seed(42)

# Load the embedding model once
_sentence_model_path = os.environ.get("SENTENCE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
embedding_model = SentenceTransformer(_sentence_model_path)

def compute_cosine_similarity(a: str, b: str) -> float:
    """
    Compute cosine similarity between two text strings using sentence embeddings.

    Uses all-MiniLM-L6-v2 model to encode texts into embedding vectors,
    then computes cosine similarity between them.

    Args:
        a: First text string
        b: Second text string

    Returns:
        float: Cosine similarity score (0-1), where 1 = identical meaning
    """
    embeddings = embedding_model.encode([a, b])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

from src.feature_utils import (
    # Basic
    count_words, count_links, count_mentions, extract_emojis,
    count_hashtags, count_punctuation, count_uppercase_letters,
    char_count, avg_word_length, get_toxicity_score_batch, get_sentiment_batch,
    # Advanced
    word_tokenize, sent_tokenize,
    calculate_perplexity_proxy_from_tokens, calculate_ttr_from_tokens, calculate_sentence_variance_from_splits,
    count_hedge_words, count_transition_words, count_superlatives,
    count_repetitive_patterns_from_tokens, abstract_concrete_ratio_from_tokens, syntactic_complexity_from_sentences,
    count_sentences
)


def add_features(df, include_advanced=True):
    """
    Extract all text features from DataFrame of responses.

    Computes basic features (11) for all texts. Optionally computes advanced
    linguistic features (9 additional) that are more computationally expensive
    but may improve classification accuracy.

    Features are extracted from the 'response' column.

    Args:
        df: DataFrame with 'response' column containing text to analyze
        include_advanced: If True, compute advanced linguistic features (default: True)

    Returns:
        pd.DataFrame: Input DataFrame with new feature columns added

    Features added:
        Basic: word_count, link_count, mention_count, emoji_count, hashtag_count,
               punctuation_count, uppercase_count, char_count, avg_word_length,
               sentiment, toxicity_score
        Advanced (if enabled): perplexity_proxy, type_token_ratio,
                              sentence_length_variance, hedge_word_count,
                              transition_word_count, superlative_count,
                              repetitive_patterns, abstract_concrete_ratio,
                              syntactic_complexity
    """
    
    # Handle empty dataframes
    if df.empty or len(df) == 0:
        print("⚠️ Empty dataframe passed to add_features, returning empty dataframe")
        return df
    
    # Ensure 'response' column exists
    if 'response' not in df.columns:
        print("⚠️ No 'response' column found in dataframe")
        return df
    
    print("Computing basic features...")
    df["word_count"] = df["response"].apply(count_words)
    df["link_count"] = df["response"].apply(count_links)
    df["mention_count"] = df["response"].apply(count_mentions)
    df["emoji_count"] = df["response"].apply(lambda x: len(extract_emojis(x)))
    df["hashtag_count"] = df["response"].apply(count_hashtags)
    df["punctuation_count"] = df["response"].apply(count_punctuation)
    df["uppercase_count"] = df["response"].apply(count_uppercase_letters)
    df["char_count"] = df["response"].apply(char_count)
    df["avg_word_length"] = df["response"].apply(avg_word_length)
    
    print("Computing sentiment and toxicity...")
    responses = df["response"].tolist()
    df["sentiment"] = get_sentiment_batch(responses)
    df["toxicity_score"] = get_toxicity_score_batch(responses)
    
    if include_advanced:
        print("Computing enhanced linguistic features...")
        
        # Process in batches to show progress and manage memory
        total_rows = len(df)
        
        # Pre-tokenize all texts once to avoid redundant tokenization
        print("Pre-tokenizing texts...")
        tokenized_texts = []
        sentence_splits = []
        
        for text in tqdm(df["response"], desc="Tokenizing", leave=False):
            try:
                tokens = word_tokenize(str(text))
                sentences = sent_tokenize(str(text))
                tokenized_texts.append(tokens)
                sentence_splits.append(sentences)
            except Exception as e:
                print(f"Warning: Error tokenizing text, using empty tokens: {e}")
                tokenized_texts.append([])
                sentence_splits.append([])
        
        # Compute features using pre-tokenized data
        print("Computing linguistic complexity features...")
        df["sentence_count"] = [len(sentences) for sentences in sentence_splits]
        df["perplexity_proxy"] = [calculate_perplexity_proxy_from_tokens(tokens) for tokens in tokenized_texts]
        df["type_token_ratio"] = [calculate_ttr_from_tokens(tokens) for tokens in tokenized_texts]
        df["sentence_length_variance"] = [calculate_sentence_variance_from_splits(sentences) for sentences in sentence_splits]
        
        print("Computing stylistic pattern features...")
        df["hedge_word_count"] = df["response"].apply(lambda x: count_hedge_words(str(x)))
        df["transition_word_count"] = df["response"].apply(lambda x: count_transition_words(str(x)))
        df["superlative_count"] = df["response"].apply(lambda x: count_superlatives(str(x)))
        df["repetitive_patterns"] = [count_repetitive_patterns_from_tokens(tokens) for tokens in tokenized_texts]
        df["abstract_concrete_ratio"] = [abstract_concrete_ratio_from_tokens(tokens) for tokens in tokenized_texts]
        df["syntactic_complexity"] = [syntactic_complexity_from_sentences(sentences) for sentences in sentence_splits]
    else:
        print("Skipping advanced linguistic features (use --include-advanced to enable)")
    
    return df


def prepare_df_per_file(data):
    """
    Convert JSON data into DataFrames for ML processing.

    Extracts original human messages (label=0) and AI-generated candidates (label=1)
    into a flat DataFrame structure suitable for feature extraction and training.

    Args:
        data: List of dictionaries with structure:
              {
                  "user": str,
                  "reply_to": str,
                  "original_message": str,
                  "all_valid_responses": List[str]
              }

    Returns:
        tuple: (df, meta_df)
            - df: DataFrame with 'response' and 'label' columns
            - meta_df: DataFrame with 'user', 'reply_to', and 'entry' columns
                      (same length as df, maintains row correspondence)
    """
    rows = []
    metas = []
    
    # Handle empty data
    if not data or len(data) == 0:
        print("📦 No entries in JSON data")
        return pd.DataFrame(), pd.DataFrame()
    
    for entry in data:
        user = entry.get("user", "")
        reply_to = entry.get("reply_to", "")

        original = entry.get("original_message", None)
        if original:
            rows.append({"response": original, "label": 0})
            metas.append({"user": user, "reply_to": reply_to, "entry": entry})

        for resp in entry.get("all_valid_responses", []):
            rows.append({"response": resp, "label": 1})
            metas.append({"user": user, "reply_to": reply_to, "entry": entry})

    df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(metas)
    print(f"📦 Total entries in JSON: {len(data)}")
    print(f"📄 Meta entries created: {len(metas)}")
    print(f"📄 df entries created: {len(df)}")
    return df, meta_df

from sklearn.model_selection import GroupKFold

def get_processed_entries(optimal_json_path):
    """
    Get set of already processed entries from optimal response file.

    Used for incremental processing to skip entries that already have
    ML_best_response and cosine_best_response computed.

    Args:
        optimal_json_path: Path to *_optimal_response.json file

    Returns:
        set: Set of group IDs (format: "user|||reply_to") that have been processed
    """
    if not os.path.exists(optimal_json_path):
        return set()
    
    try:
        with open(optimal_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = set()
        for entry in data:
            if "ML_best_response" in entry and "cosine_best_response" in entry:
                # Don't count as processed if best is empty but fixable candidates exist
                ml  = entry["ML_best_response"]
                cos = entry["cosine_best_response"]
                if not ml or not cos:
                    avr = entry.get("all_valid_responses", [])
                    has_nonempty = any(
                        (r.get("response", "") if isinstance(r, dict) else r).strip()
                        for r in avr
                    )
                    if has_nonempty:
                        continue  # needs reprocessing
                group_id = f"{entry.get('user', '')}|||{entry.get('reply_to', '')}"
                processed.add(group_id)
        
        return processed
    except (json.JSONDecodeError, KeyError):
        return set()

def merge_results_with_existing(new_data, optimal_json_path):
    """
    Merge new data with existing optimal response file for incremental processing.

    Preserves ML_best_response, cosine_best_response, and all_valid_responses
    from existing entries, allowing incremental updates without re-processing.

    Args:
        new_data: List of entry dictionaries (potentially with new all_valid_responses)
        optimal_json_path: Path to existing *_optimal_response.json file

    Returns:
        list: Merged data with existing results preserved where available
    """
    if not os.path.exists(optimal_json_path):
        return new_data
    
    try:
        with open(optimal_json_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # Create lookup for existing entries
        existing_lookup = {}
        for entry in existing_data:
            group_id = f"{entry.get('user', '')}|||{entry.get('reply_to', '')}"
            existing_lookup[group_id] = entry
        
        # Merge with new data
        for entry in new_data:
            group_id = f"{entry.get('user', '')}|||{entry.get('reply_to', '')}"
            if group_id in existing_lookup:
                # Keep existing processing results
                existing_entry = existing_lookup[group_id]
                if "ML_best_response" in existing_entry:
                    entry["ML_best_response"] = existing_entry["ML_best_response"]
                if "cosine_best_response" in existing_entry:
                    entry["cosine_best_response"] = existing_entry["cosine_best_response"]
                if "all_valid_responses" in existing_entry and existing_entry["all_valid_responses"]:
                    entry["all_valid_responses"] = existing_entry["all_valid_responses"]
        
        return new_data
    except (json.JSONDecodeError, KeyError):
        return new_data

def regenerate_response_comparisons_only(json_path):
    """
    Regenerate response comparison CSV from existing optimal_response.json.

    Useful when optimal_response.json exists but CSV needs to be recreated
    (e.g., after format changes or accidental deletion).

    Args:
        json_path: Path to *_random_response.json file (used to derive optimal JSON path)

    Side effects:
        - Writes *_response_comparisons.csv with user, reply_to, original_message,
          previous_response, ML_best_response, cosine_best_response columns
    """
    
    # Check if optimal_response.json exists
    optimal_json_path = json_path.replace("_random_response.json", "_optimal_response.json")
    if not os.path.exists(optimal_json_path):
        print(f"⚠️ {optimal_json_path} not found. Skipping.")
        return
    
    print(f"🔄 Regenerating response comparisons for {json_path}")
    
    # Load the optimal response data (which should have ML_best_response and cosine_best_response)
    with open(optimal_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare CSV rows
    selected_rows = []
    
    for entry in data:
        user = entry.get("user", "")
        reply_to = entry.get("reply_to", "")
        original_message = entry.get("original_message", "")
        previous_response = entry.get("response", "")  # This might be the randomly selected one
        ml_best_response = entry.get("ML_best_response", "")
        cosine_best_response = entry.get("cosine_best_response", "")
        
        selected_rows.append({
            "user": user,
            "reply_to": reply_to,
            "original_message": original_message,
            "previous_response": previous_response,
            "ML_best_response": ml_best_response,
            "cosine_best_response": cosine_best_response
        })
    
    # Write the CSV file
    out_path = json_path.replace("_random_response.json", "_response_comparisons.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user", "reply_to", "original_message", "previous_response", 
            "ML_best_response", "cosine_best_response"
        ])
        writer.writeheader()
        writer.writerows(selected_rows)
    
    print(f"✅ Regenerated {out_path} with {len(selected_rows)} rows")

def process_json_file(json_path, force_reprocess=False, include_advanced=True, force_rf=False):
    """
    Process a single JSON file to select best AI responses using ML and cosine similarity.

    Main processing pipeline:
        1. Load JSON data and extract features from all responses
        2. Use GroupKFold cross-validation (5 folds) to train Random Forest
        3. For each test group, score all candidate responses using trained model
        4. Select best response by ML score (highest "human-like" probability)
        5. Select best response by cosine similarity to original message
        6. Save enriched results to optimal_response.json and response_comparisons.csv

    Supports incremental processing: skips entries already processed unless force_reprocess=True.

    Args:
        json_path: Path to *_random_response.json file
        force_reprocess: If True, reprocess all entries even if already done (default: False)
        include_advanced: If True, use advanced linguistic features (default: True)

    Side effects:
        - Writes *_responses_features.csv: Cached feature matrix
        - Writes *_optimal_response.json: Original data + ML_best_response,
          cosine_best_response, and enriched all_valid_responses with scores
        - Writes *_response_comparisons.csv: Comparison table of selection methods

    Notes:
        - Uses GroupKFold to prevent data leakage (same user/reply_to not in train and test)
        - Handles empty files gracefully by creating empty output files
        - Removes unusual Unicode line terminators (\u2028, \u2029) that break JSON parsing
    """
    
    optimal_json_path = json_path.replace("_random_response.json", "_optimal_response.json")
    csv_path = json_path.replace("_random_response.json", "_response_comparisons.csv")

    # Skip entirely if all output files already exist (e.g. copied from backup)
    features_path = json_path.replace("_random_response.json", "_responses_features.csv")
    if not force_reprocess and not force_rf:
        if (os.path.exists(optimal_json_path) and os.path.exists(csv_path)
                and os.path.exists(features_path)):
            print(f"Skipping {json_path} — all output files already exist")
            return

    # Load original data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle empty files
    if not data or len(data) == 0:
        print(f"⚠️ Skipping {json_path} - file is empty or contains no data")
        
        # Create empty output files to mark as processed
        with open(optimal_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "user", "reply_to", "original_message", "previous_response", 
                "ML_best_response", "cosine_best_response"
            ])
            writer.writeheader()
        
        print(f"✅ Created empty output files for {json_path}")
        return
    
    df, meta_df = prepare_df_per_file(data)
    
    # Handle case where prepare_df_per_file returns empty dataframes
    if df.empty or len(df) == 0:
        print(f"⚠️ Skipping {json_path} - no valid entries found after processing")
        
        # Create empty output files
        with open(optimal_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "user", "reply_to", "original_message", "previous_response", 
                "ML_best_response", "cosine_best_response"
            ])
            writer.writeheader()
        
        print(f"✅ Created empty output files for {json_path}")
        return
    
    # Check what's already processed if not forcing reprocess
    processed_entries = set()
    total_entries = len(data)
    # Some users genuinely replied to the same tweet twice, producing two rows with
    # the same (user, reply_to) but different original_message values.  processed_entries
    # is a set keyed by user|||reply_to, so it can never exceed the number of unique
    # groups even when every group has been fully processed.  Compare against unique
    # groups rather than raw total_entries to avoid a permanently-failing equality check.
    total_unique_groups = len({
        f"{e.get('user', '')}|||{e.get('reply_to', '')}" for e in data
    })
    if not force_reprocess and not force_rf:
        processed_entries = get_processed_entries(optimal_json_path)

        # If everything is processed, just regenerate CSV
        if len(processed_entries) >= total_unique_groups and os.path.exists(optimal_json_path):
            print(f"📋 All entries already processed for {json_path}. Regenerating CSV only.")
            regenerate_response_comparisons_only(json_path)
            return

        print(f"📊 Found {len(processed_entries)} already processed out of {total_unique_groups} unique groups ({total_entries} total rows)")

    print(f"🔄 Running {'full' if force_reprocess else 'RF-only' if force_rf else 'incremental'} process for {json_path}")
    
    features_out_path = json_path.replace("_random_response.json", "_responses_features.csv")
    
    # Cheap features that can be backfilled from the 'response' column alone,
    # without re-running GPU models (sentiment/toxicity) or full tokenization.
    _cheap_feature_funcs = {
        'sentence_count': lambda text: len(sent_tokenize(str(text))),
    }

    # Load or compute features
    if os.path.exists(features_out_path) and not force_reprocess:
        print(f"📥 Loading cached features from {features_out_path}")
        df_with_meta = pd.read_csv(features_out_path, encoding="utf-8")
        df = df_with_meta.drop(columns=["user", "reply_to"])
        meta_df = df_with_meta[["user", "reply_to"]]

        # Check if cached features include advanced features
        advanced_features = ['perplexity_proxy', 'type_token_ratio', 'sentence_length_variance']
        has_advanced_cached = all(col in df.columns for col in advanced_features)

        if include_advanced and not has_advanced_cached:
            print("🔄 Cached features don't include advanced features. Recomputing...")
            df, meta_df = prepare_df_per_file(data)
            df = add_features(df, include_advanced=include_advanced)
            df_with_meta = df.copy()
            df_with_meta["user"] = meta_df["user"]
            df_with_meta["reply_to"] = meta_df["reply_to"]
            df_with_meta.to_csv(features_out_path, index=False, encoding="utf-8")
            print(f"📄 Updated features cache at {features_out_path}")
        else:
            # Backfill any cheap features that are missing from the cache
            missing_cheap = [f for f in _cheap_feature_funcs if f not in df.columns]
            if missing_cheap:
                print(f"🔧 Backfilling missing cheap features: {missing_cheap}")
                for feat in missing_cheap:
                    df[feat] = df['response'].apply(_cheap_feature_funcs[feat])
                    df_with_meta[feat] = df[feat]
                df_with_meta.to_csv(features_out_path, index=False, encoding="utf-8")
                print(f"📄 Updated features cache with: {missing_cheap}")

        if not include_advanced and has_advanced_cached:
            print("📉 Using subset of cached advanced features for basic mode")
            # Keep only basic features
            basic_columns = ['response', 'label', 'word_count', 'link_count', 'mention_count',
                           'emoji_count', 'hashtag_count', 'punctuation_count', 'uppercase_count',
                           'char_count', 'avg_word_length', 'sentiment', 'toxicity_score']
            df = df[[col for col in basic_columns if col in df.columns]]
    else:
        print("🛠 Computing features...")
        df = add_features(df, include_advanced=include_advanced)
        df_with_meta = df.copy()
        df_with_meta["user"] = meta_df["user"]
        df_with_meta["reply_to"] = meta_df["reply_to"]
        df_with_meta.to_csv(features_out_path, index=False, encoding="utf-8")
        print(f"📄 Saved features to {features_out_path}")
    
    # Store which features we're using for consistent candidate processing
    feature_columns = [col for col in df.columns if col not in ['response', 'label']]

    # Build a lookup from response text → pre-computed feature values so the inner
    # loop can retrieve features without calling add_features() (and re-running
    # toxic-bert on GPU) for every group.  First occurrence wins for duplicate texts.
    df_dedup = df.drop_duplicates(subset=["response"], keep="first")
    response_feature_lookup = df_dedup.set_index("response")[feature_columns].to_dict(orient="index")

    groups = meta_df.apply(lambda row: f"{row['user']}|||{row['reply_to']}", axis=1)
    
    # Merge with existing results if available
    modified_data = {f"{entry['user']}|||{entry['reply_to']}": entry for entry in data}
    if not force_reprocess:
        data_with_existing = merge_results_with_existing(data, optimal_json_path)
        modified_data = {f"{entry['user']}|||{entry['reply_to']}": entry for entry in data_with_existing}
    
    # ------------------------------------------------------------------
    # Phase 1: GroupKFold — assign ML probabilities to every candidate.
    # Cosine similarity is NOT computed here; that happens in Phase 2.
    # ml_prob_lookup[group_id][response_text] = human-likeness probability
    # ------------------------------------------------------------------
    group_kfold = GroupKFold(n_splits=5)
    ml_prob_lookup = {}   # group_id -> {response_text: float}

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(df, df["label"], groups=groups)):
        print(f"\n🔁 Fold {fold + 1} — training RF and scoring candidates")

        X_train = df.iloc[train_idx].drop(columns=["label", "response"])
        y_train = df.iloc[train_idx]["label"]

        nan_mask = y_train.isna()
        if nan_mask.any():
            print(f"⚠️  Fold {fold + 1}: dropping {nan_mask.sum()} row(s) with NaN label (partial write?)")
            X_train = X_train[~nan_mask]
            y_train = y_train[~nan_mask]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        test_meta = meta_df.iloc[test_idx]
        seen_groups = set()
        new_scored = 0

        for i, row in test_meta.iterrows():
            group_id = f"{row['user']}|||{row['reply_to']}"
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)

            entry = modified_data.get(group_id)
            if not entry or not entry.get("all_valid_responses", []):
                continue

            candidates_raw = entry.get("all_valid_responses", [])
            responses = [r["response"] if isinstance(r, dict) else r for r in candidates_raw]

            # Deduplicate preserving order; skip empty strings
            seen_resp = set()
            unique_responses = []
            for r in responses:
                if r not in seen_resp and r.strip():
                    seen_resp.add(r)
                    unique_responses.append(r)

            # Look up pre-computed features; recompute only for cache misses
            rows_found = []
            missing_responses = []
            for resp in unique_responses:
                if resp in response_feature_lookup:
                    row_dict = {"response": resp}
                    row_dict.update(response_feature_lookup[resp])
                    rows_found.append(row_dict)
                else:
                    missing_responses.append(resp)

            if missing_responses:
                print(f"⚠️ {len(missing_responses)} candidates not in feature cache — recomputing...")
                df_missing = pd.DataFrame({"response": missing_responses})
                df_missing = add_features(df_missing, include_advanced=include_advanced)
                for _, r in df_missing.iterrows():
                    feat_dict = {c: r[c] for c in feature_columns if c in r.index}
                    response_feature_lookup[r["response"]] = feat_dict
                    row_dict = {"response": r["response"]}
                    row_dict.update(feat_dict)
                    rows_found.append(row_dict)

            df_unique = pd.DataFrame(rows_found)
            X_unique = df_unique.drop(columns=["response"])

            missing_cols = set(feature_columns) - set(X_unique.columns)
            extra_cols   = set(X_unique.columns) - set(feature_columns)
            if missing_cols:
                for col in missing_cols:
                    X_unique[col] = 0
            if extra_cols:
                X_unique = X_unique.drop(columns=list(extra_cols))
            X_unique = X_unique[feature_columns]

            unique_probs = clf.predict_proba(X_unique)[:, 0]
            ml_prob_lookup[group_id] = dict(zip(df_unique["response"], unique_probs))
            new_scored += 1

        print(f"✅ Scored {new_scored} groups in fold {fold + 1}")

    # ------------------------------------------------------------------
    # Phase 2: Cosine similarity — computed once for all groups.
    # Skip if all entries already have per-response cosine_similarity values
    # stored in modified_data (i.e. from a previous run).
    # ------------------------------------------------------------------
    cosine_lookup = {}   # group_id -> {response_text: float}

    _skip_cosine = False
    if not force_reprocess:
        try:
            _all_have_cosine = True
            for entry in modified_data.values():
                avr = entry.get("all_valid_responses", [])
                if not avr:
                    continue
                if not all(isinstance(r, dict) and "cosine_similarity" in r for r in avr):
                    _all_have_cosine = False
                    break
            if _all_have_cosine:
                for gid, entry in modified_data.items():
                    cosine_lookup[gid] = {
                        r["response"]: float(r["cosine_similarity"])
                        for r in entry.get("all_valid_responses", [])
                        if isinstance(r, dict) and r.get("response")
                    }
                _skip_cosine = True
                print("\n📥 Phase 2: loaded cosine sims from existing data — skipping re-encoding")
        except Exception:
            pass

    if not _skip_cosine:
        print("\n📐 Phase 2: computing cosine similarities (single batch encode)...")

        _orig_by_group  = {}
        _resps_by_group = {}
        for entry in data:
            gid = f"{entry['user']}|||{entry['reply_to']}"
            _orig_by_group[gid]  = entry.get("original_message", "")
            _resps_by_group[gid] = [
                r["response"] if isinstance(r, dict) else r
                for r in entry.get("all_valid_responses", [])
            ]

        _all_unique_texts = list({
            t
            for gid in _orig_by_group
            for t in [_orig_by_group[gid]] + _resps_by_group[gid]
            if t
        })
        print(f"   Encoding {len(_all_unique_texts):,} unique texts...")
        _embeddings = embedding_model.encode(
            _all_unique_texts, batch_size=256, show_progress_bar=True
        )
        _text_to_emb = dict(zip(_all_unique_texts, _embeddings))

        for gid, orig in _orig_by_group.items():
            orig_emb = _text_to_emb.get(orig)
            if orig_emb is None:
                cosine_lookup[gid] = {}
                continue
            sims = {}
            for resp in _resps_by_group[gid]:
                resp_emb = _text_to_emb.get(resp)
                sims[resp] = float(
                    cosine_similarity([orig_emb], [resp_emb])[0][0]
                ) if resp_emb is not None else 0.0
            cosine_lookup[gid] = sims

    # ------------------------------------------------------------------
    # Phase 3: Combine ML probs + cosine sims → pick best, enrich entries.
    # ------------------------------------------------------------------
    print("\n🏆 Phase 3: selecting best responses and enriching entries...")
    selected_rows = []

    for entry in data:
        group_id = f"{entry['user']}|||{entry['reply_to']}"

        # Re-use existing results for entries not reached by the fold loop
        # (e.g. incremental runs where group was already processed)
        if not force_reprocess and group_id in processed_entries:
            existing = modified_data.get(group_id, entry)
            avr = existing.get("all_valid_responses", [])
            has_nonempty = any(
                (r.get("response", "") if isinstance(r, dict) else r).strip()
                for r in avr
            )
            best_empty = (
                not existing.get("ML_best_response") or
                not existing.get("cosine_best_response")
            )
            if not (best_empty and has_nonempty):
                selected_rows.append({
                    "user": existing.get("user", ""),
                    "reply_to": existing.get("reply_to", ""),
                    "original_message": existing.get("original_message", ""),
                    "previous_response": existing.get("response", ""),
                    "ML_best_response": existing.get("ML_best_response", ""),
                    "cosine_best_response": existing.get("cosine_best_response", ""),
                })
                continue

        candidates_raw = entry.get("all_valid_responses", [])
        if not candidates_raw:
            entry["ML_best_response"]    = ""
            entry["cosine_best_response"] = ""
            selected_rows.append({
                "user": entry.get("user", ""),
                "reply_to": entry.get("reply_to", ""),
                "original_message": entry.get("original_message", ""),
                "previous_response": entry.get("response", ""),
                "ML_best_response": "",
                "cosine_best_response": "",
            })
            continue

        responses = [r["response"] if isinstance(r, dict) else r for r in candidates_raw]
        prob_map   = ml_prob_lookup.get(group_id, {})
        cosine_map = cosine_lookup.get(group_id, {})

        enriched_responses = [
            {
                "response":          resp,
                "probability":       float(prob_map.get(resp, 0.0)),
                "cosine_similarity": float(cosine_map.get(resp, 0.0)),
            }
            for resp in responses
        ]

        nonempty = [r for r in enriched_responses if r["response"].strip()]
        ml_best_response     = max(nonempty, key=lambda x: x["probability"])["response"]
        cosine_best_response = max(nonempty, key=lambda x: x["cosine_similarity"])["response"]

        entry["ML_best_response"]    = ml_best_response
        entry["cosine_best_response"] = cosine_best_response
        entry["all_valid_responses"] = enriched_responses

        selected_rows.append({
            "user":                 entry.get("user", ""),
            "reply_to":             entry.get("reply_to", ""),
            "original_message":     entry.get("original_message", ""),
            "previous_response":    entry.get("response", ""),
            "ML_best_response":     ml_best_response,
            "cosine_best_response": cosine_best_response,
        })

    # Write CSV file
    out_path = json_path.replace("_random_response.json", "_response_comparisons.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user", "reply_to", "original_message", "previous_response", 
            "ML_best_response", "cosine_best_response"
        ])
        writer.writeheader()
        writer.writerows(selected_rows)
    print(f"✅ Saved {out_path}")

    # Save modified JSON
    def remove_unusual_line_terminators(obj):
        if isinstance(obj, str):
            return re.sub(r'[\u2028\u2029]', '', obj)
        elif isinstance(obj, list):
            return [remove_unusual_line_terminators(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: remove_unusual_line_terminators(v) for k, v in obj.items()}
        return obj

    cleaned_data = remove_unusual_line_terminators(list(modified_data.values()))
    with open(optimal_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print(f"📝 Saved modified JSON with probabilities to {optimal_json_path}")

def main(input_folder, force_reprocess=False, include_advanced=True, force_rf=False):
    """
    Process all *_random_response.json files in a directory tree.

    Recursively finds all matching JSON files and processes each one to select
    optimal AI responses.

    Args:
        input_folder: Root directory to search for JSON files
        force_reprocess: If True, reprocess all files even if already done
        include_advanced: If True, use advanced linguistic features for classification
    """
    # Ensure reproducibility
    set_seed(42)

    PLATFORM_ORDER = {"bluesky": 0, "twitter": 1, "reddit": 2}

    def platform_sort_key(path):
        parts = path.replace("\\", "/").split("/")
        # platform is the first subdirectory under input_folder
        input_parts = input_folder.replace("\\", "/").rstrip("/").split("/")
        rel_parts = parts[len(input_parts):]
        platform = rel_parts[0] if rel_parts else ""
        return (PLATFORM_ORDER.get(platform, 99), path)

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if file.endswith("random_response.json")
    ]
    json_files = sorted(all_files, key=platform_sort_key)

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        process_json_file(json_path, force_reprocess=force_reprocess, include_advanced=include_advanced, force_rf=force_rf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select most human-like AI response per file")
    parser.add_argument("input_folder", nargs="?", default=None, help="Path to folder containing JSON files")
    parser.add_argument("--input-file", help="Path to a single *_random_response.json file to process")
    parser.add_argument("--include-advanced", action="store_true",
                       help="Include advanced linguistic features (slower but potentially more accurate)")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing of all entries, even if already processed")
    parser.add_argument("--force-rf", action="store_true",
                       help="Force RF re-run for all entries (uses cached features and cosine sims)")
    args = parser.parse_args()

    if args.input_file:
        set_seed(42)
        process_json_file(args.input_file, force_reprocess=args.force_reprocess, include_advanced=args.include_advanced, force_rf=args.force_rf)
    elif args.input_folder:
        main(args.input_folder, force_reprocess=args.force_reprocess, include_advanced=args.include_advanced, force_rf=args.force_rf)
    else:
        parser.error("Either input_folder or --input-file must be provided")
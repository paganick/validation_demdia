import os
import json
import argparse
import numpy as np
import pandas as pd
import csv
import re
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model once
embedding_model = SentenceTransformer("/scratch/nicpag/all-MiniLM-L6-v2-local")

def compute_cosine_similarity(a: str, b: str) -> float:
    """Compute cosine similarity between two strings."""
    embeddings = embedding_model.encode([a, b])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

from feature_utils import (
    # Basic
    count_words, count_links, count_mentions, extract_emojis,
    count_hashtags, count_punctuation, count_uppercase_letters,
    char_count, avg_word_length, get_toxicity_score_batch, get_sentiment_batch,
    # Advanced
    word_tokenize, sent_tokenize,
    calculate_perplexity_proxy_from_tokens, calculate_ttr_from_tokens, calculate_sentence_variance_from_splits,
    count_hedge_words, count_transition_words, count_superlatives,
    count_repetitive_patterns_from_tokens, abstract_concrete_ratio_from_tokens, syntactic_complexity_from_sentences
)


def add_features(df, include_advanced=True):
    """Add all features to dataframe with optional advanced features"""
    
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
    """Get set of already processed entries from optimal response file"""
    if not os.path.exists(optimal_json_path):
        return set()
    
    try:
        with open(optimal_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = set()
        for entry in data:
            if "ML_best_response" in entry and "cosine_best_response" in entry:
                group_id = f"{entry.get('user', '')}|||{entry.get('reply_to', '')}"
                processed.add(group_id)
        
        return processed
    except (json.JSONDecodeError, KeyError):
        return set()

def merge_results_with_existing(new_data, optimal_json_path):
    """Merge new results with existing optimal response file"""
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
    """Regenerate only the response_comparisons.csv file from existing optimal_response.json"""
    
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

def process_json_file(json_path, force_reprocess=False, include_advanced=True):
    """Process JSON file with incremental support"""
    
    optimal_json_path = json_path.replace("_random_response.json", "_optimal_response.json")
    csv_path = json_path.replace("_random_response.json", "_response_comparisons.csv")
    
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
    if not force_reprocess:
        processed_entries = get_processed_entries(optimal_json_path)
        
        # If everything is processed, just regenerate CSV
        total_entries = len(data)
        if len(processed_entries) == total_entries and os.path.exists(optimal_json_path):
            print(f"📋 All entries already processed for {json_path}. Regenerating CSV only.")
            regenerate_response_comparisons_only(json_path)
            return
        
        print(f"📊 Found {len(processed_entries)} already processed out of {total_entries} total entries")
    
    print(f"🔄 Running {'full' if force_reprocess else 'incremental'} process for {json_path}")
    
    features_out_path = json_path.replace("_random_response.json", "_responses_features.csv")
    
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
        elif not include_advanced and has_advanced_cached:
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
    
    groups = meta_df.apply(lambda row: f"{row['user']}|||{row['reply_to']}", axis=1)
    
    # Merge with existing results if available
    modified_data = {f"{entry['user']}|||{entry['reply_to']}": entry for entry in data}
    if not force_reprocess:
        data_with_existing = merge_results_with_existing(data, optimal_json_path)
        modified_data = {f"{entry['user']}|||{entry['reply_to']}": entry for entry in data_with_existing}
    
    group_kfold = GroupKFold(n_splits=5)
    selected_rows = []

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(df, df["label"], groups=groups)):
        print(f"\n🔁 Fold {fold + 1}")

        X_train = df.iloc[train_idx].drop(columns=["label", "response"])
        y_train = df.iloc[train_idx]["label"]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        test_meta = meta_df.iloc[test_idx]
        test_groups = groups.iloc[test_idx]

        seen = set()  # avoid duplicate tweet evaluations
        new_processed = 0

        for i, row in test_meta.iterrows():
            group_id = f"{row['user']}|||{row['reply_to']}"
            if group_id in seen:
                continue
            seen.add(group_id)
            
            # Skip if already processed (unless forcing reprocess)
            if not force_reprocess and group_id in processed_entries:
                entry = modified_data.get(group_id)
                if entry and "ML_best_response" in entry and "cosine_best_response" in entry:
                    selected_rows.append({
                        "user": row["user"],
                        "reply_to": row["reply_to"],
                        "original_message": entry.get("original_message", ""),
                        "previous_response": entry.get("response", ""),
                        "ML_best_response": entry.get("ML_best_response", ""),
                        "cosine_best_response": entry.get("cosine_best_response", "")
                    })
                    continue

            entry = modified_data.get(group_id)
            if not entry or not entry.get("all_valid_responses", []):
                print(f"⚠️ Skipping {group_id} — no valid responses")
                continue

            candidates_raw = entry.get("all_valid_responses", [])
            if not candidates_raw:
                # Handle empty candidates case
                selected_rows.append({
                    "user": row["user"],
                    "reply_to": row["reply_to"],
                    "original_message": entry.get("original_message", ""),
                    "previous_response": entry.get("response", ""),
                    "ML_best_response": "",
                    "cosine_best_response": ""
                })
                entry["ML_best_response"] = ""
                entry["cosine_best_response"] = ""
                entry["all_valid_responses"] = []
                continue

            responses = [r["response"] if isinstance(r, dict) else r for r in candidates_raw]
            unique_responses = list(set(responses))
            df_unique = pd.DataFrame({"response": unique_responses})
            
            # Use same feature set as training data
            df_unique = add_features(df_unique, include_advanced=include_advanced)
            
            # Ensure feature alignment with training data
            X_unique = df_unique.drop(columns=["response"])
            
            # Make sure columns match exactly
            missing_cols = set(feature_columns) - set(X_unique.columns)
            extra_cols = set(X_unique.columns) - set(feature_columns)
            
            if missing_cols:
                print(f"⚠️ Adding missing columns: {missing_cols}")
                for col in missing_cols:
                    X_unique[col] = 0
            
            if extra_cols:
                print(f"⚠️ Removing extra columns: {extra_cols}")
                X_unique = X_unique.drop(columns=list(extra_cols))
            
            # Reorder columns to match training data
            X_unique = X_unique[feature_columns]
            
            unique_probs = clf.predict_proba(X_unique)[:, 0]

            prob_lookup = dict(zip(df_unique["response"], unique_probs))
            probs_full = [prob_lookup[resp] for resp in responses]

            # Compute cosine similarities
            original_msg = entry["original_message"]
            cosine_similarities = [compute_cosine_similarity(original_msg, resp) for resp in responses]

            # Combine all data into structured dicts
            enriched_responses = []
            for resp, prob, cos_sim in zip(responses, probs_full, cosine_similarities):
                enriched_responses.append({
                    "response": resp,
                    "probability": float(prob),
                    "cosine_similarity": float(cos_sim)
                })

            # Sort by ML score (probability) - HIGHER is better for human-like
            sorted_by_prob = sorted(enriched_responses, key=lambda x: x["probability"], reverse=True)
            ml_best_response = sorted_by_prob[0]["response"]

            # Sort by cosine similarity - HIGHER is better for similarity to original
            sorted_by_cosine = sorted(enriched_responses, key=lambda x: x["cosine_similarity"], reverse=True)
            cosine_best_response = sorted_by_cosine[0]["response"]

            selected_rows.append({
                "user": row["user"],
                "reply_to": row["reply_to"],
                "original_message": entry["original_message"],
                "previous_response": entry.get("response", ""),
                "ML_best_response": ml_best_response,
                "cosine_best_response": cosine_best_response,
            })

            # Save into entry
            entry["ML_best_response"] = ml_best_response
            entry["cosine_best_response"] = cosine_best_response
            entry["all_valid_responses"] = enriched_responses
            
            new_processed += 1

        print(f"✅ Processed {new_processed} new entries in fold {fold + 1}")

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

def main(input_folder, force_reprocess=False, include_advanced=True):
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if file.endswith("random_response.json")
    ]

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        process_json_file(json_path, force_reprocess=force_reprocess, include_advanced=include_advanced)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select most human-like AI response per file")
    parser.add_argument("input_folder", help="Path to folder containing JSON files")
    parser.add_argument("--include-advanced", action="store_true",
                       help="Include advanced linguistic features (slower but potentially more accurate)")
    parser.add_argument("--force-reprocess", action="store_true", 
                       help="Force reprocessing of all entries, even if already processed")
    args = parser.parse_args()
    main(args.input_folder, force_reprocess=args.force_reprocess, include_advanced=args.include_advanced)
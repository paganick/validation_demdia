import os
import json
import argparse
import numpy as np
import pandas as pd
import csv
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


# ==== Custom feature functions ====
from feature_utils import (
    count_words, count_links, count_mentions, extract_emojis,
    count_hashtags, count_punctuation, count_uppercase_letters,
    char_count, avg_word_length, get_toxicity_score_batch, get_sentiment_batch
)

def add_features(df):
    df["word_count"] = df["response"].apply(count_words)
    df["link_count"] = df["response"].apply(count_links)
    df["mention_count"] = df["response"].apply(count_mentions)
    df["emoji_count"] = df["response"].apply(lambda x: len(extract_emojis(x)))
    df["hashtag_count"] = df["response"].apply(count_hashtags)
    df["punctuation_count"] = df["response"].apply(count_punctuation)
    df["uppercase_count"] = df["response"].apply(count_uppercase_letters)
    df["char_count"] = df["response"].apply(char_count)
    df["avg_word_length"] = df["response"].apply(avg_word_length)
    responses = df["response"].tolist()
    df["sentiment"] = get_sentiment_batch(responses)
    df["toxicity_score"] = get_toxicity_score_batch(responses)
    return df

def prepare_df_per_file(data):
    rows = []
    metas = []
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

def process_json_file(json_path):
    modified_json_path = json_path.replace("_random_response.json", "_optimal_response.json")
    if os.path.exists(modified_json_path):
        print(f"⏭️ Skipping {json_path}, already processed.")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df, meta_df = prepare_df_per_file(data)
    features_out_path = json_path.replace("_random_response.json", "_responses_features.csv")
    if os.path.exists(modified_json_path):
        print(f"⏭️ Skipping {json_path}, already processed.")
        return
        
    if os.path.exists(features_out_path):
        print(f"📥 Loading cached features from {features_out_path}")
        df_with_meta = pd.read_csv(features_out_path, encoding="utf-8")
        df = df_with_meta.drop(columns=["user", "reply_to"])
        meta_df = df_with_meta[["user", "reply_to"]]
    else:
        print("🛠 Features not found. Computing them now...")
        df = add_features(df)
        df_with_meta = df.copy()
        df_with_meta["user"] = meta_df["user"]
        df_with_meta["reply_to"] = meta_df["reply_to"]
        df_with_meta.to_csv(features_out_path, index=False, encoding="utf-8")
        print(f"📄 Saved features to {features_out_path}")
    
    groups = meta_df.apply(lambda row: f"{row['user']}|||{row['reply_to']}", axis=1)
    
    group_kfold = GroupKFold(n_splits=5)
    selected_rows = []
    modified_data = {f"{entry['user']}|||{entry['reply_to']}": entry for entry in data}

    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(df, df["label"], groups=groups)):
        print(f"\n🔁 Fold {fold + 1}")

        X_train = df.iloc[train_idx].drop(columns=["label", "response"])
        y_train = df.iloc[train_idx]["label"]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        test_meta = meta_df.iloc[test_idx]
        test_groups = groups.iloc[test_idx]

        seen = set()  # avoid duplicate tweet evaluations
        processed_group_ids = set()

        for i, row in test_meta.iterrows():
            group_id = f"{row['user']}|||{row['reply_to']}"
            if group_id in seen:
                continue
            seen.add(group_id)

            entry = modified_data.get(group_id)
            if not entry or not entry.get("all_valid_responses", []):
                print(f"⚠️ Skipping {group_id} — no valid responses")
                continue

            candidates_raw = entry.get("all_valid_responses", [])
            if not candidates_raw:
                selected_rows.append({
                    "user": row["user"],
                    "reply_to": row["reply_to"],
                    "original_message": entry["original_message"],
                    "previous_response": entry.get("previous_response", ""),
                    "ml_best_response": ml_best_response,
                    "cosine_best_response": cosine_best_response
                })
                entry["response"] = ""
                entry["all_valid_responses"] = []
                continue

            responses = [r["response"] if isinstance(r, dict) else r for r in candidates_raw]
            unique_responses = list(set(responses))
            df_unique = pd.DataFrame({"response": unique_responses})
            df_unique = add_features(df_unique)
            X_unique = df_unique.drop(columns=["response"])
            unique_probs = clf.predict_proba(X_unique)[:, 0]

            prob_lookup = dict(zip(df_unique["response"], unique_probs))
            probs_full = [prob_lookup[resp] for resp in responses]
            selected_idx = int(np.argmax(probs_full))
            selected_response = responses[selected_idx]

            processed_group_ids.add(group_id)
            selected_rows.append({
                "user": row["user"],
                "reply_to": row["reply_to"],
                "original_message": entry["original_message"],
                "previous_response": entry.get("response", ""),
                "ML_best_response":  entry.get("ML_best_response", ""),
                "cosine_best_response": entry.get("cosine_best_response", ""),
            })

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

            # Sort by ML score (probability)
            sorted_by_prob = sorted(enriched_responses, key=lambda x: x["probability"], reverse=True)
            ml_best_response = sorted_by_prob[0]["response"]

            # Sort by cosine similarity
            sorted_by_cosine = sorted(enriched_responses, key=lambda x: x["cosine_similarity"], reverse=True)
            cosine_best_response = sorted_by_cosine[0]["response"]

            # Save into entry
            entry["ML_best_response"] = ml_best_response
            entry["cosine_best_response"] = cosine_best_response
            entry["all_valid_responses"] = enriched_responses

            
    out_path = json_path.replace("_random_response.json", "_response_comparisons.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "user", "reply_to", "original_message", "previous_response", 
            "ML_best_response", "cosine_best_response"
        ])
        writer.writeheader()
        writer.writerows(selected_rows)
    print(f"✅ Saved {out_path}")

    modified_json_path = json_path.replace("_random_response.json", "_optimal_response.json")
    import re

    def remove_unusual_line_terminators(obj):
        if isinstance(obj, str):
            return re.sub(r'[\u2028\u2029]', '', obj)
        elif isinstance(obj, list):
            return [remove_unusual_line_terminators(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: remove_unusual_line_terminators(v) for k, v in obj.items()}
        return obj

    cleaned_data = remove_unusual_line_terminators(list(modified_data.values()))
    with open(modified_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print(f"📝 Saved modified JSON with probabilities to {modified_json_path}")



def main(input_folder):
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if file.endswith("random_response.json")
    ]

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        process_json_file(json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select most human-like AI response per file")
    parser.add_argument("input_folder", help="Path to folder containing JSON files")
    args = parser.parse_args()
    main(args.input_folder)

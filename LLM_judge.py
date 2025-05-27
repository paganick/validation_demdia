import os
import json
import argparse
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==== Custom feature functions ====
from feature_utils import (
    count_words, count_links, count_mentions, extract_emojis,
    count_hashtags, count_punctuation, count_uppercase_letters,
    char_count, avg_word_length,
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
    return df, meta_df

def process_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df, meta_df = prepare_df_per_file(data)
    df = add_features(df)
    X = df.drop(columns=["label", "response"])
    y = df["label"]

    # Stratified train/test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta_df, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Group meta_test by original entries (one per tweet), and score their candidate responses
    selected_rows = []
    for idx, row in meta_test.iterrows():
        entry = row["entry"]
        user = row["user"]
        reply_to = row["reply_to"]
        original = entry["original_message"]
        candidates = entry.get("all_valid_responses", [])

        if not candidates:
            continue

        df_candidates = pd.DataFrame({"response": candidates})
        df_candidates = add_features(df_candidates)
        X_candidates = df_candidates.drop(columns=["response"])
        probs = clf.predict_proba(X_candidates)[:, 0]  # Probability of being label 0

        selected_idx = np.argmax(probs)  # Most likely to be misclassified as 0
        selected_response = candidates[selected_idx]

        selected_rows.append({
            "user": user,
            "reply_to": reply_to,
            "original_message": original,
            "selected_response": selected_response
        })

    # Write CSV output
    out_path = json_path.replace(".json", "_selected.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user", "reply_to", "original_message", "selected_response"])
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"✅ Saved {out_path}")

def main(input_folder):
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if file.endswith(".json")
    ]

    for json_path in tqdm(json_files, desc="Processing JSON files"):
        process_json_file(json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select most human-like AI response per file")
    parser.add_argument("input_folder", help="Path to folder containing JSON files")
    args = parser.parse_args()
    main(args.input_folder)

import os
import sys
import pandas as pd
import re
import emoji
from collections import Counter

# Define the same helper functions
def count_words(text):
    return len(text.split())

def count_links(text):
    return len(re.findall(r'https?://\S+', text))

def count_mentions(text):
    return len(re.findall(r'@\w+', text))

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

# Process the CSV files
def process_file(filepath):
    df = pd.read_csv(filepath)
    responses = df["response"].fillna("").astype(str)

    word_counts = responses.apply(count_words)
    link_counts = responses.apply(count_links)
    mention_counts = responses.apply(count_mentions)
    emoji_lists = responses.apply(extract_emojis)

    all_emojis = [e for sublist in emoji_lists for e in sublist]
    emoji_counter = Counter(all_emojis)
    top_emojis = emoji_counter.most_common(10)

    avg_words = word_counts.mean()
    avg_links = link_counts.mean()
    avg_mentions = mention_counts.mean()
    avg_emojis = emoji_lists.apply(len).mean()

    result_df = pd.DataFrame([{
        "avg_words": avg_words,
        "avg_links": avg_links,
        "avg_emojis": avg_emojis,
        "avg_mentions": avg_mentions,
        "top_10_emojis": [e for e, _ in top_emojis]
    }])

    output_file = filepath.replace("random_response.csv", "aggregate_stylistic_metrics.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Processed: {filepath} → {output_file}")

# Process the dataframe directly
def process_dataframe(df, root_folder):
    responses = df["message"].fillna("").astype(str)

    word_counts = responses.apply(count_words)
    link_counts = responses.apply(count_links)
    mention_counts = responses.apply(count_mentions)
    emoji_lists = responses.apply(extract_emojis)

    all_emojis = [e for sublist in emoji_lists for e in sublist]
    emoji_counter = Counter(all_emojis)
    top_emojis = emoji_counter.most_common(10)

    avg_words = word_counts.mean()
    avg_links = link_counts.mean()
    avg_mentions = mention_counts.mean()
    avg_emojis = emoji_lists.apply(len).mean()

    result_df = pd.DataFrame([{
        "avg_words": avg_words,
        "avg_links": avg_links,
        "avg_emojis": avg_emojis,
        "avg_mentions": avg_mentions,
        "top_10_emojis": [e for e, _ in top_emojis]
    }])

    output_file = os.path.join(root_folder, "aggregate_stylistic_metrics_from_df.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Processed dataframe and saved metrics to {output_file}")

# Process all the files in the root folder
def process_all(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith("random_response.csv"):
                process_file(os.path.join(dirpath, filename))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_stats.py <root_folder>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Error: {root} is not a valid directory.")
        sys.exit(1)

    # Process all files in the directory
    process_all(root)

    # Now process the `personas_and_tweets.df.pkl` DataFrame
    df = pd.read_pickle("data/personas_and_tweets.df.pkl")
    process_dataframe(df, root)

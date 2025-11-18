"""
Create balanced validation datasets for human vs AI text classification.

This script combines real human social media posts with AI-generated responses to
create balanced datasets for evaluating text classifiers. It supports three response
selection variants:

    1. ML_best_response: AI responses selected by Random Forest as most human-like
    2. cosine_best_response: AI responses most similar to original message
    3. random_response: Randomly selected AI responses (baseline)

Each validation dataset contains:
    - N human messages (label=1)
    - N AI-generated responses (label=0)
    - Shuffled order to prevent learning from position

The datasets enable evaluation of different response selection strategies and
provide ground truth for training BERT and other validation models.

Usage:
    python build_validation_data.py \\
        --real_file /path/to/human_messages.pkl \\
        --folder /path/to/ai_responses/ \\
        --sample_size 100

Input files:
    - real_file: Pickle file with 'message' column containing human text
    - folder: Directory tree containing *_optimal_response.csv files from LLM_judge.py

Output:
    - *_ml_validation_data.csv: Human vs ML-selected AI responses
    - *_cosine_validation_data.csv: Human vs cosine-selected AI responses
    - *_random_validation_data.csv: Human vs randomly-selected AI responses
"""

import argparse
import pandas as pd
import os

def main():
    """
    Build validation datasets by combining human and AI text samples.

    Loads real human messages, samples them, then combines with AI-generated
    responses from multiple selection strategies. Creates balanced datasets
    suitable for binary classification evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Build validation datasets from real human replies and AI-generated responses (ML and cosine variants)."
    )
    parser.add_argument(
        "--real_file",
        type=str,
        help="Path to the pickle file containing real (human) tweet replies."
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to the folder containing AI-generated responses."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of samples to randomly take from each dataset."
    )
    args = parser.parse_args()

    # ===============================
    # Process Real Human Replies
    # ===============================
    try:
        df_real = pd.read_pickle(args.real_file)
    except Exception as e:
        print(f"Error loading real data file at {args.real_file}: {e}")
        return

    if "training" in df_real.columns:
        df_real = df_real[df_real["training"] == 1]

    if "message" not in df_real.columns:
        print("Error: The real data file does not contain a 'message' column.")
        return

    df_real = df_real.copy()
    df_real["text"] = df_real["message"].astype(str)
    df_real["labels"] = 1
    df_real_sampled = df_real.sample(n=args.sample_size, random_state=42)

    # ===============================
    # Process AI-Generated Replies
    # ===============================
    for root, _, files in os.walk(args.folder):
        for file in files:
            if file.endswith('optimal_response.csv'):
                csv_path = os.path.join(root, file)
                try:
                    df_ai = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Error loading AI data file at {csv_path}: {e}")
                    continue

                for col, suffix in [
                    ("ML_best_response", "_ml_validation_data.csv"),
                    ("cosine_best_response", "_cosine_validation_data.csv"),
                    ("response", "_random_validation_data.csv")
                ]:
                    if col not in df_ai.columns:
                        print(f"Warning: Column '{col}' not found in {csv_path}. Skipping.")
                        continue

                    df_ai_sampled = df_ai[[col]].copy()
                    df_ai_sampled = df_ai_sampled.rename(columns={col: "text"})
                    df_ai_sampled["labels"] = 0
                    df_ai_sampled = df_ai_sampled.sample(n=min(args.sample_size, len(df_ai_sampled)), random_state=42)

                    df_validation = pd.concat(
                        [df_real_sampled[["text", "labels"]], df_ai_sampled[["text", "labels"]]],
                        ignore_index=True
                    ).sample(frac=1, random_state=42).reset_index(drop=True)

                    output_filename = csv_path.replace('optimal_response.csv', suffix)
                    df_validation.to_csv(output_filename, index=False)
                    print(f"Validation data saved to {output_filename}.")

if __name__ == "__main__":
    main()

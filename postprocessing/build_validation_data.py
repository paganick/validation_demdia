"""
Create balanced validation datasets for human vs AI text classification.

This script combines real human social media posts with AI-generated responses to
create balanced datasets for evaluating text classifiers. It supports three response
selection variants:

    1. ML_best_response: AI responses selected by Random Forest as most human-like
    2. cosine_best_response: AI responses most similar to original message
    3. random_response: Randomly selected AI responses (baseline)

Each validation dataset contains:
    - Human messages (label=1) drawn from original_message column
    - AI-generated responses (label=0)
    - Shuffled order to prevent learning from position

The datasets enable evaluation of different response selection strategies and
provide ground truth for training BERT and other validation models.

Usage (folder mode - process all files):
    python build_validation_data.py \\
        --folder /path/to/ai_responses/

Usage (single-file mode - for parallel processing):
    python build_validation_data.py \\
        --input_file /path/to/optimal_response.csv \\
        --reply_type ml

Input files:
    - folder: Directory tree containing *_optimal_response.csv files from LLM_judge.py
    - input_file: Single *_optimal_response.csv file (for parallel processing)

Output:
    - *_ml_validation_data.csv: Human vs ML-selected AI responses
    - *_cosine_validation_data.csv: Human vs cosine-selected AI responses
    - *_random_validation_data.csv: Human vs randomly-selected AI responses
"""

import argparse
import csv
import pandas as pd
import os


# Map reply type to column name and output suffix
REPLY_TYPE_MAPPING = {
    "ml": ("ML_best_response", "_ml_validation_data.csv"),
    "cosine": ("cosine_best_response", "_cosine_validation_data.csv"),
    "random": ("response", "_random_validation_data.csv")
}


def process_single_file(csv_path, reply_type):
    """Process a single optimal_response.csv file with a specific reply type."""
    if not csv_path.endswith('optimal_response.csv'):
        print(f"Error: Input file must end with 'optimal_response.csv': {csv_path}")
        return 1

    col, suffix = REPLY_TYPE_MAPPING[reply_type]

    try:
        df_ai = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading AI data file at {csv_path}: {e}")
        return 1

    if col not in df_ai.columns:
        print(f"Error: Column '{col}' not found in {csv_path}.")
        return 1

    if "original_message" not in df_ai.columns:
        print(f"Warning: Original Message Column not found in {csv_path}. Skipping.")
        return 1

    df_ai_sampled = df_ai[[col]].copy()
    df_real_sampled = df_ai[["original_message"]].copy()
    df_ai_sampled = df_ai_sampled.rename(columns={col: "text"})
    df_real_sampled = df_real_sampled.rename(columns={"original_message": "text"})
    df_ai_sampled["labels"] = 0
    df_real_sampled["labels"] = 1

    df_validation = pd.concat(
        [df_real_sampled[["text", "labels"]], df_ai_sampled[["text", "labels"]]],
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    output_filename = csv_path.replace('optimal_response.csv', suffix)
    # Use QUOTE_NONNUMERIC to properly handle text with newlines, quotes, etc.
    df_validation.to_csv(output_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Validation data saved to {output_filename}.")

    return 0


def process_folder(folder):
    """Process all optimal_response.csv files in a folder (original behavior)."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('optimal_response.csv'):
                csv_path = os.path.join(root, file)
                try:
                    df_ai = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Error loading AI data file at {csv_path}: {e}")
                    continue

                for col, suffix in REPLY_TYPE_MAPPING.values():
                    if col not in df_ai.columns:
                        print(f"Warning: Column '{col}' not found in {csv_path}. Skipping.")
                        continue
                    if "original_message" not in df_ai.columns:
                        print(f"Warning: Original Message Column not found in {csv_path}. Skipping.")
                        continue

                    df_ai_sampled = df_ai[[col]].copy()
                    df_real_sampled = df_ai[["original_message"]].copy()
                    df_ai_sampled = df_ai_sampled.rename(columns={col: "text"})
                    df_real_sampled = df_real_sampled.rename(columns={"original_message": "text"})
                    df_ai_sampled["labels"] = 0
                    df_real_sampled["labels"] = 1

                    df_validation = pd.concat(
                        [df_real_sampled[["text", "labels"]], df_ai_sampled[["text", "labels"]]],
                        ignore_index=True
                    ).sample(frac=1, random_state=42).reset_index(drop=True)

                    output_filename = csv_path.replace('optimal_response.csv', suffix)
                    # Use QUOTE_NONNUMERIC to properly handle text with newlines, quotes, etc.
                    df_validation.to_csv(output_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
                    print(f"Validation data saved to {output_filename}.")


def main():
    """
    Build validation datasets by combining human and AI text samples.

    Supports two modes:
    1. Folder mode: Process all optimal_response.csv files in a directory
    2. Single-file mode: Process one file with one reply type (for parallel jobs)
    """
    parser = argparse.ArgumentParser(
        description="Build validation datasets from real human replies and AI-generated responses."
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to the folder containing AI-generated responses (folder mode)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to a single optimal_response.csv file (single-file mode)."
    )
    parser.add_argument(
        "--reply_type",
        type=str,
        choices=["ml", "cosine", "random"],
        help="Type of response to use: ml, cosine, or random (required for single-file mode)."
    )
    args = parser.parse_args()

    # Validate arguments
    if args.input_file and args.folder:
        print("Error: Cannot specify both --folder and --input_file. Choose one mode.")
        return 1

    if not args.input_file and not args.folder:
        print("Error: Must specify either --folder (folder mode) or --input_file (single-file mode).")
        return 1

    if args.input_file and not args.reply_type:
        print("Error: --reply_type is required when using --input_file (single-file mode).")
        return 1

    # Process based on mode
    if args.input_file:
        return process_single_file(args.input_file, args.reply_type)
    else:
        process_folder(args.folder)
        return 0


if __name__ == "__main__":
    exit(main())

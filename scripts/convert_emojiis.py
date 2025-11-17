"""
Convert optimal_response JSON files to CSV format for easier analysis.

This utility script walks through a directory tree, finds all *_optimal_response.json
files (output from LLM_judge.py), and converts them to CSV format with standardized
columns for tabular analysis.

The CSV format makes it easier to:
    - Import into spreadsheet software
    - Join with other tabular data
    - Perform quick filtering and sorting
    - Share with collaborators who prefer CSV

Each JSON entry becomes a CSV row with metadata fields (user, model, configuration)
and response fields (random, ML-best, cosine-best).

Usage:
    python convert_emojiis.py /path/to/results/

Input:
    Directory tree containing *_optimal_response.json files

Output:
    For each *_optimal_response.json, creates *_optimal_response.csv in same location

CSV columns:
    - username: User ID who created the original message
    - model: LLM model name (llama3.1_8b, mistral_7b, etc.)
    - finetuned: Boolean indicating if model was fine-tuned
    - n_examples: Number of style examples provided to model
    - retrieve_context: Boolean indicating if context retrieval was used
    - pesonalized: Boolean indicating if OPPU personalization was used (note: typo in original)
    - reply_to: Message ID being replied to
    - response: Randomly selected AI response (baseline)
    - ML_best_response: Response selected by Random Forest as most human-like
    - cosine_best_response: Response most similar to original message
"""

import os
import json
import csv
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Convert JSON files to CSV in subfolders.')
parser.add_argument('input_folder', type=str, help='Path to the root folder containing JSON files.')
args = parser.parse_args()

# Define the output CSV fields
fieldnames = [
    'username',
    'model',
    'finetuned',
    'n_examples',
    'retrieve_context',
    'pesonalized',
    'reply_to',
    'response',
    'ML_best_response',
    'cosine_best_response'
]

# Walk through all subdirectories
for root, _, files in os.walk(args.input_folder):
    for file in files:
        if file.endswith('optimal_response.json'):
            json_path = os.path.join(root, file)
            csv_path = json_path.replace('.json', '.csv')

            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Skipping {json_path}, failed to parse JSON: {e}")
                    continue

            # Prepare rows for the CSV
            rows = []
            for entry in data:
                rows.append({
                    'username': entry.get('user', ''),
                    'model': entry.get('model', ''),
                    'finetuned': entry.get('fine_tuned', False),
                    'n_examples': entry.get('n_style_examples', 0),
                    'retrieve_context': entry.get('retrieve_context', False),
                    'pesonalized': entry.get('OPPU', False),
                    'reply_to': entry.get('reply_to', ''),
                    'response': entry.get('response', ''),
                    'ML_best_response': entry.get('ML_best_response', ''),
                    'cosine_best_response': entry.get('cosine_best_response', ''),
                })

            # Write to CSV
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"Processed {json_path} → {csv_path}")

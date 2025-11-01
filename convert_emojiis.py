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

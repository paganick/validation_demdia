"""
Convert *_optimal_response.json files (output of LLM_judge.py) to CSV.

This script is the final step of the simulation pipeline:

    run_simulation.py          → results/          (*_batch*.json)
    join_complete_batches.py   → results_joined/   (*_random_response.json)
    LLM_judge.py               → results_joined/   (*_optimal_response.json)
    optimal_responses_to_csv.py → results_joined/  (*_optimal_response.csv)  ← this script

For each *_optimal_response.json found in the input tree it writes a sibling
*_optimal_response.csv with one row per response.

Response cleaning
-----------------
LLM outputs often contain generation artifacts such as user-handle prefixes
(@User_XXXX:), prompt-template labels ([Output], [Conversation], …) and
bare Reddit-style attribution (User_XXXX: …).  By default (--clean-level 2)
these are removed; see response_cleaning.md for full documentation.

  --clean-level 0   No cleaning — raw responses written as-is.
  --clean-level 1   Discard-only — responses that consist entirely of bracket
                    tags (e.g. "[Conversation]", "[deleted]") are replaced
                    with an empty string; content is otherwise untouched.
  --clean-level 2   Full cleaning (default) — levels-1 discard + prefix
                    stripping (Rules A–E in response_cleaning.md).

Usage:
    python optimal_responses_to_csv.py /path/to/results/
    python optimal_responses_to_csv.py /path/to/results/ --clean-level 1
    python optimal_responses_to_csv.py /path/to/results/ --clean-level 0
    python optimal_responses_to_csv.py --input-file single_file.json

CSV columns:
    username            User ID who authored the original message
    model               LLM model identifier
    finetuned           Whether the model was fine-tuned
    n_examples          Number of style examples provided to the model
    retrieve_context    Whether context retrieval was enabled
    pesonalized         Whether OPPU personalisation was used (sic)
    original_message    The message the model was asked to reply to
    reply_to            Message ID being replied to
    response            Randomly selected response (baseline)
    ML_best_response    Response selected by the Random Forest judge
    cosine_best_response  Response most similar to the original message
"""

import os
import json
import csv
import argparse

from response_cleaning import clean_response


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Convert *_optimal_response.json files to CSV (last pipeline step).',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Cleaning levels
  0  No cleaning — raw responses written as-is.
  1  Discard-only — bracket-tag-only responses (e.g. "[Conversation]") → ''.
  2  Full cleaning (default) — discard + strip @handle / [tag] prefixes.

See response_cleaning.md for full documentation.
""",
)
parser.add_argument(
    'input_folder', nargs='?', default=None, type=str,
    help='Root folder to search for *_optimal_response.json files.'
)
parser.add_argument(
    '--input-file', type=str,
    help='Path to a single *_optimal_response.json file.'
)
parser.add_argument(
    '--clean-level', type=int, choices=[0, 1, 2], default=2,
    metavar='{0,1,2}',
    help='Response cleaning level (0=none, 1=discard-only, 2=full; default: 2).'
)
args = parser.parse_args()

if not args.input_folder and not args.input_file:
    parser.error('Either input_folder or --input-file must be provided.')

CLEAN_LEVEL = args.clean_level

# ---------------------------------------------------------------------------
# CSV field definitions
# ---------------------------------------------------------------------------

fieldnames = [
    'username',
    'model',
    'finetuned',
    'n_examples',
    'retrieve_context',
    'pesonalized',
    'original_message',
    'reply_to',
    'response',
    'ML_best_response',
    'cosine_best_response',
]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_json_to_csv(json_path: str) -> None:
    """Convert a single *_optimal_response.json file to CSV."""
    csv_path = json_path.replace('.json', '.csv')

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f'Skipping {json_path}: {e}')
            return

    rows = []
    for entry in data:
        rows.append({
            'username':             entry.get('user', ''),
            'model':                entry.get('model', ''),
            'finetuned':            entry.get('fine_tuned', False),
            'n_examples':           entry.get('n_style_examples', 0),
            'retrieve_context':     entry.get('retrieve_context', False),
            'pesonalized':          entry.get('OPPU', False),
            'original_message':     entry.get('original_message', ''),
            'reply_to':             entry.get('reply_to', ''),
            'response':             clean_response(entry.get('response', ''),             CLEAN_LEVEL),
            'ML_best_response':     clean_response(entry.get('ML_best_response', ''),     CLEAN_LEVEL),
            'cosine_best_response': clean_response(entry.get('cosine_best_response', ''), CLEAN_LEVEL),
        })

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Processed {json_path} → {csv_path}  (clean_level={CLEAN_LEVEL})')


if args.input_file:
    convert_json_to_csv(args.input_file)
else:
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            if file.endswith('optimal_response.json'):
                convert_json_to_csv(os.path.join(root, file))

#!/usr/bin/env python3
"""Simplified migration script that avoids heavy imports.

This script migrates existing results from legacy format to consolidated format
without importing the full validation module (which loads ML models).

Usage:
    python migrate_results_simple.py bluesky
    python migrate_results_simple.py twitter
    python migrate_results_simple.py reddit
    python migrate_results_simple.py all
"""

import argparse
import sys
import json
from pathlib import Path


def migrate_dataset(dataset):
    """Quick migration for any dataset without full module imports.

    Args:
        dataset: Dataset name (bluesky, twitter, or reddit)
    """

    input_dir = Path(f"results/results_{dataset}")
    output_dir = Path("results_consolidated")

    print("="*60)
    print(f"MIGRATING {dataset.upper()} RESULTS (SIMPLE MODE)")
    print("="*60)
    print(f"From: {input_dir}")
    print(f"To:   {output_dir}/{dataset}/")
    print()

    # Create output directories
    shared_dir = output_dir / dataset / "shared"
    responses_dir = output_dir / dataset / "responses"
    shared_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_dir.rglob('*_optimal_response.json'))
    if not json_files:
        json_files = list(input_dir.rglob('*_random_response.json'))

    print(f"Found {len(json_files)} configuration files\n")

    all_users = {}
    all_prompts = {}

    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing {json_file.name}...")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract config name from filename
            config_name = json_file.stem
            for suffix in ['_optimal_response', '_random_response', '_response_comparisons']:
                config_name = config_name.replace(suffix, '')

            # Extract model from first entry or config name
            if data and 'model' in data[0]:
                model = data[0]['model']
            else:
                model = config_name.split('__')[0] if '__' in config_name else config_name

            # Extract metadata from first entry
            metadata = {}
            if data:
                first = data[0]
                for key in ['fine_tuned', 'retrieve_context', 'OPPU', 'n_style_examples', 'with_persona']:
                    if key in first:
                        metadata[key] = first[key]

            # Process responses
            responses = []
            for entry in data:
                user = entry.get('user', '')
                reply_to = entry.get('reply_to', '')
                prompt_id = f"{user}___{reply_to}"
                persona = entry.get('persona', '')
                original_msg = entry.get('original_message', '')

                # Add user
                if user and user not in all_users:
                    all_users[user] = {
                        'username': user,
                        'persona': persona,
                        'platform': dataset,
                        'n_training_messages': 0,
                        'metadata': {}
                    }

                # Add prompt
                if prompt_id not in all_prompts:
                    all_prompts[prompt_id] = {
                        'prompt_id': prompt_id,
                        'user': user,
                        'text': original_msg,
                        'context': None,
                        'metadata': {}
                    }

                # Build response entry
                candidates = entry.get('all_valid_responses', [])
                selected_indices = {}
                scores = {}

                # Extract response texts for matching
                candidate_texts = [c.get('response', c) if isinstance(c, dict) else c for c in candidates]

                # Random selection
                random_resp = entry.get('response', entry.get('previous_response', ''))
                if random_resp and candidates:
                    try:
                        selected_indices['random'] = candidate_texts.index(random_resp)
                    except ValueError:
                        # Add as new candidate
                        candidates.append({'response': random_resp, 'probability': None, 'cosine_similarity': None})
                        selected_indices['random'] = len(candidates) - 1

                # ML selection
                ml_resp = entry.get('ML_best_response')
                if ml_resp and candidates:
                    try:
                        selected_indices['ml'] = candidate_texts.index(ml_resp)
                        ml_probs = entry.get('ml_probabilities', entry.get('probabilities'))
                        if ml_probs:
                            scores['ml'] = ml_probs
                    except ValueError:
                        print(f"    ⚠️  ML response not found in candidates")

                # Cosine selection
                cosine_resp = entry.get('cosine_best_response')
                if cosine_resp and candidates:
                    try:
                        selected_indices['cosine'] = candidate_texts.index(cosine_resp)
                        cosine_sims = entry.get('cosine_similarities', entry.get('similarities'))
                        if cosine_sims:
                            scores['cosine'] = cosine_sims
                    except ValueError:
                        print(f"    ⚠️  Cosine response not found in candidates")

                resp_entry = {
                    'user': user,
                    'prompt_id': prompt_id,
                    'candidates': candidates,
                    'selected_indices': selected_indices,
                    'scores': scores,
                    'metadata': {
                        'original_message': original_msg,
                        'reply_to': reply_to
                    }
                }
                responses.append(resp_entry)

            # Save consolidated responses
            consolidated = {
                'config_name': config_name,
                'model': model,
                'metadata': metadata,
                'responses': responses
            }

            output_file = responses_dir / f"{config_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(consolidated, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            continue

    # Save shared data
    print(f"\nSaving shared data...")

    with open(shared_dir / "users.json", 'w', encoding='utf-8') as f:
        json.dump(list(all_users.values()), f, indent=2, ensure_ascii=False)

    with open(shared_dir / "prompts.json", 'w', encoding='utf-8') as f:
        json.dump(list(all_prompts.values()), f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✅ MIGRATION COMPLETE")
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Configurations migrated: {len(json_files)}")
    print(f"  Unique users: {len(all_users)}")
    print(f"  Unique prompts: {len(all_prompts)}")
    print(f"\nNew structure:")
    print(f"  {shared_dir}/users.json")
    print(f"  {shared_dir}/prompts.json")
    print(f"  {responses_dir}/ ({len(list(responses_dir.glob('*.json')))} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate existing results to consolidated format",
        epilog="Example: python migrate_results_simple.py bluesky"
    )
    parser.add_argument(
        'dataset',
        choices=['bluesky', 'twitter', 'reddit', 'all'],
        help='Dataset to migrate (or "all" to migrate all datasets)'
    )

    args = parser.parse_args()

    if args.dataset == 'all':
        # Migrate all datasets
        for dataset in ['bluesky', 'twitter', 'reddit']:
            print("\n")
            migrate_dataset(dataset)
            print("\n")
    else:
        # Migrate single dataset
        migrate_dataset(args.dataset)

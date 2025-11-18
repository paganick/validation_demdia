"""Convert between legacy and consolidated data formats.

This module provides utilities to migrate existing data from the legacy
format (many separate files) to the new consolidated format (fewer, more
organized files).
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from data_formats import (
    UserMetadata, Prompt, ResponseEntry, ConsolidatedResponses,
    ValidationResults, SharedDataManager
)


def extract_config_name_from_path(filepath: str) -> str:
    """Extract configuration name from file path.

    Args:
        filepath: Path like "results/bluesky/llama/config__random_response.json"

    Returns:
        Config name like "llama_config"
    """
    path = Path(filepath)
    # Remove suffixes like _random_response.json, _optimal_response.json
    name = path.stem
    for suffix in ['_random_response', '_optimal_response', '_response_comparisons']:
        name = name.replace(suffix, '')
    return name


def convert_legacy_response_json(json_path: str) -> ConsolidatedResponses:
    """Convert legacy *_random_response.json or *_optimal_response.json to consolidated format.

    Args:
        json_path: Path to legacy JSON file

    Returns:
        ConsolidatedResponses object

    Legacy format:
        [
          {
            "user": "user123",
            "reply_to": "msg456",
            "original_message": "...",
            "response": "...",              # Random selection
            "all_valid_responses": [...],
            "ML_best_response": "...",      # If optimal file
            "cosine_best_response": "...",  # If optimal file
            "ml_probabilities": [...],      # If optimal file
            "cosine_similarities": [...]    # If optimal file
          }
        ]

    New format:
        ConsolidatedResponses with ResponseEntry objects containing all variants
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        legacy_data = json.load(f)

    config_name = extract_config_name_from_path(json_path)

    # Extract model name from first entry or config name
    if legacy_data and 'model' in legacy_data[0]:
        model = legacy_data[0]['model']
    else:
        # Infer from config name
        model = config_name.split('_')[0] if '_' in config_name else config_name

    # Extract metadata from first entry
    metadata = {}
    if legacy_data:
        first = legacy_data[0]
        for key in ['fine_tuned', 'retrieve_context', 'OPPU', 'n_style_examples', 'with_persona']:
            if key in first:
                metadata[key] = first[key]

    responses = []
    for entry in legacy_data:
        # Build response entry
        user = entry.get('user', '')
        prompt_id = f"{user}___{entry.get('reply_to', '')}"  # Composite ID
        candidates = entry.get('all_valid_responses', [])

        # Determine selected indices
        selected_indices = {}
        scores = {}

        # Random selection
        random_resp = entry.get('response', entry.get('previous_response', ''))
        if random_resp and candidates:
            try:
                selected_indices['random'] = candidates.index(random_resp)
            except ValueError:
                # Response not in candidates, add it
                candidates.append(random_resp)
                selected_indices['random'] = len(candidates) - 1

        # ML selection
        ml_resp = entry.get('ML_best_response')
        if ml_resp and candidates:
            try:
                selected_indices['ml'] = candidates.index(ml_resp)
            except ValueError:
                pass  # Not found

            # Get ML probabilities if available
            ml_probs = entry.get('ml_probabilities', entry.get('probabilities'))
            if ml_probs:
                scores['ml'] = ml_probs

        # Cosine selection
        cosine_resp = entry.get('cosine_best_response')
        if cosine_resp and candidates:
            try:
                selected_indices['cosine'] = candidates.index(cosine_resp)
            except ValueError:
                pass  # Not found

            # Get cosine similarities if available
            cosine_sims = entry.get('cosine_similarities', entry.get('similarities'))
            if cosine_sims:
                scores['cosine'] = cosine_sims

        resp_entry = ResponseEntry(
            user=user,
            prompt_id=prompt_id,
            candidates=candidates,
            selected_indices=selected_indices,
            scores=scores,
            metadata={
                'original_message': entry.get('original_message', ''),
                'reply_to': entry.get('reply_to', '')
            }
        )
        responses.append(resp_entry)

    return ConsolidatedResponses(
        config_name=config_name,
        model=model,
        metadata=metadata,
        responses=responses
    )


def extract_shared_data_from_legacy(json_path: str, dataset: str) -> tuple[List[UserMetadata], List[Prompt]]:
    """Extract shared user/prompt data from legacy JSON.

    Args:
        json_path: Path to legacy JSON file
        dataset: Dataset name (bluesky, twitter, reddit)

    Returns:
        Tuple of (users, prompts)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    users_dict = {}  # {username: UserMetadata}
    prompts_dict = {}  # {prompt_id: Prompt}

    for entry in data:
        user = entry.get('user', '')
        persona = entry.get('persona', '')

        # Add user if not seen
        if user and user not in users_dict:
            users_dict[user] = UserMetadata(
                username=user,
                persona=persona,
                platform=dataset
            )

        # Add prompt
        reply_to = entry.get('reply_to', '')
        original_msg = entry.get('original_message', '')
        prompt_id = f"{user}___{reply_to}"

        if prompt_id not in prompts_dict:
            prompts_dict[prompt_id] = Prompt(
                prompt_id=prompt_id,
                user=user,
                text=original_msg
            )

    return list(users_dict.values()), list(prompts_dict.values())


def convert_validation_results(validation_dir: Path, config_name: str, selection_method: str) -> Dict[str, Any]:
    """Convert legacy validation results to new format.

    Args:
        validation_dir: Directory containing validation files
        config_name: Configuration name
        selection_method: Selection method (random, ml, cosine)

    Returns:
        Dictionary with validation results
    """
    results = {}

    # Look for BERT results
    trainer_file = validation_dir / f"{config_name}_{selection_method}_validation_data_trainer_results.json"
    if trainer_file.exists():
        with open(trainer_file, 'r') as f:
            results['bert'] = json.load(f)

    # Look for confusion matrix
    cm_file = validation_dir / f"{config_name}_{selection_method}_validation_data_confusion_matrix.csv"
    if cm_file.exists():
        cm_df = pd.read_csv(cm_file)
        results['confusion_matrix'] = cm_df.values.tolist()

    # Look for feature analysis results
    feature_file = validation_dir / f"{config_name}_{selection_method}_validation_data_from_labels_median_run_agreement.csv"
    if feature_file.exists():
        feat_df = pd.read_csv(feature_file)
        results['feature_analysis'] = feat_df.to_dict('records')

    return results


def migrate_dataset(legacy_dir: str, output_dir: str, dataset: str):
    """Migrate entire dataset from legacy to consolidated format.

    Args:
        legacy_dir: Directory with legacy files (e.g., "results/bluesky")
        output_dir: Output directory for consolidated format
        dataset: Dataset name (bluesky, twitter, reddit)
    """
    legacy_path = Path(legacy_dir)
    output_path = Path(output_dir)

    print(f"Migrating {dataset} from {legacy_dir} to {output_dir}...")

    # Initialize shared data manager
    shared_mgr = SharedDataManager(output_dir, dataset)

    # Collect all users and prompts
    all_users = {}
    all_prompts = {}

    # Process all JSON files
    json_files = list(legacy_path.rglob('*_optimal_response.json'))
    if not json_files:
        json_files = list(legacy_path.rglob('*_random_response.json'))

    print(f"Found {len(json_files)} configuration files")

    for json_file in json_files:
        print(f"  Processing {json_file.name}...")

        # Convert responses
        consolidated = convert_legacy_response_json(str(json_file))

        # Save consolidated responses
        responses_dir = output_path / dataset / 'responses'
        responses_dir.mkdir(parents=True, exist_ok=True)
        output_file = responses_dir / f"{consolidated.config_name}.json"
        consolidated.save(str(output_file))

        # Extract shared data
        users, prompts = extract_shared_data_from_legacy(str(json_file), dataset)
        for u in users:
            all_users[u.username] = u
        for p in prompts:
            all_prompts[p.prompt_id] = p

    # Save shared data
    print(f"\nSaving shared data...")
    shared_mgr.save_users(list(all_users.values()))
    shared_mgr.save_prompts(list(all_prompts.values()))

    print(f"\n✅ Migration complete!")
    print(f"  Configurations: {len(json_files)}")
    print(f"  Users: {len(all_users)}")
    print(f"  Prompts: {len(all_prompts)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python format_converter.py <legacy_dir> <output_dir> <dataset>")
        print("Example: python format_converter.py results/bluesky results_new bluesky")
        sys.exit(1)

    legacy_dir = sys.argv[1]
    output_dir = sys.argv[2]
    dataset = sys.argv[3]

    migrate_dataset(legacy_dir, output_dir, dataset)

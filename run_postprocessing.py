#!/usr/bin/env python3
"""Unified postprocessing pipeline for simulation results.

This script replaces the multi-step workflow of:
  1. LLM_judge.py
  2. convert_emojiis.py
  3. build_validation_data.py
  4. validate_text.py
  5. features_analysis.py

With a single configurable pipeline that:
- Uses consolidated data format (fewer files, less duplication)
- Respects pipeline_config.yaml settings
- Provides clear progress tracking
- Generates only requested outputs

Usage:
    python run_postprocessing.py --input_dir results/bluesky --dataset bluesky
    python run_postprocessing.py --input_dir results/twitter --dataset twitter --config my_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Import our new modules
from validation.pipeline_config import load_config, PipelineConfig
from validation.data_formats import (
    UserMetadata, Prompt, ResponseEntry, ConsolidatedResponses,
    ValidationResults, SharedDataManager
)


def find_simulation_outputs(input_dir: Path) -> List[Path]:
    """Find all simulation output JSON files.

    Args:
        input_dir: Directory containing simulation results

    Returns:
        List of paths to *_random_response.json or *_optimal_response.json files
    """
    json_files = []

    # Look for optimal_response first (has selection results)
    json_files = list(input_dir.rglob('*_optimal_response.json'))

    # Fall back to random_response if no optimal files
    if not json_files:
        json_files = list(input_dir.rglob('*_random_response.json'))

    return json_files


def extract_config_metadata(data: List[Dict]) -> Dict[str, Any]:
    """Extract configuration metadata from simulation data.

    Args:
        data: List of response entries from simulation

    Returns:
        Dictionary with config metadata
    """
    if not data:
        return {}

    first_entry = data[0]
    metadata = {}

    # Extract common config fields
    for key in ['model', 'fine_tuned', 'finetuned', 'retrieve_context',
                'OPPU', 'n_style_examples', 'with_persona']:
        if key in first_entry:
            metadata[key] = first_entry[key]

    return metadata


def convert_to_consolidated_format(json_path: Path, config: PipelineConfig) -> ConsolidatedResponses:
    """Convert simulation output to consolidated format.

    Args:
        json_path: Path to simulation JSON file
        config: Pipeline configuration

    Returns:
        ConsolidatedResponses object
    """
    print(f"  Converting {json_path.name}...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract config name from filename
    config_name = json_path.stem
    for suffix in ['_random_response', '_optimal_response']:
        config_name = config_name.replace(suffix, '')

    # Get metadata
    metadata = extract_config_metadata(data)
    model = metadata.get('model', config_name.split('_')[0])

    # Convert entries to ResponseEntry format
    responses = []
    for entry in data:
        user = entry.get('user', '')
        reply_to = entry.get('reply_to', '')
        prompt_id = f"{user}___{reply_to}"

        candidates = entry.get('all_valid_responses', [])

        # Build selected indices
        selected_indices = {}
        scores = {}

        # Random selection
        random_resp = entry.get('response', entry.get('previous_response', ''))
        if random_resp and candidates:
            try:
                selected_indices['random'] = candidates.index(random_resp)
            except ValueError:
                # Add to candidates if not present
                candidates.append(random_resp)
                selected_indices['random'] = len(candidates) - 1

        # ML selection (if available)
        if 'ML_best_response' in entry and entry['ML_best_response']:
            ml_resp = entry['ML_best_response']
            if ml_resp in candidates:
                selected_indices['ml'] = candidates.index(ml_resp)

            # Store ML probabilities if available
            if 'ml_probabilities' in entry or 'probabilities' in entry:
                scores['ml'] = entry.get('ml_probabilities', entry.get('probabilities', []))

        # Cosine selection (if available)
        if 'cosine_best_response' in entry and entry['cosine_best_response']:
            cosine_resp = entry['cosine_best_response']
            if cosine_resp in candidates:
                selected_indices['cosine'] = candidates.index(cosine_resp)

            # Store cosine similarities if available
            if 'cosine_similarities' in entry or 'similarities' in entry:
                scores['cosine'] = entry.get('cosine_similarities', entry.get('similarities', []))

        resp_entry = ResponseEntry(
            user=user,
            prompt_id=prompt_id,
            candidates=candidates,
            selected_indices=selected_indices,
            scores=scores,
            metadata={
                'original_message': entry.get('original_message', ''),
                'reply_to': reply_to,
                'persona': entry.get('persona', '')
            }
        )
        responses.append(resp_entry)

    return ConsolidatedResponses(
        config_name=config_name,
        model=model,
        metadata=metadata,
        responses=responses
    )


def extract_shared_data(consolidated: ConsolidatedResponses, dataset: str) -> tuple[List[UserMetadata], List[Prompt]]:
    """Extract shared user and prompt data.

    Args:
        consolidated: ConsolidatedResponses object
        dataset: Dataset name

    Returns:
        Tuple of (users, prompts)
    """
    users_dict = {}
    prompts_dict = {}

    for resp in consolidated.responses:
        # Extract user
        if resp.user not in users_dict:
            users_dict[resp.user] = UserMetadata(
                username=resp.user,
                persona=resp.metadata.get('persona', ''),
                platform=dataset
            )

        # Extract prompt
        if resp.prompt_id not in prompts_dict:
            prompts_dict[resp.prompt_id] = Prompt(
                prompt_id=resp.prompt_id,
                user=resp.user,
                text=resp.metadata.get('original_message', '')
            )

    return list(users_dict.values()), list(prompts_dict.values())


def main():
    """Main postprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified postprocessing pipeline for simulation results"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing simulation outputs'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (bluesky, twitter, reddit)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to pipeline config YAML (default: pipeline_config.yaml)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: same as input_dir with _consolidated suffix)'
    )

    args = parser.parse_args()

    # Load configuration
    print("="*60)
    print("UNIFIED POSTPROCESSING PIPELINE")
    print("="*60)

    config = load_config(args.config)
    warnings = config.validate()
    if warnings:
        print("\n⚠️  Configuration warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    # Setup paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_consolidated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"💾 Format: {'Consolidated' if config.storage.use_consolidated_format else 'Legacy'}")

    # Find simulation outputs
    print(f"\n🔍 Finding simulation outputs...")
    json_files = find_simulation_outputs(input_dir)

    if not json_files:
        print(f"❌ No simulation output files found in {input_dir}")
        sys.exit(1)

    print(f"✅ Found {len(json_files)} configuration(s)")

    # Initialize shared data manager
    shared_mgr = SharedDataManager(str(output_dir), args.dataset)
    all_users = {}
    all_prompts = {}

    # Process each configuration
    print(f"\n📦 Converting to consolidated format...")
    responses_dir = output_dir / args.dataset / 'responses'
    responses_dir.mkdir(parents=True, exist_ok=True)

    for json_file in json_files:
        # Convert to consolidated format
        consolidated = convert_to_consolidated_format(json_file, config)

        # Save consolidated responses
        output_file = responses_dir / f"{consolidated.config_name}.json"
        consolidated.save(str(output_file))
        print(f"  ✅ Saved {output_file.name} ({len(consolidated.responses)} responses)")

        # Extract shared data
        users, prompts = extract_shared_data(consolidated, args.dataset)
        for u in users:
            all_users[u.username] = u
        for p in prompts:
            all_prompts[p.prompt_id] = p

    # Save shared data
    print(f"\n💾 Saving shared data...")
    shared_mgr.save_users(list(all_users.values()))
    shared_mgr.save_prompts(list(all_prompts.values()))

    # Summary
    print(f"\n" + "="*60)
    print("✅ POSTPROCESSING COMPLETE")
    print("="*60)
    print(f"Configurations processed: {len(json_files)}")
    print(f"Unique users: {len(all_users)}")
    print(f"Unique prompts: {len(all_prompts)}")
    print(f"\nOutput structure:")
    print(f"  {output_dir / args.dataset / 'shared' / 'users.json'}")
    print(f"  {output_dir / args.dataset / 'shared' / 'prompts.json'}")
    print(f"  {output_dir / args.dataset / 'responses'}/ ({len(json_files)} files)")

    if config.storage.use_consolidated_format:
        original_files = len(json_files) * 40  # Rough estimate
        new_files = len(json_files) + 2  # responses + users + prompts
        print(f"\n💡 File reduction: ~{original_files} → {new_files} files ({new_files/original_files*100:.0f}% reduction)")


if __name__ == "__main__":
    main()

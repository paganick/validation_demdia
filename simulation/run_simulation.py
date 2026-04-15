import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.simulate import *
from src.config_utils import load_config
import src.globals as globals_module


def load_batch_users(batch_file: str, dataset: str, batch_id: int) -> list:
    """Load users for a specific batch from the batch assignments file."""
    with open(batch_file, 'r') as f:
        assignments = json.load(f)

    if dataset not in assignments["platforms"]:
        raise ValueError(f"Dataset '{dataset}' not found in batch file")

    platform_data = assignments["platforms"][dataset]

    if batch_id >= platform_data["n_batches"]:
        raise ValueError(f"Batch {batch_id} does not exist for {dataset} (max: {platform_data['n_batches']-1})")

    return platform_data["batches"][batch_id]["users"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help="Path to a single config file")
    parser.add_argument('--config_dir', type=str, help="Folder containing multiple config files")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['twitter', 'reddit', 'bluesky'],
                        help="Dataset to process")
    parser.add_argument('--output_dir', type=str, default="results/")
    parser.add_argument('--n_responses_per_user', type=int, default=20)
    parser.add_argument('--filter_keywords', nargs='*', default=[],
                        help="List of keywords to filter config filenames")

    # Batch processing arguments
    parser.add_argument('--user_batch', type=int, default=None,
                        help="User batch index (0-indexed). If set, only process users in this batch.")
    parser.add_argument('--batch_file', type=str, default="user_batches.json",
                        help="Path to batch assignments file (default: user_batches.json)")
    parser.add_argument('--fill_incomplete', action='store_true', default=False,
                        help="Re-attempt samples with fewer than 20 valid responses (default: skip any sample already in the file)")
    parser.add_argument('--data_dir', type=str, default="data",
                        help="Root data directory containing {dataset}/posts.pkl and personas.pkl (default: data/)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all config file paths
    if args.config_file:
        config_paths = [args.config_file]
    elif args.config_dir:
        config_paths = [
            os.path.join(args.config_dir, f)
            for f in os.listdir(args.config_dir)
            if f.endswith(".yaml") or f.endswith(".yml") or f.endswith(".json")
        ]
        # Filter based on keywords if provided
        if args.filter_keywords:
            config_paths = [
                path for path in config_paths
                if all(kw.lower() in os.path.basename(path).lower() for kw in args.filter_keywords)
            ]
    else:
        raise ValueError("You must provide either --config_file or --config_dir")

    # Data file paths (separate posts and personas files)
    posts_file = os.path.join(args.data_dir, args.dataset, 'posts.pkl')
    personas_file = os.path.join(args.data_dir, args.dataset, 'personas.pkl')

    print(f"📁 Dataset: {args.dataset}")
    print(f"   Posts: {posts_file}")
    print(f"   Personas: {personas_file}")

    # Load batch users if batch processing is enabled
    batch_users = None
    if args.user_batch is not None:
        batch_users = load_batch_users(args.batch_file, args.dataset, args.user_batch)
        print(f"📦 Batch {args.user_batch}: {len(batch_users)} users")

    for cfg_path in config_paths:
        config = load_config(cfg_path)

        # Set defaults
        _default_ft_dir = os.path.join(
            os.environ.get("SCRATCH", "."), "finetuned_models"
        ) if "SCRATCH" in os.environ else os.environ.get("FINETUNING_DIR", "finetuned_models/")
        config.setdefault("finetuning_dir", os.environ.get("FINETUNING_DIR", _default_ft_dir))
        config.setdefault("instruction_tuned", False)
        config.setdefault("persona", True)
        config.setdefault("n_style_examples", 0)
        config.setdefault("retrieve_context", False)
        config.setdefault("finetuned", False)

        # Generate output filename
        filename_parts = [
            config["model"].replace("/", "_"),
            "ft" if config["finetuned"] else "noft",
            f"ctx{int(config['retrieve_context'])}",
            f"style{config['n_style_examples']}",
        ]
        output_filename = "__".join(filename_parts)
        if not config['persona']:
            output_filename += "__no_persona"

        if config.get("ft_variant"):
            output_filename += f"__{config['ft_variant']}"

        # Add batch suffix if batch processing
        if args.user_batch is not None:
            output_filename += f"__batch{args.user_batch}"

        output_filename += ".json"
        output_path = os.path.join(args.output_dir, output_filename)

        print(f"\n[RUNNING] Config: {cfg_path}")
        print(f"[OUTPUT] {output_path}")

        # Determine n_users based on batch or full dataset
        if batch_users is not None:
            n_users = len(batch_users)
        else:
            # If no batch specified, process all users
            import pandas as pd
            df_personas = pd.read_pickle(personas_file)
            n_users = len(df_personas)

        results = run_simulation_random_response(
            config, args.dataset, posts_file, personas_file,
            n_users=n_users,
            n_responses_per_user=args.n_responses_per_user,
            output_path=output_path,
            batch_users=batch_users,
            fill_incomplete=args.fill_incomplete,
        )


if __name__ == "__main__":
    main()

import argparse
import os
import json
from src.simulate import *
from src.config_utils import load_config
import src.globals as globals_module


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to a single config file")
    parser.add_argument('--config_dir', type=str, help="Folder containing multiple config files")
    parser.add_argument('--data_file', type=str, help="Path to a data file")
    parser.add_argument('--output_dir', type=str, default="simulations/")
    parser.add_argument('--n_users', type=int, default=250)
    parser.add_argument('--n_responses_per_user', type=int, default=20)
    parser.add_argument('--filter_keywords', nargs='*', default=[],
                        help="List of keywords to filter config filenames (e.g. --filter_keywords deepseek withcontext)")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiment_type not in ["random_response", "political_affiliation"]:
        raise ValueError("You must provide a valid experiment_type between political_affiliation and random_response")

    # Collect all config file paths
    if args.config:
        config_paths = [args.config]
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
        raise ValueError("You must provide either --config or --config_dir")

    if args.data_file:
        data_file = args.data_file
    else:
       raise ValueError("You must provide either a data file using --data_file")

    for cfg_path in config_paths:
        config = load_config(cfg_path)

        # Set defaults (can also be enforced in config loader)
        config.setdefault("finetuning_dir", "/home/nicpag/scratch/finetuned_models/") # NEEDS A LARGE STORAGE SPACE
        config.setdefault("finetuned", False)
        config.setdefault("retrieve_context", False)
        config.setdefault("n_style_examples", 0)
        config.setdefault("OPPU", False)

        # Generate an output filename based on config values
        filename_parts = [
            config["model"],
            "ft" if config["finetuned"] else "noft",
            f"ctx{int(config['retrieve_context'])}",
            f"style{config['n_style_examples']}",
            "OPPU" if config["OPPU"] else "no_OPPU"
        ]
        output_filename = "__".join(filename_parts) 
        output_filename += "__random_response.json"
        output_path = os.path.join(args.output_dir, output_filename)
        print(output_filename)
        print(output_path)

        print(f"[RUNNING] Config: {cfg_path}")
       
        results = run_simulation_random_response(config, data_file, n_users=args.n_users, 
                                             n_responses_per_user= args.n_responses_per_user, 
                                             output_path=output_path)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[DONE] Results saved to {output_path}")

if __name__ == "__main__":
    main()

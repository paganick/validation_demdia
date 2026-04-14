#!/usr/bin/env python3
"""
Join complete batched simulation results into consolidated files.

For each (platform, config) combination where ALL batch files are present under
results/, merges them into a single JSON file at:

    {output_dir}/{platform}/{vendor}/{ModelName}__{config_flags}__random_response.json

A config is considered complete when every expected batch file exists.
Incomplete configs (missing any batch) are skipped and reported in the summary.

Re-running is safe — already-joined files are skipped unless --force is given.

Usage:
    python join_complete_batches.py                            # default: results_joined/
    python join_complete_batches.py --output-dir results_v2   # custom output dir
    python join_complete_batches.py --dry-run                  # preview only
    python join_complete_batches.py --force                    # re-join existing
    python join_complete_batches.py --list-tasks               # print TSV of all jobs

    # Test / custom layout:
    python join_complete_batches.py \\
        --results-dir tests/results \\
        --configs-dir tests/configs \\
        --batch-file  tests/user_batches_test.json \\
        --output-dir  tests/results_joined
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

BASE_DIR  = Path(__file__).resolve().parent
PLATFORMS = ["bluesky", "twitter", "reddit"]


# ---------------------------------------------------------------------------
# Slug construction — must mirror run_simulation.py exactly
# ---------------------------------------------------------------------------

def config_to_slug(config: dict) -> str:
    """Build the batch filename stem from a config dict."""
    parts = [
        config["model"].replace("/", "_"),
        "ft" if config.get("finetuned", False) else "noft",
        f"ctx{int(config.get('retrieve_context', False))}",
        f"style{config.get('n_style_examples', 0)}",
    ]
    slug = "__".join(parts)
    if not config.get("persona", True):
        slug += "__no_persona"
    if config.get("ft_variant"):
        slug += f"__{config['ft_variant']}"
    return slug


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_configs(config_dir: Path) -> list:
    """Load all YAML configs from config_dir and return list of info dicts."""
    configs = []
    for cfg_path in sorted(config_dir.glob("*.yaml")):
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        config.setdefault("finetuned", False)
        config.setdefault("persona", True)
        config.setdefault("n_style_examples", 0)
        config.setdefault("retrieve_context", False)
        configs.append({
            "filename": cfg_path.name,
            "config": config,
            "slug": config_to_slug(config),
        })
    return configs


def load_batch_counts(batch_file: Path) -> dict:
    """Return {platform: n_batches} from the batch assignments file."""
    with open(batch_file) as f:
        data = json.load(f)
    return {
        platform: data["platforms"][platform]["n_batches"]
        for platform in PLATFORMS
    }


def load_json_safe(path: Path) -> list | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def parse_model_info(config: dict) -> tuple[str, str]:
    """Split 'vendor/ModelName' into (vendor, ModelName)."""
    model = config["model"]
    if "/" in model:
        vendor, model_name = model.split("/", 1)
    else:
        vendor, model_name = "unknown", model
    return vendor, model_name


def build_output_filename(slug: str, vendor: str, model_name: str) -> str:
    """Strip the vendor_modelname__ prefix from the slug and append __random_response.json."""
    prefix = f"{vendor}_{model_name}__"
    config_flags = slug[len(prefix):] if slug.startswith(prefix) else slug.split("__", 1)[-1]
    return f"{model_name}__{config_flags}__random_response.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge complete batched simulation results into consolidated files."
    )
    parser.add_argument(
        "--output-dir", default="results_joined",
        help="Output directory (default: results_joined/)"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory containing per-platform batch files (default: results/)"
    )
    parser.add_argument(
        "--configs-dir", default="configs",
        help="Directory containing YAML config files (default: configs/)"
    )
    parser.add_argument(
        "--batch-file", default="user_batches.json",
        help="Batch assignments JSON file (default: user_batches.json)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be joined without writing any files"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-join even if the output file already exists"
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help=(
            "Print a TSV of all (config, platform, batch_id) jobs to stdout and exit. "
            "Redirect to a file to use as input for a SLURM array job: "
            "python join_complete_batches.py --list-tasks > tasks.tsv"
        )
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config_dir  = Path(args.configs_dir)
    batch_file  = Path(args.batch_file)

    configs      = load_configs(config_dir)
    batch_counts = load_batch_counts(batch_file)

    # --list-tasks: enumerate all jobs and exit
    if args.list_tasks:
        for platform in PLATFORMS:
            n_batches = batch_counts[platform]
            for c in configs:
                for batch_id in range(n_batches):
                    print(f"{c['filename']}\t{platform}\t{batch_id}")
        return

    output_dir = Path(args.output_dir)

    n_joined     = 0
    n_skipped    = 0
    n_incomplete = 0

    for platform in PLATFORMS:
        n_batches = batch_counts[platform]

        for c in configs:
            slug = c["slug"]

            # Check all batch files exist
            batch_paths = [
                results_dir / platform / f"{slug}__batch{b}.json"
                for b in range(n_batches)
            ]
            missing = [p for p in batch_paths if not p.exists()]
            if missing:
                n_incomplete += 1
                continue

            # Build output path
            vendor, model_name = parse_model_info(c["config"])
            out_path = (
                output_dir / platform / vendor
                / build_output_filename(slug, vendor, model_name)
            )

            if out_path.exists() and not args.force:
                n_skipped += 1
                continue

            if args.dry_run:
                print(f"[DRY RUN] {n_batches} batches → {out_path}")
                n_joined += 1
                continue

            # Merge and write
            merged = []
            for p in batch_paths:
                data = load_json_safe(p)
                if data is not None:
                    merged.extend(data)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"Joined {n_batches} batches ({len(merged)} entries) → {out_path}")
            n_joined += 1

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Newly joined:    {n_joined}", file=sys.stderr)
    print(f"  Already existed: {n_skipped} (skipped, use --force to redo)", file=sys.stderr)
    print(f"  Incomplete:      {n_incomplete} (not all batch files present)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()

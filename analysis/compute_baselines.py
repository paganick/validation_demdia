#!/usr/bin/env python3
"""
compute_baselines.py

Compute cosine-similarity baseline distributions from *_optimal_response.json
files produced by LLM_judge.py, plus optional human-message baselines from
personas.pkl files.

Outputs
-------
<output-dir>/main_scores.csv
    One row per entry. Columns:
    platform, model, config, user, entry_id, sim_random, sim_ml_best, sim_cosine_best

<output-dir>/baselines.csv
    Long format. Columns:
    platform, model, config, baseline_type, sim
    baseline_type values:
        intra_ai            - pairs from same entry's candidate pool
        random_ai_ai_random - random-response pairs across entries
        random_ai_ai_ml     - ML-best pairs across entries
        random_ai_ai_cosine - cosine-best pairs across entries
        intra_human         - same-user training-message pairs
        random_human_human  - cross-user training-message pairs

Usage
-----
python compute_baselines.py \\
    --optimal-dir results_optimal/ \\
    --platform-data bluesky:/path/to/bluesky_data/personas.pkl \\
    --output-dir baselines/ \\
    --n-pairs 5 --n-random-pairs 1000 --seed 42
"""

import argparse
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D arrays, epsilon-guarded."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def sample_pairs(n: int, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Return up to k unique unordered pairs from range(n)."""
    all_pairs = list(combinations(range(n), 2))
    k = min(k, len(all_pairs))
    if k == 0:
        return []
    chosen_idx = rng.choice(len(all_pairs), size=k, replace=False)
    return [all_pairs[i] for i in chosen_idx]


def make_entry_id(reply_to: str) -> str:
    """sha256(reply_to)[:8] hex — compact unique handle for the parent message."""
    return hashlib.sha256(reply_to.encode()).hexdigest()[:8]


def parse_path(fpath: Path) -> tuple[str, str, str, str]:
    """Extract (platform, vendor, model_name, config) from file path.

    Expects path structure:
        .../platform/vendor/ModelName__config__optimal_response.json
    """
    platform = fpath.parts[-3]
    vendor = fpath.parts[-2]
    stem = fpath.stem  # e.g. Llama-3.1-8B__ft__ctx1__style10__optimal_response
    base = stem.replace("__optimal_response", "")  # Llama-3.1-8B__ft__ctx1__style10
    model_name, config = base.split("__", 1)
    return platform, vendor, model_name, config


def get_candidate_text(candidate) -> str:
    """Return text from a candidate that is either a str or a dict with 'response'."""
    if isinstance(candidate, dict):
        return candidate.get("response", "") or ""
    return candidate or ""


# ---------------------------------------------------------------------------
# Main scores per entry
# ---------------------------------------------------------------------------

MAIN_FIELDNAMES = [
    "platform", "model", "config", "user", "entry_id",
    "sim_random", "sim_ml_best", "sim_cosine_best",
]

BASELINE_FIELDNAMES = ["platform", "model", "config", "baseline_type", "sim"]


def process_optimal_file(
    fpath: Path,
    st_model: SentenceTransformer,
    n_pairs: int,
    n_random_pairs: int,
    rng: np.random.Generator,
    main_writer: csv.DictWriter,
    baseline_writer: csv.DictWriter,
) -> None:
    """Process one *_optimal_response.json file."""
    try:
        platform, vendor, model_name, config = parse_path(fpath)
    except (IndexError, ValueError) as exc:
        print(f"  [WARN] Cannot parse path {fpath}: {exc} — skipping", file=sys.stderr)
        return

    model = f"{vendor}/{model_name}"

    with open(fpath, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"  [WARN] Cannot parse JSON {fpath}: {exc} — skipping", file=sys.stderr)
            return

    # Filter entries without an original message
    entries = [e for e in data if e.get("original_message")]
    if not entries:
        print(f"  [WARN] No valid entries in {fpath} — skipping", file=sys.stderr)
        return

    # ------------------------------------------------------------------
    # Collect all unique non-empty texts for batch encoding
    # ------------------------------------------------------------------
    texts_set: set[str] = set()
    for e in entries:
        texts_set.add(e["original_message"])
        for field in ("response", "ML_best_response", "cosine_best_response"):
            val = e.get(field, "")
            if val:
                texts_set.add(val)
        for c in e.get("all_valid_responses", []):
            t = get_candidate_text(c)
            if t:
                texts_set.add(t)

    texts_list = list(texts_set)
    embs = st_model.encode(texts_list, batch_size=512, show_progress_bar=False)
    emb_cache: dict[str, np.ndarray] = dict(zip(texts_list, embs))

    # ------------------------------------------------------------------
    # 1. Main scores — one row per entry
    # ------------------------------------------------------------------
    for e in entries:
        orig_emb = emb_cache[e["original_message"]]
        entry_id = make_entry_id(str(e.get("reply_to", "")))
        user = e.get("user", "")

        sims = {}
        for field, col in [
            ("response",             "sim_random"),
            ("ML_best_response",     "sim_ml_best"),
            ("cosine_best_response", "sim_cosine_best"),
        ]:
            val = e.get(field, "")
            if val and val in emb_cache:
                sims[col] = cosine_sim(emb_cache[val], orig_emb)
            else:
                sims[col] = float("nan")

        main_writer.writerow({
            "platform":       platform,
            "model":          model,
            "config":         config,
            "user":           user,
            "entry_id":       entry_id,
            "sim_random":     sims["sim_random"],
            "sim_ml_best":    sims["sim_ml_best"],
            "sim_cosine_best": sims["sim_cosine_best"],
        })

    # ------------------------------------------------------------------
    # 2. Intra-AI baseline — pairs within each entry's candidate pool
    # ------------------------------------------------------------------
    for e in entries:
        candidates = [
            get_candidate_text(c)
            for c in e.get("all_valid_responses", [])
        ]
        candidates = [t for t in candidates if t and t in emb_cache]
        if len(candidates) < 2:
            continue
        c_embs = np.array([emb_cache[t] for t in candidates])
        for i, j in sample_pairs(len(candidates), n_pairs, rng):
            baseline_writer.writerow({
                "platform":      platform,
                "model":         model,
                "config":        config,
                "baseline_type": "intra_ai",
                "sim":           cosine_sim(c_embs[i], c_embs[j]),
            })

    # ------------------------------------------------------------------
    # 3. Random AI-AI baselines — cross-entry pairs per response type
    # ------------------------------------------------------------------
    for field, label in [
        ("response",             "random"),
        ("ML_best_response",     "ml"),
        ("cosine_best_response", "cosine"),
    ]:
        pool = [
            (idx, emb_cache[e[field]])
            for idx, e in enumerate(entries)
            if e.get(field) and e[field] in emb_cache
        ]
        if len(pool) < 2:
            continue

        baseline_type = f"random_ai_ai_{label}"
        n_written = 0
        max_attempts = n_random_pairs * 5
        attempts = 0
        while n_written < n_random_pairs and attempts < max_attempts:
            attempts += 1
            i, j = rng.integers(0, len(pool), size=2)
            if pool[i][0] != pool[j][0]:  # different entries (different contexts)
                baseline_writer.writerow({
                    "platform":      platform,
                    "model":         model,
                    "config":        config,
                    "baseline_type": baseline_type,
                    "sim":           cosine_sim(pool[i][1], pool[j][1]),
                })
                n_written += 1


# ---------------------------------------------------------------------------
# Human baselines (optional, per platform)
# ---------------------------------------------------------------------------

def compute_human_baselines(
    platform: str,
    personas_pkl: Path,
    st_model: SentenceTransformer,
    n_pairs: int,
    n_random_pairs: int,
    rng: np.random.Generator,
    baseline_writer: csv.DictWriter,
    subsample: int = 0,
) -> None:
    """Compute intra_human and random_human_human baselines from a posts pkl."""
    print(f"  Loading data from {personas_pkl} ...", flush=True)
    df = pd.read_pickle(personas_pkl)

    if "training" not in df.columns:
        print(f"  [WARN] No 'training' column in {personas_pkl} — skipping human baselines", file=sys.stderr)
        return
    if "message" not in df.columns:
        print(f"  [WARN] No 'message' column in {personas_pkl} — skipping human baselines", file=sys.stderr)
        return

    train = df[df["training"] == 1].copy()
    train = train[train["message"].notna() & (train["message"].str.strip() != "")]
    if train.empty:
        print(f"  [WARN] No training messages found in {personas_pkl}", file=sys.stderr)
        return

    if subsample and len(train) > subsample:
        train = train.sample(n=subsample, random_state=int(rng.integers(1 << 31)))
        print(f"  Subsampled to {subsample:,} messages.", flush=True)

    texts = train["message"].tolist()
    print(f"  Encoding {len(texts):,} training messages for platform={platform} ...", flush=True)
    embs = st_model.encode(texts, batch_size=512, show_progress_bar=True)

    # Intra-human: per-user pairs
    if "username" not in train.columns:
        print(f"  [WARN] No 'username' column — skipping intra_human baseline", file=sys.stderr)
    else:
        train = train.reset_index(drop=True)
        for user, group in train.groupby("username"):
            idxs = group.index.tolist()
            if len(idxs) < 2:
                continue
            for i, j in sample_pairs(len(idxs), n_pairs, rng):
                baseline_writer.writerow({
                    "platform":      platform,
                    "model":         "",
                    "config":        "",
                    "baseline_type": "intra_human",
                    "sim":           cosine_sim(embs[idxs[i]], embs[idxs[j]]),
                })

    # Random human-human: cross-user pairs
    if "username" in train.columns:
        user_of = train["username"].tolist()
    else:
        user_of = [""] * len(texts)

    n_written = 0
    max_attempts = n_random_pairs * 5
    attempts = 0
    while n_written < n_random_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, len(texts), size=2)
        if user_of[i] != user_of[j]:
            baseline_writer.writerow({
                "platform":      platform,
                "model":         "",
                "config":        "",
                "baseline_type": "random_human_human",
                "sim":           cosine_sim(embs[i], embs[j]),
            })
            n_written += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosine-similarity baseline distributions from *_optimal_response.json files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compute_baselines.py \\
      --optimal-dir results_optimal/ \\
      --platform-data bluesky:/path/to/bluesky_data/personas.pkl \\
      --output-dir baselines/ \\
      --n-pairs 5 --n-random-pairs 1000 --seed 42
""",
    )
    parser.add_argument(
        "--optimal-dir", required=True,
        help="Root directory; recursively scanned for *_optimal_response.json files.",
    )
    parser.add_argument(
        "--platform-data", action="append", default=[], metavar="PLATFORM:PATH",
        help="PLATFORM:path to a posts/personas pkl with (username, message, training) columns. "
             "Repeat the flag for each platform.",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name or local path (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=10,
        help="Intra-* pairs per entry/user (default: 10).",
    )
    parser.add_argument(
        "--n-random-pairs", type=int, default=10_000,
        help="Random-* pairs per file (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Embedding batch size (default: 512).",
    )
    parser.add_argument(
        "--output-dir", default="baselines",
        help="Output directory (default: baselines/).",
    )
    parser.add_argument(
        "--human-only", action="store_true",
        help="Skip AI files entirely; only compute human baselines from --platform-data.",
    )
    parser.add_argument(
        "--subsample", type=int, default=0, metavar="N",
        help="If > 0, randomly subsample N training messages per platform (useful for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse --platform-data PLATFORM:path pairs
    platform_data: dict[str, Path] = {}
    for spec in args.platform_data:
        try:
            plat, path_str = spec.split(":", 1)
        except ValueError:
            print(f"[ERROR] --platform-data value '{spec}' must be PLATFORM:path", file=sys.stderr)
            sys.exit(1)
        platform_data[plat] = Path(path_str)

    # Discover input files
    json_files: list[Path] = []
    if not args.human_only:
        optimal_dir = Path(args.optimal_dir)
        json_files = sorted(optimal_dir.rglob("*_optimal_response.json"))
        if not json_files:
            print(f"[ERROR] No *_optimal_response.json files found under {optimal_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(json_files)} *_optimal_response.json file(s).")
    else:
        if not platform_data:
            print("[ERROR] --human-only requires at least one --platform-data entry.", file=sys.stderr)
            sys.exit(1)
        print("--human-only mode: skipping AI files.")

    # Load sentence-transformer model once
    print(f"Loading sentence-transformer model: {args.model} ...", flush=True)
    st_model = SentenceTransformer(args.model)

    rng = np.random.default_rng(args.seed)

    # Prepare output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_scores_path = output_dir / "main_scores.csv"
    baselines_path = output_dir / "baselines.csv"

    with (
        open(main_scores_path, "w", newline="", encoding="utf-8") as main_f,
        open(baselines_path, "w", newline="", encoding="utf-8") as baseline_f,
    ):
        main_writer = csv.DictWriter(main_f, fieldnames=MAIN_FIELDNAMES)
        main_writer.writeheader()

        baseline_writer = csv.DictWriter(baseline_f, fieldnames=BASELINE_FIELDNAMES)
        baseline_writer.writeheader()

        # Process AI files
        for fpath in tqdm(json_files, desc="Processing files"):
            process_optimal_file(
                fpath, st_model,
                args.n_pairs, args.n_random_pairs,
                rng, main_writer, baseline_writer,
            )

        # Human baselines (once per platform)
        seen_platforms: set[str] = set()
        # In --human-only mode, derive platforms directly from --platform-data keys
        platforms_to_process = (
            list(platform_data.keys()) if args.human_only
            else [fpath.parts[-3] for fpath in json_files if len(fpath.parts) >= 3]
        )
        for platform in platforms_to_process:
            if platform in seen_platforms:
                continue
            seen_platforms.add(platform)
            if platform in platform_data:
                print(f"Computing human baselines for platform={platform} ...", flush=True)
                compute_human_baselines(
                    platform, platform_data[platform],
                    st_model, args.n_pairs, args.n_random_pairs,
                    rng, baseline_writer,
                    subsample=args.subsample,
                )
            else:
                print(f"[INFO] No personas.pkl provided for platform={platform} — skipping human baselines.")

    print(f"\nDone.")
    print(f"  main_scores.csv → {main_scores_path}")
    print(f"  baselines.csv   → {baselines_path}")


if __name__ == "__main__":
    main()

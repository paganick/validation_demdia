#!/usr/bin/env python3
"""
Compute cosine similarity baseline distributions for reviewer comparison.

For each platform and SOTA config (baseline + persona), computes 5 distributions:
  1. human_vs_ai     — cosine(selected_AI_response, human_ground_truth)   [existing]
  2. intra_ai        — cosine between pairs of AI candidates for SAME context
  3. intra_human     — cosine between pairs of messages from the SAME user
  4. random_human    — cosine between RANDOM pairs of human messages (cross-user)
  5. random_ai       — cosine between RANDOM pairs of AI responses (cross-context)

All distributions are sampled to the same size (n = number of test entries per platform)
for fair comparison. Pass --skip-random to drop distributions 4-5 (random_human,
random_ai) — this also avoids encoding the full training-message pool (only needed
for random_human), which is the main cost driver for configs with high candidate
diversity (e.g. retrieval/style-conditioned or fine-tuned models can 10x+ the unique
text count vs the SOTA config, since fewer candidates collapse to duplicates).

Output: cosine_baselines/cosine_baselines_{platform}.csv and cosine_baselines_all.csv
        (default location relative to the results folder; override with --output-dir)

This script is a prerequisite for plot_same_context_similarity.py and
plot_config_comparison.py. Run it once after LLM_judge.py completes, before
running those scripts.

Usage:
    python compute_cosine_baselines.py <results_cleaned_folder> [--output-dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import json
import glob
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path

SENTENCE_MODEL_PATH = os.environ.get("SENTENCE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
RANDOM_SEED = 42

# SOTA config filter: noft, ctx0, style0, with persona (not no_persona, not ctx1, not style10)
def is_sota_config(fname):
    fname = os.path.basename(fname)
    return (
        "__noft__" in fname
        and "__ctx0__" in fname
        and "__style0__" in fname
        and "no_persona" not in fname
        and "__no_persona" not in fname
    )


def cosine_matrix(embeddings):
    """Return NxN cosine similarity matrix for a set of embeddings (already L2-normalised)."""
    return embeddings @ embeddings.T


def batch_encode(model, texts, batch_size=512):
    """Encode texts in batches, return L2-normalised embeddings as numpy array."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False,
                            normalize_embeddings=True)
        all_embs.append(embs)
    return np.vstack(all_embs)


def sample_pairs(idx_a, idx_b, n, rng):
    """Sample n random (i, j) index pairs where i comes from idx_a and j from idx_b."""
    pairs = [(rng.choice(idx_a), rng.choice(idx_b)) for _ in range(n)]
    return pairs


def compute_platform_baselines(platform_folder, posts_pkl, model, rng, config_filter=None, skip_random=False):
    """
    Return a DataFrame with columns [platform, distribution, similarity]
    for the five baseline distributions.
    """
    platform = os.path.basename(platform_folder)
    print(f"\n  Platform: {platform}")

    # ------------------------------------------------------------------ #
    # 1. Load all matching-config optimal_response.json entries            #
    # ------------------------------------------------------------------ #
    opt_files = glob.glob(
        str(Path(platform_folder) / "**" / "*_optimal_response.json"), recursive=True
    )
    if config_filter:
        opt_files = [f for f in opt_files if config_filter in os.path.basename(f)]
    else:
        opt_files = [f for f in opt_files if is_sota_config(f)]
    print(f"  Matching optimal_response files: {len(opt_files)}")

    entries = []
    for fpath in opt_files:
        with open(fpath) as f:
            data = json.load(f)
        entries.extend(data)

    if not entries:
        print("  No entries found, skipping.")
        return None

    n = len(entries)
    print(f"  Total entries: {n}")

    # ------------------------------------------------------------------ #
    # 2. Collect texts                                                     #
    # ------------------------------------------------------------------ #
    # Per-entry: human ground truth and all 20 AI candidates
    human_texts   = [e["original_message"] for e in entries]
    # selected random response per entry (for human-vs-AI)
    ai_texts      = [e["response"] for e in entries]
    # list of lists: 20 candidates per entry
    candidates    = [[c["response"] for c in e["all_valid_responses"]] for e in entries]

    # Human training messages per user (for intra-human baseline)
    with open(posts_pkl, "rb") as f:
        posts_df = pickle.load(f)
    posts_df = posts_df.reset_index(drop=True)
    training_msgs = (
        posts_df[posts_df["training"] == 1]
        .groupby("username")["message"]
        .apply(list)
        .to_dict()
    ) if "username" in posts_df.columns else (
        posts_df[posts_df["training"] == 1]
        .groupby("user")["message"]
        .apply(list)
        .to_dict()
    )
    # Keep only users with ≥2 messages
    multi_msg_users = {u: msgs for u, msgs in training_msgs.items() if len(msgs) >= 2}
    print(f"  Users with ≥2 training messages: {len(multi_msg_users)}")

    # ------------------------------------------------------------------ #
    # 3. Encode everything                                                 #
    # ------------------------------------------------------------------ #
    # Unique texts to encode (deduplicate for efficiency)
    flat_candidates = [c for cands in candidates for c in cands]

    if skip_random:
        # random_human is the only consumer of the full training-message pool (every
        # user, not just those with ≥2 messages) — skip it entirely when not needed.
        all_human_pool = None
        multi_msg_texts = [m for msgs in multi_msg_users.values() for m in msgs]
        unique_texts = list(set(human_texts + ai_texts + flat_candidates + multi_msg_texts))
    else:
        all_human_pool = list(posts_df[posts_df["training"] == 1]["message"].dropna().astype(str))
        unique_texts = list(set(human_texts + ai_texts + flat_candidates + all_human_pool))
    print(f"  Encoding {len(unique_texts)} unique texts...")
    all_embs = batch_encode(model, unique_texts)
    text2idx = {t: i for i, t in enumerate(unique_texts)}

    def emb(text):
        return all_embs[text2idx[text]]

    # ------------------------------------------------------------------ #
    # 4. Distribution 1: human_vs_ai (one value per entry)                #
    # ------------------------------------------------------------------ #
    hva_sims = []
    for i in range(n):
        h = emb(human_texts[i])
        a = emb(ai_texts[i])
        hva_sims.append(float(h @ a))

    # ------------------------------------------------------------------ #
    # 5. Distribution 2: intra_ai (same context, pairs of candidates)     #
    # ------------------------------------------------------------------ #
    intra_ai_sims = []
    for cands in candidates:
        if len(cands) < 2:
            continue
        # Sample 1 random pair per entry (gives n values total)
        idx1, idx2 = rng.sample(range(len(cands)), 2)
        e1, e2 = emb(cands[idx1]), emb(cands[idx2])
        intra_ai_sims.append(float(e1 @ e2))
    # trim/pad to n
    intra_ai_sims = intra_ai_sims[:n]

    # ------------------------------------------------------------------ #
    # 6. Distribution 3: intra_human (same user, pairs of messages)       #
    # ------------------------------------------------------------------ #
    intra_human_sims = []
    user_list = list(multi_msg_users.keys())
    while len(intra_human_sims) < n and user_list:
        user = rng.choice(user_list)
        msgs = multi_msg_users[user]
        m1, m2 = rng.sample(msgs, 2)
        if m1 in text2idx and m2 in text2idx:
            intra_human_sims.append(float(emb(m1) @ emb(m2)))
    intra_human_sims = intra_human_sims[:n]

    # ------------------------------------------------------------------ #
    # 7-8. Distributions 4-5: random_human / random_ai (skippable)        #
    # ------------------------------------------------------------------ #
    labels = {
        "human_vs_ai":    hva_sims,
        "intra_ai":       intra_ai_sims,
        "intra_human":    intra_human_sims,
    }

    if not skip_random:
        pool_human_texts = [t for t in all_human_pool if t in text2idx]
        random_human_sims = []
        for _ in range(n):
            t1, t2 = rng.sample(pool_human_texts, 2)
            random_human_sims.append(float(emb(t1) @ emb(t2)))
        labels["random_human"] = random_human_sims

        pool_ai_texts = [t for t in ai_texts if t in text2idx]
        random_ai_sims = []
        for _ in range(n):
            t1, t2 = rng.sample(pool_ai_texts, 2)
            random_ai_sims.append(float(emb(t1) @ emb(t2)))
        labels["random_ai"] = random_ai_sims

    # ------------------------------------------------------------------ #
    # 9. Assemble DataFrame                                                #
    # ------------------------------------------------------------------ #
    min_n = min(len(sims) for sims in labels.values())
    print(
        "  Final sample sizes: " +
        ", ".join(f"{name}={len(sims)}" for name, sims in labels.items()) +
        f"  → using {min_n}"
    )

    rows = []
    for dist_name, sims in labels.items():
        for s in sims[:min_n]:
            rows.append({"platform": platform, "distribution": dist_name, "similarity": s})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity baselines.")
    parser.add_argument("results_folder",
                        help="Cleaned results folder (e.g. results_cleaned_20260309_095640). "
                             "Platform subfolders are detected automatically.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save output CSVs (default: <results_folder>/cosine_baselines)")
    parser.add_argument("--config-filter", default=None,
                        help="Substring to match in result filenames (default: SOTA config noft/ctx0/style0)")
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random_human/random_ai (and avoid encoding the full training-message "
                             "pool they require) — much faster for configs with diverse candidates")
    args = parser.parse_args()

    base = Path(args.results_folder)
    platforms = sorted([p for p in base.iterdir() if p.is_dir()])
    if not platforms:
        print(f"No platform subdirectories found in {base}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else base / "cosine_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("Loading sentence transformer model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SENTENCE_MODEL_PATH)

    rng = random.Random(RANDOM_SEED)
    all_dfs = []

    for platform_path in platforms:
        platform = platform_path.name
        posts_pkl = Path("data") / platform / "posts.pkl"
        if not posts_pkl.exists():
            print(f"  posts.pkl not found for {platform}, skipping.")
            continue

        df = compute_platform_baselines(
            str(platform_path), str(posts_pkl), model, rng, args.config_filter, args.skip_random
        )
        if df is not None:
            out_path = output_dir / f"cosine_baselines_{platform}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = output_dir / "cosine_baselines_all.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nCombined saved: {out_path}")

        # Print summary stats
        print("\n=== Summary ===")
        print(combined.groupby(["platform", "distribution"])["similarity"]
                      .agg(["median", "mean", "std"])
                      .round(3)
                      .to_string())


if __name__ == "__main__":
    main()

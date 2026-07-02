#!/usr/bin/env python3
"""
Compute same-context cosine similarity distributions (reviewer calibration analysis).

For each platform, finds source posts (reply_to contexts) where ≥2 different users
both have a human response AND an AI-generated response, then computes:

  1. human_human_same_ctx    — cosine similarity between two DIFFERENT users' human
                               responses to the SAME source post
  2. ai_ai_same_ctx          — cosine similarity between two DIFFERENT users' AI-generated
                               responses to the SAME source post, from the SAME model
  3. ai_intra_same_user_ctx  — cosine similarity between two DIFFERENT candidate
                               responses generated for the SAME (model, user, context),
                               drawn from `all_valid_responses` (requires
                               *_optimal_response.json, produced by LLM_judge.py)

Distributions 2 and 3 are computed per (model, context) and pooled across all SOTA-config
models, so that a single model's data can't silently crowd out the others (all evaluated
the same (context, user) pairs, so a naive context-only dedup would keep only the first
model glob() happens to return). Pairs are always same-model: this isolates genuine
user-level (or generation-level) diversity from cross-model stylistic differences.

These distributions together show whether AI responses are more or less diverse
than human responses when conditioned on the same context, and how much of that
diversity is just generation noise for a single user/context (distribution 3),
providing the calibration requested by reviewers for interpreting cosine
similarity scores.

The analysis uses the SOTA config (noft, ctx0, style0, with persona) by default.
Override with --config-filter to select a different config substring.

Inputs:
  <results_folder>           — cleaned results folder (e.g. results_cleaned/)
  data/{platform}/posts.pkl  — already present in the repo

Output:
  same_context_similarity/same_ctx_sims_{platform}.csv
  same_context_similarity/same_ctx_sims_all.csv

Usage:
    python analysis/compute_same_context_similarity.py results_cleaned/ [--output-dir DIR]
    python analysis/compute_same_context_similarity.py results_cleaned/ --config-filter noft__ctx0__style10
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
from itertools import combinations

SENTENCE_MODEL_PATH = os.environ.get("SENTENCE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
RANDOM_SEED = 42
DELETED = {"[deleted by user]", "[deleted]", "[removed]", ""}
MIN_CONTEXT_LEN = 20


def is_sota_config(fname):
    fname = os.path.basename(fname)
    return (
        "__noft__" in fname
        and "__ctx0__" in fname
        and "__style0__" in fname
        and "no_persona" not in fname
        and "__no_persona" not in fname
    )


def batch_encode(model, texts, batch_size=512):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embs = model.encode(
            batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        all_embs.append(embs)
    return np.vstack(all_embs)


def compute_platform(platform_folder, posts_pkl, sentence_model, rng, config_filter=None):
    platform = os.path.basename(platform_folder)
    print(f"\n=== {platform.upper()} ===")

    # ------------------------------------------------------------------ #
    # 1. Load AI results                                                   #
    # ------------------------------------------------------------------ #
    if config_filter:
        opt_files = glob.glob(
            str(Path(platform_folder) / "**" / "*_optimal_response.json"), recursive=True
        )
        opt_files = [f for f in opt_files if config_filter in os.path.basename(f)]
    else:
        opt_files = glob.glob(
            str(Path(platform_folder) / "**" / "*_optimal_response.json"), recursive=True
        )
        opt_files = [f for f in opt_files if is_sota_config(f)]

    print(f"  Config files matched: {len(opt_files)}")
    if not opt_files:
        print("  No matching result files found, skipping.")
        return None

    def candidate_texts(entry):
        """Distinct candidate response texts from all_valid_responses (same user/context)."""
        candidates = entry.get("all_valid_responses") or []
        texts = []
        for c in candidates:
            text = c.get("response") if isinstance(c, dict) else c
            if text and str(text).strip() not in DELETED:
                texts.append(text)
        return list(set(texts))

    # Index AI entries by (model, reply_to, user) — one response per (model, context,
    # author). All SOTA-config models are evaluated on the same (context, user) pairs,
    # so deduping only by (context, user) — ignoring the model — would let whichever
    # model glob() returns first silently claim every pair, leaving zero data for the
    # other models. Keeping the model in the key avoids that; within-model duplicates
    # (e.g. across batches of the same model) still collapse to the first entry.
    ai_by_model_context = {}          # model -> ctx -> {user: response}
    ai_candidates_by_model_ctx_user = {}  # (model, ctx, user) -> [candidate texts]
    seen_per_model = {}
    for fpath in opt_files:
        with open(fpath) as f:
            entries = json.load(f)
        for e in entries:
            model = e.get("model", os.path.basename(fpath).split("__")[0])
            ctx = e.get("reply_to", "")
            if not ctx or str(ctx).strip() in DELETED or len(str(ctx)) < MIN_CONTEXT_LEN:
                continue
            user = e.get("user", e.get("username", ""))
            if not user:
                continue
            seen = seen_per_model.setdefault(model, set())
            if (ctx, user) in seen:
                continue
            response = e.get("ML_best_response") or e.get("response", "")
            if not response:
                continue
            seen.add((ctx, user))
            ai_by_model_context.setdefault(model, {}).setdefault(ctx, {})[user] = response
            ai_candidates_by_model_ctx_user[(model, ctx, user)] = candidate_texts(e)

    print(f"  Models found: {sorted(ai_by_model_context)}")

    # Keep only (model, context) pairs with ≥2 distinct users for that model
    ai_multi_by_model = {
        model: {ctx: u2r for ctx, u2r in ctx_dict.items() if len(u2r) >= 2}
        for model, ctx_dict in ai_by_model_context.items()
    }
    n_model_ctx = sum(len(d) for d in ai_multi_by_model.values())
    print(f"  AI: (model, context) pairs with ≥2 users: {n_model_ctx}")

    # ------------------------------------------------------------------ #
    # 2. Load human test posts                                             #
    # ------------------------------------------------------------------ #
    with open(posts_pkl, "rb") as f:
        posts_df = pickle.load(f)
    posts_df = posts_df.reset_index(drop=True)

    test = posts_df[posts_df["training"] == 0].copy()
    test = test[test["reply_to"].notna()]
    test = test[~test["reply_to"].str.strip().isin(DELETED)]
    test = test[test["reply_to"].str.len() >= MIN_CONTEXT_LEN]

    # One response per (context, author) — if a user replied multiple times to the
    # same source post, keep only their first message so pairs are always cross-author.
    test = test.drop_duplicates(subset=["reply_to", "username"], keep="first")

    human_by_context = (
        test.groupby("reply_to")
        .apply(lambda g: dict(zip(g["username"], g["message"])))
        .to_dict()
    )
    human_multi = {ctx: u2r for ctx, u2r in human_by_context.items() if len(u2r) >= 2}
    print(f"  Human: contexts with ≥2 users: {len(human_multi)}")

    # ------------------------------------------------------------------ #
    # 3. Find (model, context) pairs whose context also has ≥2 human users #
    # ------------------------------------------------------------------ #
    model_ctx_pairs = [
        (model, ctx)
        for model, ctx_dict in ai_multi_by_model.items()
        for ctx in ctx_dict
        if ctx in human_multi
    ]
    print(f"  (model, context) pairs also present in human data: {len(model_ctx_pairs)}")

    if not model_ctx_pairs:
        print("  No shared contexts found, skipping.")
        return None

    shared_contexts = sorted({ctx for _, ctx in model_ctx_pairs})

    # ------------------------------------------------------------------ #
    # 4. Sample pairs. Human-human is context-level (one sample per shared #
    #    context — human data has no model dimension). AI-AI and intra-AI #
    #    are (model, context)-level and always same-model, so every SOTA  #
    #    model contributes its own samples instead of one model silently  #
    #    crowding out the rest.                                           #
    # ------------------------------------------------------------------ #
    rng.shuffle(shared_contexts)
    rng.shuffle(model_ctx_pairs)

    human_pair_texts = {}   # ctx -> (m1, m2), two different users' human responses
    ai_pair_texts = {}      # (model, ctx) -> (r1, r2), two different users' AI responses (same model)
    intra_pair_texts = {}   # (model, ctx) -> (c1, c2), one user's two distinct AI candidates

    for ctx in shared_contexts:
        h_users = list(human_multi[ctx].keys())
        h_pairs = list(combinations(h_users, 2))
        rng.shuffle(h_pairs)
        if h_pairs:
            u1, u2 = h_pairs[0]
            human_pair_texts[ctx] = (human_multi[ctx][u1], human_multi[ctx][u2])

    for model, ctx in model_ctx_pairs:
        a_users = list(ai_multi_by_model[model][ctx].keys())

        a_pairs = list(combinations(a_users, 2))
        rng.shuffle(a_pairs)
        u1, u2 = a_pairs[0]
        ai_pair_texts[(model, ctx)] = (
            ai_multi_by_model[model][ctx][u1], ai_multi_by_model[model][ctx][u2]
        )

        candidate_users = list(a_users)
        rng.shuffle(candidate_users)
        for u in candidate_users:
            candidates = ai_candidates_by_model_ctx_user.get((model, ctx, u), [])
            if len(candidates) >= 2:
                c1, c2 = rng.sample(candidates, 2)
                intra_pair_texts[(model, ctx)] = (c1, c2)
                break

    # ------------------------------------------------------------------ #
    # 5. Collect texts and encode                                          #
    # ------------------------------------------------------------------ #
    texts_needed = set()
    for m1, m2 in human_pair_texts.values():
        texts_needed.update([m1, m2])
    for r1, r2 in ai_pair_texts.values():
        texts_needed.update([r1, r2])
    for c1, c2 in intra_pair_texts.values():
        texts_needed.update([c1, c2])

    unique_texts = list(texts_needed)
    print(f"  Encoding {len(unique_texts)} unique texts...")
    embs = batch_encode(sentence_model, unique_texts)
    text2idx = {t: i for i, t in enumerate(unique_texts)}

    def emb(text):
        return embs[text2idx[text]]

    human_sims = [float(emb(m1) @ emb(m2)) for m1, m2 in human_pair_texts.values()]
    ai_sims = {mc: float(emb(r1) @ emb(r2)) for mc, (r1, r2) in ai_pair_texts.items()}
    intra_sims = {mc: float(emb(c1) @ emb(c2)) for mc, (c1, c2) in intra_pair_texts.items()}

    print(
        f"  Same-context pairs: human-human={len(human_sims)}, "
        f"AI-AI (same model)={len(ai_sims)}, intra-AI (same model+user)={len(intra_sims)}"
    )

    if not human_sims and not ai_sims and not intra_sims:
        return None

    rows = []
    for s in human_sims:
        rows.append({"platform": platform, "model": None,
                     "distribution": "human_human_same_ctx", "similarity": s})
    for (model, ctx), s in ai_sims.items():
        rows.append({"platform": platform, "model": model,
                     "distribution": "ai_ai_same_ctx", "similarity": s})
    for (model, ctx), s in intra_sims.items():
        rows.append({"platform": platform, "model": model,
                     "distribution": "ai_intra_same_user_ctx", "similarity": s})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute same-context human-human and AI-AI cosine similarity distributions."
    )
    parser.add_argument(
        "results_folder",
        help="Cleaned results folder (e.g. results_cleaned/). Platform subfolders detected automatically."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save output CSVs (default: same_context_similarity/ next to results_folder)"
    )
    parser.add_argument(
        "--config-filter", default=None,
        help="Substring to match in result filenames (default: SOTA config noft/ctx0/style0)"
    )
    args = parser.parse_args()

    base = Path(args.results_folder)
    platforms = sorted([p for p in base.iterdir() if p.is_dir()])
    if not platforms:
        print(f"No platform subdirectories found in {base}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else base.parent / "same_context_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("Loading sentence transformer model...")
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer(SENTENCE_MODEL_PATH)

    rng = random.Random(RANDOM_SEED)
    all_dfs = []

    for platform_path in platforms:
        platform = platform_path.name
        posts_pkl = Path("data") / platform / "posts.pkl"
        if not posts_pkl.exists():
            print(f"  posts.pkl not found for {platform}, skipping.")
            continue

        df = compute_platform(
            str(platform_path), str(posts_pkl), sentence_model, rng, args.config_filter
        )
        if df is not None:
            out_path = output_dir / f"same_ctx_sims_{platform}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = output_dir / "same_ctx_sims_all.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nCombined saved: {out_path}")

        print("\n=== Summary (pooled across models) ===")
        print(
            combined.groupby(["platform", "distribution"])["similarity"]
            .agg(["count", "median", "mean", "std"])
            .round(3)
            .to_string()
        )

        print("\n=== Summary by model ===")
        print(
            combined[combined["model"].notna()]
            .groupby(["platform", "distribution", "model"])["similarity"]
            .agg(["count", "median", "mean", "std"])
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()

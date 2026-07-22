#!/usr/bin/env python3
"""
Intra-AI same-user-same-context similarity, computed over EVERY entry in the
test set (not restricted to contexts with >=2 human repliers).

Unlike ai_ai_same_ctx / human_human_same_ctx, this metric needs no other user to
have replied to the same post: for one (model, context, user) entry, it samples
two of that entry's own generated candidates and computes their cosine
similarity. That only requires the entry to have >=2 distinct candidates, which
is true almost everywhere -- so it can be computed for Bluesky too, and with a
much larger sample on Reddit/Twitter than the shared-context-restricted version
in compute_same_context_similarity.py.

There is also no cross-model-mixing risk here (unlike the cross-user metrics):
each entry belongs to exactly one model by construction.

Configs: SOTA (noft/ctx0/style0) by default; override with --config-filter.
Platforms: reddit, twitter, bluesky (all three -- no same-context restriction).

Usage:
    python analysis/compute_intra_ai_all_entries.py results_PNAS_revision/ \
        --output-dir same_context_similarity
"""

import os
import argparse
import json
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd

SENTENCE_MODEL_PATH = os.environ.get("SENTENCE_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
RANDOM_SEED = 42
DELETED = {"[deleted by user]", "[deleted]", "[removed]", ""}
MIN_CONTEXT_LEN = 20
PLATFORMS = ["reddit", "twitter", "bluesky"]


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
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(embs)
    return np.vstack(all_embs)


def candidate_texts(entry):
    candidates = entry.get("all_valid_responses") or []
    texts = []
    for c in candidates:
        text = c.get("response") if isinstance(c, dict) else c
        if text and str(text).strip() not in DELETED:
            texts.append(text)
    return list(set(texts))


def compute_platform(platform_folder, config_filter, rng, sentence_model):
    opt_files = glob.glob(str(Path(platform_folder) / "**" / "*_optimal_response.json"), recursive=True)
    if config_filter:
        opt_files = [f for f in opt_files if config_filter in os.path.basename(f)]
    else:
        opt_files = [f for f in opt_files if is_sota_config(f)]

    rows = []
    for fpath in opt_files:
        entries = json.load(open(fpath))
        if not entries:
            continue
        model = entries[0].get("model", os.path.basename(fpath).split("__")[0])

        pairs = []  # (c1, c2) per valid entry
        for e in entries:
            ctx = e.get("reply_to", "")
            if not ctx or str(ctx).strip() in DELETED or len(str(ctx)) < MIN_CONTEXT_LEN:
                continue
            cands = candidate_texts(e)
            if len(cands) < 2:
                continue
            c1, c2 = rng.sample(cands, 2)
            pairs.append((c1, c2))

        if not pairs:
            continue

        unique_texts = list({t for pair in pairs for t in pair})
        embs = batch_encode(sentence_model, unique_texts)
        text2idx = {t: i for i, t in enumerate(unique_texts)}

        for c1, c2 in pairs:
            sim = float(embs[text2idx[c1]] @ embs[text2idx[c2]])
            rows.append({"model": model, "similarity": sim})

        print(f"    {model}: {len(pairs)} entries")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Intra-AI same-user-same-context similarity over all entries.")
    parser.add_argument("results_folder", help="Root results folder (e.g. results_PNAS_revision/)")
    parser.add_argument("--config-filter", default=None, help="Substring to match (default: SOTA config)")
    parser.add_argument("--output-dir", default=None, help="Where to save output CSV")
    args = parser.parse_args()

    base = Path(args.results_folder)
    output_dir = Path(args.output_dir) if args.output_dir else Path("same_context_similarity")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sentence transformer model...")
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer(SENTENCE_MODEL_PATH)

    rng = random.Random(RANDOM_SEED)
    all_rows = []
    for platform in PLATFORMS:
        platform_folder = base / platform
        if not platform_folder.exists():
            continue
        print(f"\n=== {platform.upper()} ===")
        rows = compute_platform(str(platform_folder), args.config_filter, rng, sentence_model)
        for r in rows:
            r["platform"] = platform
            r["distribution"] = "ai_intra_same_user_ctx"
        all_rows.extend(rows)
        print(f"  Total: {len(rows)} rows")

    df = pd.DataFrame(all_rows)[["platform", "model", "distribution", "similarity"]]
    out_path = output_dir / "ai_intra_same_user_ctx_all_entries.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")
    print(df.groupby("platform")["similarity"].agg(["count", "median", "mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()

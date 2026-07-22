# Reviewer Experiment Instructions

Steps to run the experiments added in response to peer-review R2.

---

## Experiment A — Looser decoding hyperparameters (Concern 2)

Platform: Bluesky (60 users, 1 batch — smallest, fully self-contained). Baseline
(reference) numbers come from the default-decoding run already in
`results_PNAS_revision/bluesky/`.

| Config | Model | Decoding variant | Temperature | top_p | top_k | Config file |
|---|---|---|---|---|---|---|
| Default (SOTA, reference) | Llama-3.1-8B | — | 0.8 | 0.9 | 50 | `configs/llama_3_1_8b_persona_style_context.yaml` |
| Llama looser | Llama-3.1-8B | looser | 1.0 | 0.95 | 0 (disabled) | `configs/llama_3_1_8b_persona_style_context_looser.yaml` |
| Llama medium | Llama-3.1-8B | medium | 0.9 | 0.925 | 25 | `configs/llama_3_1_8b_persona_style_context_medium.yaml` |
| Mistral looser | Mistral-7B-Instruct-v0.2 | looser | 1.0 | 0.95 | 0 (disabled) | `configs/mistral_7b_2_instruct_persona_style_context_looser.yaml` |

### Step 1 — Generate responses

```bash
python simulation/run_simulation.py \
    --config_file configs/<config>.yaml \
    --dataset bluesky \
    --user_batch 0 \
    --output_dir results_hyperparameters
```

### Step 2 — Join batch (single batch, so this is trivial)

```bash
python simulation/join_complete_batches.py \
    --output-dir results_hyperparameters_joined/ \
    --input-dir results_hyperparameters/
```

### Step 3 — Clean responses

```bash
python postprocessing/response_cleaning.py results_hyperparameters_joined/ \
    --output-dir results_hyperparameters_cleaned/
```

### Step 4 — Select best responses

```bash
python postprocessing/LLM_judge.py results_hyperparameters_cleaned/ --include-advanced
```

### Step 5 — Convert to CSV

```bash
python postprocessing/optimal_responses_to_csv.py results_hyperparameters_cleaned/
```

### Step 6 — Build validation dataset

```bash
python postprocessing/build_validation_data.py \
    --folder results_hyperparameters_cleaned/bluesky
```

### Step 7 — Run BERT validation

```bash
python postprocessing/validate_text.py \
    --input_dir results_hyperparameters_cleaned/bluesky/ \
    --validation bert
```

Compare the resulting BERT accuracy against the baseline Bluesky result for the
matching model in `results_PNAS_revision/bluesky/`.

### Results (BERT detection accuracy — lower means harder to tell AI from human)

| Model | Validation | Baseline | Medium | Looser | Looser − Baseline |
|---|---|---|---|---|---|
| Llama-3.1-8B | cosine-selected | 70.8% | 71.4% | 69.5% | −1.3pp |
| Llama-3.1-8B | ML-selected | 72.0% | 68.9% | 68.9% | **−3.2pp** |
| Llama-3.1-8B | random (unfiltered) | 82.0% | 85.8% | 84.5% | +2.5pp |
| Mistral-7B-Instruct-v0.2 | cosine-selected | 96.0% | — | 95.5% | −0.5pp |
| Mistral-7B-Instruct-v0.2 | ML-selected | 92.6% | — | 88.9% | **−3.7pp** |
| Mistral-7B-Instruct-v0.2 | random (unfiltered) | 96.8% | — | 96.8% | ~0pp |

Looser decoding consistently helps on the judge-selected pathways (what the paper
actually reports), especially ML-validation, but *hurts* on the unfiltered random
pick for Llama — consistent with looser decoding widening the candidate spread
(less homogeneous) rather than uniformly improving quality.

Human-vs-AI cosine similarity tells a complementary story — extracted directly
from the precomputed `cosine_similarity` scores already stored in each
`all_valid_responses` entry of `*_optimal_response.json` (no re-embedding needed):

| Model | Decoding | Random-pick cosine (median) | Cosine-best-of-20 (median) |
|---|---|---|---|
| Llama-3.1-8B | baseline | 0.160 | 0.505 |
| Llama-3.1-8B | medium | 0.145 | 0.495 |
| Llama-3.1-8B | looser | 0.133 | 0.492 |
| Mistral-7B-Instruct-v0.2 | baseline | 0.384 | 0.537 |
| Mistral-7B-Instruct-v0.2 | looser | 0.386 | 0.539 |

For Llama, looser decoding *decreases* raw cosine similarity to the human ground
truth on both readings — so the BERT-detectability improvement isn't coming from
AI text becoming more semantically similar to what that specific human wrote; it's
more likely a shift in surface-level/stylistic features (word count, punctuation,
repetition) that the ML classifier's hand-crafted features pick up on, largely
independent of embedding-level fidelity. For Mistral, cosine similarity is
essentially flat across decoding variants.

---

## Experiment B — Same-context similarity: AI-AI vs Human-Human (Concern 1)

Separate standalone analysis. No new simulation needed, but requires the
`LLM_judge.py` postprocessing step to have already run (needs `*_optimal_response.json`,
not just the raw `*_random_response.json` from `response_cleaning.py`) — the
`all_valid_responses` field it writes is what makes the intra-AI distribution
possible.

**Inputs:**
- `data/{platform}/posts.pkl` — already in repo
- `results_cleaned/` (or e.g. `results_PNAS_revision/`) — must contain
  `*_optimal_response.json` files (i.e. `LLM_judge.py` has already run)

**What it measures**, per platform:
- `human_human_same_ctx` — cosine similarity between two *different* users' human
  responses to the *same* source post (calibration baseline)
- `ai_ai_same_ctx` — cosine similarity between two *different* users' AI-generated
  responses to the *same* source post, from the *same* model
- `ai_intra_same_user_ctx` — cosine similarity between two *different* candidate
  responses generated for the *same* (model, user, context) — drawn from
  `all_valid_responses` — i.e. generation-level noise for one fixed input

`ai_ai_same_ctx` and `ai_intra_same_user_ctx` are always computed *within a single
model*: all SOTA-config models are evaluated on the same (context, user) pairs, so
naively deduping by (context, user) alone — ignoring which model produced the
response — lets whichever model `glob()` happens to return first silently claim
every pair, leaving zero data for the other 8 models. Keying by `(model, context,
user)` avoids this and lets every SOTA model contribute its own same-model pairs.

**Script:** `analysis/compute_same_context_similarity.py`

```bash
python analysis/compute_same_context_similarity.py results_PNAS_revision/
```

Output: `same_context_similarity/same_ctx_sims_{platform}.csv` and `same_ctx_sims_all.csv`
(default location: `same_context_similarity/` next to the results folder passed in).
Rows have a `model` column (`None` for `human_human_same_ctx`, which is pooled
across all models since human data has no model dimension).

**Observed data (test split, SOTA config, after the same-model fix):**
- Reddit: 1,371 shared contexts (human) → 12,278 (model, context) AI pairs across 9 models
- Twitter: 313 shared contexts (human) → 152 (model, context) AI pairs across 9 models
- Bluesky: excluded automatically (only 1 human context has ≥2 users)

By default the SOTA config (noft, ctx0, style0, with persona) is used, pooled
across all base models. To use a different config: `--config-filter <substring>`.

### Plotting

**Script:** `analysis/plot_same_context_similarity.py` — boxplots in the same visual
style as `generate_SOTA_plots.py`'s `SOTA_cosine_baselines.png` (per-platform panels,
seaborn-style palette, legend below). Bluesky is always excluded.

```bash
# 3 columns: human_human_same_ctx, ai_ai_same_ctx, ai_intra_same_user_ctx
python analysis/plot_same_context_similarity.py same_context_similarity/same_ctx_sims_all.csv

# 7 columns: also pulls in random_human, intra_human, human_vs_ai, random_ai from
# compute_cosine_baselines.py's output, for full context (unrelated-pair floor,
# same-user-different-context baseline, and the paper's main human-vs-AI metric)
python analysis/plot_same_context_similarity.py same_context_similarity/same_ctx_sims_all.csv \
    --baselines-csv results_PNAS_revision/cosine_baselines/cosine_baselines_all.csv
```

Output: `same_context_similarity/same_ctx_sims_boxplot.png` and `same_ctx_sims_stats.csv`.

Columns are grouped left to right by comparison type (Random pairs / Human–Human /
Human–AI / AI–AI); each group's members and order are controlled by the
`GROUPS_BASE` / `GROUPS_WITH_BASELINES` lists near the top of the script — reorder
those to reorder the plot. Note `SOTA_cosine_baselines.png`'s `intra_ai` column
(same prompt, pooled across models without the same-model fix) has no equivalent
here — the closest analogue is `ai_intra_same_user_ctx`, but it is a different,
corrected computation, not the same values.

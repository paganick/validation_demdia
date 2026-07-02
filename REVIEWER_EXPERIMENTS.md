# Reviewer Experiment Instructions

Steps to run the experiments added in response to peer-review R2.

---

## Experiment A — Looser decoding hyperparameters (Concern 2)

Config: `configs/llama_3_1_8b_persona_style_context_looser.yaml`  
Settings: T=1.0, top_p=0.95, top_k disabled (vs. original T=0.8, top_p=0.9, top_k=50)  
Platform: Bluesky (60 users, 1 batch — smallest, fully self-contained)

### Step 1 — Generate responses

```bash
python simulation/run_simulation.py \
    --config_file configs/llama_3_1_8b_persona_style_context_looser.yaml \
    --dataset bluesky \
    --user_batch 0 \
    --output_dir results_looser
```

Output: `results_looser/meta-llama_Llama-3.1-8B__noft__ctx1__style10__looser__batch0.json`

### Step 2 — Join batch (single batch, so this is trivial)

```bash
python simulation/join_complete_batches.py \
    --output-dir results_looser_joined/ \
    --input-dir results_looser/
```

### Step 3 — Clean responses

```bash
python postprocessing/response_cleaning.py results_looser_joined/ \
    --output-dir results_looser_cleaned/
```

### Step 4 — Select best responses

```bash
python postprocessing/LLM_judge.py results_looser_cleaned/ --include-advanced
```

### Step 5 — Convert to CSV

```bash
python postprocessing/optimal_responses_to_csv.py results_looser_cleaned/
```

### Step 6 — Build validation dataset

```bash
python postprocessing/build_validation_data.py \
    --folder results_looser_cleaned/bluesky
```

### Step 7 — Run BERT validation

```bash
python postprocessing/validate_text.py \
    --input_dir results_looser_cleaned/bluesky/ \
    --validation bert
```

Compare the resulting BERT accuracy against the baseline Bluesky result for
`llama_3_1_8b_persona_style_context` in `results_cleaned/bluesky/`.

---

## Experiment B — Same-context similarity: AI-AI vs Human-Human (Concern 1)

Separate standalone analysis. No new simulation needed.

**Inputs:**
- `data/{platform}/posts.pkl` — already in repo
- `results_cleaned/` — simulation results (upload required)

**What it measures:**
- `human_human_same_ctx` — cosine similarity between two *different* users' human
  responses to the *same* source post (calibration baseline)
- `ai_ai_same_ctx` — cosine similarity between two *different* users' AI-generated
  responses to the *same* source post

**Script:** `analysis/compute_same_context_similarity.py`

```bash
python analysis/compute_same_context_similarity.py results_cleaned/
```

Output: `same_context_similarity/same_ctx_sims_{platform}.csv` and `same_ctx_sims_all.csv`

**Available data (test split, contexts with ≥2 users):**
- Reddit: 7,932 contexts — excellent
- Twitter: 313 contexts — feasible
- Bluesky: 1 context — excluded automatically

By default the SOTA config (noft, ctx0, style0, with persona) is used.
To use a different config: `--config-filter <substring>`.

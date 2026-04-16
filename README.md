# Validation of AI-Generated Social Media Text

Research project for validating AI-generated social media posts against real human writing across multiple platforms (Bluesky, Twitter/X, Reddit).

## Overview

This repository implements a comprehensive system for:
1. **Generating** AI-written social media posts using various LLMs with personalization techniques
2. **Selecting** the most human-like responses using ML-based judging
3. **Validating** the quality of generated text through multiple methods
4. **Analyzing** linguistic features that distinguish human from AI writing

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Workflow](#workflow)
- [Personalization Methods](#personalization-methods)
- [Validation Methods](#validation-methods)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Output Files](#output-files)

## Architecture

### 1. Text Generation Pipeline

**Models Supported:**
- Llama 3.1 (8B, 70B)
- Mistral 7B
- DeepSeek-R1-Distill
- Gemma
- Qwen
- Apertus

**Personalization Techniques:**
- **Persona descriptions**: GPT-4 generated user profiles
- **Style examples**: 0, 1, 3, or 5 example posts from target user
- **Context retrieval**: BM25-based retrieval of relevant past posts
- **Fine-tuning**: LoRA-based adaptation on user's writing
- **OPPU**: Personalized LoRA adapters trained per-user

### 2. Response Selection ("LLM Judge")

For each prompt, the system generates multiple candidate responses and selects the best using:

**ML-based selection:**
- Trains Random Forest on 20 linguistic features
- Classifies text as human (label=1) or AI (label=0)
- Selects candidate with highest "human-like" probability

**Cosine similarity selection:**
- Embeds candidates using all-MiniLM-L6-v2
- Computes semantic similarity to original human message
- Selects most similar candidate

### 3. Validation Methods

**BERT Classification:**
- Fine-tunes BERT to distinguish human from AI text
- 5 training runs, median accuracy selected
- Reports accuracy, F1, confusion matrix

**Empath Linguistic Analysis:**
- Computes 200 psycholinguistic categories per text
- Wilcoxon rank-sum tests for significance
- FDR correction for multiple comparisons
- Identifies categories that differ between human and AI

**Random Forest Feature Analysis:**
- Extracts 20 linguistic features (word count, sentiment, perplexity, etc.)
- Trains classifier on ground truth labels or BERT predictions
- Computes feature importance and correlation signs
- Evaluates using median BERT validation set

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate validation_demdia
```

**PyTorch + CUDA:** the default `torch>=2.10.0` entry in `environment.yml` installs the CPU build.
For GPU support, install PyTorch separately with the appropriate CUDA index URL after activating the environment:

```bash
pip install torch>=2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with your CUDA version (e.g. `cu118`, `cu121`). See [pytorch.org](https://pytorch.org/get-started/locally/) for options.

**Apptainer/Singularity:** the `conda.def` definition file builds a container image (`conda.sif`) that mirrors this environment and is used for cluster runs.

## Workflow

### Step 0: Generate Persona Descriptions

Each user in the dataset is represented by a **persona description** — a paragraph
summarising their communication style and interests, used to condition the LLM.
Pre-generated GPT-4 personas are provided in `data/{platform}/personas.pkl`.

To regenerate them using **Llama-3.1-8B-Instruct** (e.g. for a new dataset or platform),
run the two-step pipeline:

**Step 0a — Generate third-person descriptions from posts:**

```bash
python preprocessing/generate_personas_llama.py --platform all
```

Reads `data/{platform}/posts.pkl`, writes `data/{platform}/personas_llama.pkl`
(third-person, one row per user with a `persona` column).

**Step 0b — Transform to second-person:**

```bash
python preprocessing/transform_llama_personas_to_second_person.py --platform all
```

Reads `personas_llama.pkl`, adds a `persona_third_person` column (original) and
overwrites `persona` with the second-person version required by instruction-tuned
prompts. Saves back to `personas_llama.pkl`.

**Step 0c — Promote to canonical file:**

```bash
cp data/{platform}/personas_llama.pkl data/{platform}/personas.pkl
```

After this, `personas.pkl` has the two columns expected by `simulation/run_simulation.py`:
- `persona` — second-person ("You are @User_XXXX…"), used with instruction-tuned models
- `persona_third_person` — third-person ("@User_XXXX is…"), used with base models

Both steps accept `--data-dir <path>` to target a custom data directory (e.g. for testing).

---

### Step 1: Generate AI Responses

Simulations must be run **in batches** to ensure reproducibility. Users are split into
fixed batches of ≤ 60 (stored in `user_batches.json`), and each batch is processed
independently. This is required for fine-tuned configurations, where each batch trains
a LoRA model on its own users — running without batches would mix users across
fine-tuning runs and produce non-reproducible results.

| Platform | Total users | Batches |
|----------|-------------|---------|
| Twitter  | 279         | 5       |
| Reddit   | 492         | 9       |
| Bluesky  | 60          | 1       |

**Setup:**

```bash
# (Optional) Regenerate batch assignments if personas change
python simulation/prepare_user_batches.py
```

**Run each batch** (one process per platform × config × batch, parallelise as your
cluster allows):

```bash
python simulation/run_simulation.py \
    --config_file=configs/<model>.yaml \
    --dataset=<platform> \
    --user_batch=<N> \
    --output_dir=results
```

Depending on model size and available GPU memory, runs may time out before completing
all users. Re-running the same command is safe — already-complete users are skipped
automatically. Repeat as needed until all batches are fully populated.

**After all batches are complete, join into per-config files:**

```bash
python simulation/join_complete_batches.py --output-dir results_joined/
```

This skips any config/platform combination where not all batch files are present and
produces `results_joined/{platform}/{vendor}/{ModelName}__{config_flags}__random_response.json`.
Use `--dry-run` to preview what would be joined without writing files.

**Fine-tuned model storage:** LoRA adapters can be large. By default they are written to
`finetuned_models/` in the project root. Set `FINETUNING_DIR` to redirect them elsewhere
(e.g. a scratch partition):

```bash
export FINETUNING_DIR=/scratch/$USER/finetuned_models
```

Alternatively, if `$SCRATCH` is set in your environment, the default automatically
becomes `$SCRATCH/finetuned_models/`.

**Fine-tuning coordination:** when multiple batches run in parallel, the first job that
needs a fine-tuned model trains it; the others wait via file lock and load it once
training finishes. Each batch trains on its own users and saves to a separate directory.

**Input:**
- Config YAMLs in `configs/` (one per model/configuration)
- User datasets: `data/{platform}/posts.pkl`, `data/{platform}/personas.pkl`
- Batch assignments: `user_batches.json`
- Per-user history files (`data/{platform}/user_histories/`) are created automatically
  on the first run from `posts.pkl` and do not need to be provided separately.

#### Running on a SLURM cluster

Each simulation job is a single call to `simulation/run_simulation.py` for one (config, platform,
batch) triple. The full job list can be enumerated from `configs/*.yaml` and
`user_batches.json`. A typical array job looks like this:

```bash
#!/bin/bash
#SBATCH --array=1-<N_TASKS>%<CONCURRENCY>
#SBATCH --time=<WALL_TIME>   # 6h for non-finetuned, 24h for finetuned
#SBATCH --gpus=1
#SBATCH --mem=64G            # 128G for finetuned configs

TASK_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks.tsv)
CONFIG=$(echo "$TASK_LINE" | cut -f1)
PLATFORM=$(echo "$TASK_LINE" | cut -f2)
BATCH=$(echo "$TASK_LINE" | cut -f3)

apptainer exec --nv conda.sif \
    python simulation/run_simulation.py \
        --config_file  "configs/$CONFIG" \
        --dataset      "$PLATFORM" \
        --output_dir   "results/$PLATFORM" \
        --user_batch   "$BATCH" \
        --batch_file   user_batches.json \
        --n_responses_per_user 20
```

where `tasks.tsv` is a tab-separated file with columns `config_yaml  platform  batch_id`,
one row per job. Generate it from the current configs and batch assignments with:

```bash
python simulation/join_complete_batches.py --list-tasks > tasks.tsv
```

You can monitor overall progress with:

```bash
python analyze_pipeline_status.py
```

**Output:**
- `results_joined/{platform}/{vendor}/{ModelName}__{config_flags}__random_response.json`

### Step 2: Clean Responses

Strip formatting artifacts introduced by instruction-tuned models (e.g. `[Response]`
headers, bold wrappers, handle prefixes) before response selection:

```bash
python postprocessing/response_cleaning.py results_joined/ --output-dir results_cleaned/
```

Always specify `--output-dir` so that `results_joined/` is kept intact as the raw
backup. If you later need to adjust the cleaning level or re-run with different
settings, the original joined files remain available. Running without `--output-dir`
modifies files in-place and makes the raw responses unrecoverable.

**Output:**
- `results_cleaned/{platform}/.../*_random_response.json` (cleaned copies)

### Step 3: Select Best Responses

Use ML and cosine similarity to select most human-like responses:

```bash
python postprocessing/LLM_judge.py results_cleaned/ --include-advanced
```

By default the script downloads `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace.
On air-gapped clusters, save it locally first and point to it via an environment variable:

```bash
export SENTENCE_MODEL_PATH=/path/to/all-MiniLM-L6-v2-local
```

**Input:**
- `*_random_response.json` files

**Output:**
- `*_optimal_response.json` (enriched with ML_best and cosine_best)
- `*_response_comparisons.csv` (side-by-side comparison)
- `*_responses_features.csv` (cached feature matrix)

### Step 4: Convert Optimal Responses to CSV

```bash
python postprocessing/optimal_responses_to_csv.py results_cleaned/
```

Converts each `*_optimal_response.json` produced by `LLM_judge.py` into a sibling `*_optimal_response.csv` file.

### Step 5: Build Validation Datasets

Create balanced datasets (human vs AI) for each platform:

```bash
# Bluesky
python postprocessing/build_validation_data.py --folder=results_cleaned/bluesky

# Twitter
python postprocessing/build_validation_data.py --folder=results_cleaned/twitter

# Reddit
python postprocessing/build_validation_data.py --folder=results_cleaned/reddit
```

**Output:**
- `*_ml_validation_data.csv` (ML-selected responses)
- `*_cosine_validation_data.csv` (Cosine-selected responses)
- `*_random_validation_data.csv` (Baseline)

### Step 6: Run Validation

```bash
python postprocessing/validate_text.py --input_dir=results_cleaned/bluesky/ --validation=all
python postprocessing/validate_text.py --input_dir=results_cleaned/twitter/ --validation=all
python postprocessing/validate_text.py --input_dir=results_cleaned/reddit/ --validation=all
```

**What `--validation=all` does:**
1. BERT validation (5 runs, median selected)
2. Empath linguistic analysis with statistical testing
3. Generates confusion matrices and feature lists

**Output files:**
- `*_validation_data_trainer_results.json` (BERT training metrics)
- `*_validation_data_labelled.csv` (texts with BERT predictions)
- `*_confusion_matrix.csv` (classification results)
- `*_significant_features.csv` (Empath categories with significant differences)

### Step 7: Feature Analysis

Compute linguistic features and train Random Forest:

```bash
# Compute and cache features
python postprocessing/features_analysis.py compute_features results_cleaned/

# Train RF on ground truth labels
python postprocessing/features_analysis.py evaluate results_cleaned/ labels

# Optional: Train RF on BERT predictions
python postprocessing/features_analysis.py evaluate results_cleaned/ bert_prediction
```

### Step 8: Compute Cosine Similarity Baselines

Required prerequisite for Figure 9 in `generate_SOTA_plots.py`. Computes five cosine
similarity distributions per platform (human-vs-AI, intra-AI, intra-human, random-human,
random-AI) and writes them to `cosine_baselines/`.

```bash
python analysis/compute_cosine_baselines.py results_cleaned/
```

### Step 9: Generate Plots

```bash
python analysis/generate_SOTA_plots.py results_cleaned/
python analysis/generate_config_optimal_plots.py results_cleaned/
```

Creates publication-ready figures for research papers.

### Fine-tuning Hyperparameter Ablation (reviewer response)

A self-contained ablation study comparing LoRA hyperparameter variants for Llama-3.1-8B
and Mistral-7B-v0.1 on Twitter. Configs for the sweep are in `configs/ft_hyperparam/`.

**To run from scratch:**

1. Run simulations (Steps 1–2 of the main pipeline) using the configs in `configs/ft_hyperparam/`.
2. Run postprocessing (Steps 3–7) on the sweep results to produce a postprocessed sweep folder.
3. Run the analysis scripts, pointing them at the sweep results and the main reference results:

```bash
SWEEP_PP=<postprocessed_sweep_dir>   # output of step 2 above
REF_PP=<postprocessed_ref_dir>       # main pipeline postprocessed results

python analysis/ft_hyperparam/analyze_ft_hyperparam.py    --sweep-dir $SWEEP_PP --ref-dir $REF_PP
python analysis/ft_hyperparam/analyze_ft_hyperparam_features.py --sweep-dir $SWEEP_PP --ref-dir $REF_PP
python analysis/ft_hyperparam/analyze_feature_differences.py    --sweep-dir $SWEEP_PP --ref-dir $REF_PP
```

Plots are written to `<sweep_dir>/`.

## Personalization Methods

### 1. Persona Description
- GPT-4 generated profile: "You are u/{username}, a person who..."
- Captures user interests, tone, typical topics
- Used as system prompt for LLM

### 2. Style Examples
- Provide 0, 1, 3, or 5 real posts from target user
- Model learns by few-shot imitation
- Sampled randomly from user's history

### 3. Context Retrieval
- BM25 search over user's past posts
- Retrieves contextually relevant examples for each new prompt
- Provides dynamic, query-specific examples

### 4. Fine-Tuning
- LoRA fine-tuning on user's complete message history
- Adapts model weights to user's writing patterns
- Computationally expensive but effective

### 5. OPPU (Optimal Personalized Prompt Unification)
- Trains separate LoRA adapter per user
- Combines fine-tuning with other personalization methods
- Best results but highest computational cost

## Validation Methods

### BERT Classification

Fine-tunes BERT-base to classify text as human (1) or AI (0).

**Procedure:**
1. Train/val split (80/20)
2. Fine-tune for 3 epochs with early stopping
3. Run 5 times with different random seeds
4. Select median accuracy run for evaluation
5. Compute accuracy, F1, confusion matrix

**Key metrics:**
- Accuracy: Overall classification correctness
- F1 Score: Harmonic mean of precision and recall
- Confusion Matrix: True positives, false positives, etc.

### Empath Linguistic Analysis

Uses Empath library to compute 200 psycholinguistic categories (e.g., "positive_emotion", "violence", "technology").

**Procedure:**
1. Compute category scores for all texts
2. Split by label (human vs AI)
3. Wilcoxon rank-sum test for each category
4. FDR correction for multiple comparisons (p < 0.05)
5. Report significantly different categories

**Insights:**
- Which linguistic dimensions differ between human and AI?
- E.g., AI may overuse formal language, humans use more emojis

### Random Forest Feature Analysis

Trains Random Forest on 20 engineered features to classify text.

**Basic Features (11):**
- word_count, link_count, mention_count, emoji_count
- hashtag_count, punctuation_count, uppercase_count
- char_count, avg_word_length
- sentiment (VADER), toxicity (toxic-bert)

**Advanced Features (9):**
- perplexity_proxy, type_token_ratio
- sentence_length_variance
- hedge_word_count, transition_word_count, superlative_count
- repetitive_patterns, abstract_concrete_ratio
- syntactic_complexity

**Procedure:**
1. Extract features from all texts
2. Train RF on training data (excluding BERT's validation set)
3. Predict on BERT's median-run validation set
4. Compare RF predictions with BERT predictions
5. Compute agreement statistics

**Key Insights:**
- Feature importance: Which features best distinguish human from AI?
- Agreement analysis: Do RF and BERT make similar mistakes?
- Correlation signs: Which features correlate with "human-ness"?

## Project Structure

```
validation_demdia/
├── configs/                           # Model configuration YAMLs (one per model/config)
│   └── ft_hyperparam/                 # Sweep configs for ft hyperparameter ablation
├── data/                              # Input datasets (anonymized)
│   ├── bluesky/posts.pkl, personas.pkl
│   ├── twitter/posts.pkl, personas.pkl
│   └── reddit/posts.pkl, personas.pkl
│
│   — preprocessing/: one-time data preparation —
├── preprocessing/
│   ├── anonymize_usernames.py          # Replace real usernames with User_XXXX IDs
│   ├── anonymize_mentions.py           # Anonymize @mentions in post text
│   └── parse_reddit_data.py            # Parse raw Reddit JSON dumps (gitignored)
│
│   — simulation/: batch setup, generation, joining —
├── simulation/
│   ├── prepare_user_batches.py         # Assign users to reproducible batches
│   ├── run_simulation.py               # Step 1a: generate AI responses (per batch)
│   ├── join_complete_batches.py        # Step 1b: merge batches → results_joined/
│   └── src/
│       ├── agent.py                    # User agent (loads posts, generates responses)
│       ├── model.py                    # LLM wrapper (loading, fine-tuning)
│       ├── simulate.py                 # Simulation orchestration
│       ├── feature_utils.py            # Feature extraction functions (20 features)
│       ├── plotting_utils.py           # Shared plotting utilities
│       ├── utils.py                    # Validator class (BERT, Empath)
│       └── config_utils.py, model_utils.py, globals.py
│
│   — postprocessing/: cleaning, judging, validation —
├── postprocessing/
│   ├── response_cleaning.py            # Step 2: strip formatting artifacts
│   ├── LLM_judge.py                    # Step 3: select best responses (ML + cosine)
│   ├── optimal_responses_to_csv.py     # Step 4: convert JSON → CSV
│   ├── build_validation_data.py        # Step 5: build human vs AI datasets
│   ├── validate_text.py                # Step 6: BERT + Empath validation
│   └── features_analysis.py            # Step 7: Random Forest feature analysis
│
│   — analysis/: final analysis and plots —
├── analysis/
│   ├── generate_SOTA_plots.py          # Main accuracy/comparison figures
│   ├── generate_config_optimal_plots.py   # Per-config optimality plots
│   ├── compute_cosine_baselines.py     # Step 8: cosine similarity baseline distributions
│   └── ft_hyperparam/                  # Fine-tuning hyperparameter ablation study
│       ├── analyze_ft_hyperparam.py        # BERT accuracy across sweep variants
│       ├── analyze_ft_hyperparam_features.py  # RF feature importances across variants
│       └── analyze_feature_differences.py  # Feature-level AI vs human differences
│
├── analyze_pipeline_status.py          # Monitor batch completion and SLURM queue
```

## Usage Examples

### Generate responses for a single configuration

```bash
python simulation/run_simulation.py \
    --config_file configs/llama_3_1_8b_base.yaml \
    --dataset bluesky \
    --user_batch 0 \
    --output_dir results_test
```

### Force reprocess all LLM Judge results

```bash
python postprocessing/LLM_judge.py results_cleaned/ --force-reprocess --include-advanced
```

### Validate with only BERT (skip Empath)

```bash
python postprocessing/validate_text.py --input_dir=results_cleaned/bluesky/ --validation=bert
```

### Validate with only Empath (skip BERT)

```bash
python postprocessing/validate_text.py --input_dir=results_cleaned/bluesky/ --validation=empath
```

## Output Files

### Generation Stage

- `*_random_response.json`: Original simulation output
  - Contains all_valid_responses (multiple candidates per prompt)
  - Randomly selected response for baseline

### Selection Stage

- `*_optimal_response.json`: Enriched with ML and cosine selections
  - ML_best_response: Random Forest selected
  - cosine_best_response: Cosine similarity selected
  - All candidates with scores (probability, cosine_similarity)

- `*_response_comparisons.csv`: Side-by-side comparison
  - Columns: user, reply_to, original_message, previous_response, ML_best_response, cosine_best_response

- `*_responses_features.csv`: Cached feature matrix
  - 20 features for all candidate responses

### Validation Stage

- `*_validation_data_labelled.csv`: Human vs AI texts with BERT predictions
  - Columns: text, labels (ground truth), bert_prediction

- `*_validation_data_trainer_results.json`: BERT training details
  - Accuracy, F1, predictions for each of 5 runs

- `*_confusion_matrix.csv`: Classification results (2x2 matrix)
  - Rows: Actual labels (0=AI, 1=Human)
  - Cols: Predicted labels

- `*_significant_features.csv`: Empath categories with significant differences
  - Category name, p-value, effect size

### Feature Analysis Stage

- `*_features.csv`: Extracted features for Random Forest
  - 20 columns (11 basic + 9 advanced)

- `*_from_labels_median_run_results.csv`: RF predictions on validation set
  - Columns: text, actual_label, bert_prediction, rf_prediction, rf_probability

- `*_from_labels_median_run_agreement.csv`: Agreement statistics
  - BERT-RF agreement, BERT-actual agreement, RF-actual agreement
  - Three-way agreement analysis

## Key Research Questions

1. **Which personalization method works best?**
   - Compare BERT accuracy across persona, style examples, context, fine-tuning, OPPU

2. **Which models produce most human-like text?**
   - Compare confusion matrices across Llama, Mistral, Gemma, etc.

3. **What linguistic features distinguish human from AI?**
   - Feature importance from Random Forest
   - Significant Empath categories

4. **How do different selection strategies compare?**
   - ML-based vs cosine similarity vs random baseline

5. **Do BERT and Random Forest agree?**
   - Agreement analysis reveals consistent vs inconsistent patterns
   - Models agreeing but wrong = systematic blind spots

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pagan2025computational,
  title   = {Computational Turing Test Reveals Systematic Differences Between Human and AI Language},
  author  = {Pagan, Nicolò and Törnberg, Petter and Bail, Christopher A. and Hannák, Anikó and Barrie, Christopher},
  year    = {2025},
  eprint  = {2511.04195},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://arxiv.org/abs/2511.04195}
}
```

## License

This code is released under the [MIT License](LICENSE).

## Contact

Nicolò Pagan — see paper for institutional affiliation and contact details.

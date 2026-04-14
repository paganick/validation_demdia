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
  - [Phase 1: Preprocessing](#phase-1-preprocessing)
  - [Phase 2: Simulation](#phase-2-simulation)
  - [Phase 3: Postprocessing](#phase-3-postprocessing)
  - [Phase 4: Plotting and Analysis](#phase-4-plotting-and-analysis)
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
- Optional SHAP analysis for explainability

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

# For SHAP analysis (optional)
conda activate shap_env
```

## Workflow

### Phase 1: Preprocessing

Preprocessing is a **one-time step** performed before any simulations. The distributed
dataset already includes pre-generated `personas.pkl` files, so this phase only needs
to be repeated if you are working with a new dataset or platform.

**Step 1a — Assign users to batches:**

```bash
python preprocessing/prepare_user_batches.py
```

Creates `user_batches.json`, which assigns users to fixed reproducible batches (≤ 60
users each). This is required for fine-tuned configurations to ensure each batch trains
its own LoRA adapter on a consistent set of users.

| Platform | Total users | Batches |
|----------|-------------|---------|
| Twitter  | 279         | 5       |
| Reddit   | 492         | 9       |
| Bluesky  | 60          | 1       |

**Step 1b — Generate persona descriptions:**

Each user is represented by a persona description — a paragraph summarising their
communication style and interests, used to condition the LLM.

```bash
python preprocessing/generate_personas_llama.py --platform all
```

Reads `data/{platform}/posts.pkl`, writes `data/{platform}/personas_llama.pkl`
(third-person, one row per user).

**Step 1c — Transform to second-person:**

```bash
python preprocessing/transform_llama_personas_to_second_person.py --platform all
```

Adds a `persona_third_person` column (original) and overwrites `persona` with the
second-person version required by instruction-tuned prompts.

**Step 1d — Promote to canonical file:**

```bash
cp data/{platform}/personas_llama.pkl data/{platform}/personas.pkl
```

After this, `personas.pkl` contains the two columns expected by `run_simulation.py`:
- `persona` — second-person ("You are @User_XXXX…"), for instruction-tuned models
- `persona_third_person` — third-person ("@User_XXXX is…"), for base models

Both persona scripts accept `--data-dir <path>` to target a custom data directory.

---

### Phase 2: Simulation

**Step 2a — Run simulations (one process per platform × config × batch):**

```bash
python run_simulation.py \
    --config_file=configs/<model>.yaml \
    --dataset=<platform> \
    --user_batch=<N> \
    --output_dir=results
```

Re-running is safe — already-complete users are skipped automatically. Repeat as
needed until all batches are fully populated.

**Fine-tuning note:** when multiple batches run in parallel, the first job to need a
fine-tuned model trains it; the others wait via file lock and load it once training
finishes. Each batch trains on its own users and saves to a separate directory.

**Inputs:**
- Config YAMLs in `configs/` (one per model/configuration)
- `data/{platform}/posts.pkl`, `data/{platform}/personas.pkl`
- `user_batches.json`

#### Running on a SLURM cluster

Generate the full task list (one row per config × platform × batch):

```bash
python join_complete_batches.py --list-tasks > tasks.tsv
```

Then submit as an array job:

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
    python run_simulation.py \
        --config_file  "configs/$CONFIG" \
        --dataset      "$PLATFORM" \
        --output_dir   "results/$PLATFORM" \
        --user_batch   "$BATCH" \
        --batch_file   user_batches.json \
        --n_responses_per_user 20
```

Monitor batch completion with:

```bash
python analyze_pipeline_status.py
```

**Step 2b — Join completed batches:**

```bash
python join_complete_batches.py --output-dir results_joined/
```

Skips any config/platform combination where not all batch files are present.
Use `--dry-run` to preview without writing. Output:
- `results_joined/{platform}/{vendor}/{ModelName}__{config_flags}__random_response.json`

---

### Phase 3: Postprocessing

**Step 3a — Clean responses:**

Strip formatting artifacts introduced by instruction-tuned models (e.g. `[Response]`
headers, bold wrappers, handle prefixes):

```bash
python pipeline/response_cleaning.py results_joined/ --output-dir results_cleaned/
```

Always specify `--output-dir` to keep `results_joined/` intact as a raw backup.

**Step 3b — Select best responses:**

```bash
python pipeline/LLM_judge.py results_cleaned/ --include-advanced
```

Uses ML (Random Forest on 20 linguistic features) and cosine similarity
(all-MiniLM-L6-v2) to select the most human-like candidate per prompt. Output:
- `*_optimal_response.json`, `*_response_comparisons.csv`, `*_responses_features.csv`

**Step 3c — Convert to CSV:**

```bash
python pipeline/optimal_responses_to_csv.py results_cleaned/
```

Converts each `*_optimal_response.json` into a sibling `*_optimal_response.csv`.

**Step 3d — Build validation datasets:**

```bash
python pipeline/build_validation_data.py --folder=results_cleaned/bluesky
python pipeline/build_validation_data.py --folder=results_cleaned/twitter
python pipeline/build_validation_data.py --folder=results_cleaned/reddit
```

Creates balanced human-vs-AI datasets for each platform and response type
(`*_ml_validation_data.csv`, `*_cosine_validation_data.csv`, `*_random_validation_data.csv`).

**Step 3e — Run BERT and Empath validation:**

```bash
python pipeline/validate_text.py --input_dir=results_cleaned/bluesky/ --validation=all
python pipeline/validate_text.py --input_dir=results_cleaned/twitter/ --validation=all
python pipeline/validate_text.py --input_dir=results_cleaned/reddit/ --validation=all
```

Runs BERT classification (5 runs, median selected) and Empath linguistic analysis.
Output: `*_labelled.csv`, `*_trainer_results.json`, `*_confusion_matrix.csv`,
`*_significant_features.csv`.

**Step 3f — Feature analysis:**

```bash
python pipeline/features_analysis.py compute_features results_cleaned/
python pipeline/features_analysis.py evaluate results_cleaned/ labels
```

Extracts 20 linguistic features and trains a Random Forest classifier.

---

### Phase 4: Plotting and Analysis

```bash
python analysis/post_process.py results_cleaned/
python analysis/generate_SOTA_plots.py results_cleaned/
python analysis/generate_config_optimal_plots.py results_cleaned/
python analysis/compute_cosine_baselines.py results_cleaned/
python analysis/analyze_feature_differences.py results_cleaned/
```

`post_process.py` aggregates per-file validation results into a single
`summary_metrics.csv`. The remaining scripts produce publication-ready figures.

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
├── data/                              # Input datasets (anonymized)
│   ├── bluesky/posts.pkl, personas.pkl
│   ├── twitter/posts.pkl, personas.pkl
│   └── reddit/posts.pkl, personas.pkl
│
│   — Phase 2: Simulation entry points —
├── run_simulation.py                  # Generate AI responses (one batch per run)
├── join_complete_batches.py           # Merge completed batches → results_joined/
├── analyze_pipeline_status.py         # Monitor batch completion and SLURM queue
│
│   — Shared library —
├── src/
│   ├── agent.py                       # User agent: loads posts and persona, formats prompts
│   ├── model.py                       # LLM wrapper: loading, inference, LoRA fine-tuning
│   ├── simulate.py                    # Simulation loop and result serialisation
│   ├── feature_utils.py               # Linguistic feature extraction (20 features)
│   ├── plotting_utils.py              # Shared plotting utilities and filename parsing
│   ├── utils.py                       # Validator class (BERT, Empath)
│   └── config_utils.py, model_utils.py, globals.py
│
│   — Phase 3: Postprocessing —
├── pipeline/
│   ├── response_cleaning.py           # Step 3a: strip formatting artifacts
│   ├── LLM_judge.py                   # Step 3b: select best responses (ML + cosine)
│   ├── optimal_responses_to_csv.py    # Step 3c: convert JSON → CSV
│   ├── build_validation_data.py       # Step 3d: build human vs AI datasets
│   ├── validate_text.py               # Step 3e: BERT + Empath validation
│   └── features_analysis.py           # Step 3f: Random Forest feature analysis
│
│   — Phase 4: Plotting and Analysis —
├── analysis/
│   ├── post_process.py                # Aggregate per-file results → summary_metrics.csv
│   ├── generate_SOTA_plots.py         # Main accuracy and comparison figures
│   ├── generate_config_optimal_plots.py  # Per-configuration optimality plots
│   ├── compute_cosine_baselines.py    # Cosine similarity baseline distributions
│   ├── analyze_feature_differences.py # Feature-level human vs AI differences
│   ├── analyze_ft_hyperparam.py       # Fine-tuning hyperparameter sweep analysis
│   ├── compute_baselines.py           # Baseline metrics
│   └── compute_cosine_similarities.py # Cosine similarity calculations
│
│   — Phase 1: Preprocessing (one-time) —
└── preprocessing/
    ├── prepare_user_batches.py         # Step 1a: assign users to reproducible batches
    ├── generate_personas_llama.py      # Step 1b: generate persona descriptions (Llama)
    ├── transform_llama_personas_to_second_person.py  # Step 1c: third → second person
    ├── anonymize_usernames.py          # Documentation only — run once on original data
    └── anonymize_mentions.py           # Documentation only — run once on original data
```

## Usage Examples

### Generate responses for a single configuration

```bash
python run_simulation.py \
    --config_file configs/llama_3_1_8b_base.yaml \
    --dataset bluesky \
    --user_batch 0 \
    --output_dir results_test
```

### Force reprocess all LLM Judge results

```bash
python pipeline/LLM_judge.py results --force-reprocess --include-advanced
```

### Validate with only BERT (skip Empath)

```bash
python pipeline/validate_text.py --input_dir=results/results_bluesky/ --validation=bert
```

### Validate with only Empath (skip BERT)

```bash
python pipeline/validate_text.py --input_dir=results/results_bluesky/ --validation=empath
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

### Summary Files

- `summary_metrics.csv`: Aggregated results across all configurations
  - Model configuration columns (model, finetuning, context, style, OPPU)
  - Confusion matrix cells (0-0, 1-0, 0-1, 1-1)
  - Stylistic metrics (avg_words, avg_links, avg_emojis, avg_mentions)

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

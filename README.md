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
python prepare_user_batches.py
```

**Run each batch** (one process per platform × config × batch, parallelise as your
cluster allows):

```bash
python run_simulation.py \
    --config_file=configs/<model>.yaml \
    --dataset=<platform> \
    --user_batch=<N> \
    --output_dir=results
```

Depending on model size and available GPU memory, runs may time out before completing
all users. Re-running the same command is safe — already-complete users are skipped
automatically. Repeat as needed until all batches are fully populated.

**After all batches are complete, merge into per-config files:**

```bash
python merge_batch_results.py --results_dir results/ --delete_batches
```

This produces `results/{platform}/{config}__random_response.json`.

**Fine-tuning coordination:** when multiple batches run in parallel, the first job that
needs a fine-tuned model trains it; the others wait via file lock and load it once
training finishes. Each batch trains on its own users and saves to a separate directory.

**Input:**
- Config YAMLs in `configs/` (one per model/configuration)
- User datasets: `bluesky_data/personas.pkl`, Twitter data, Reddit data
- Batch assignments: `user_batches.json`

**Output:**
- `results/{platform}/{config}__random_response.json` (after merging)

### Step 2: Select Best Responses

Use ML and cosine similarity to select most human-like responses:

```bash
python LLM_judge.py results --include-advanced
```

**Input:**
- `*_random_response.json` files

**Output:**
- `*_optimal_response.json` (enriched with ML_best and cosine_best)
- `*_response_comparisons.csv` (side-by-side comparison)
- `*_responses_features.csv` (cached feature matrix)

### Step 3: Convert Optimal Responses to CSV

```bash
python optimal_responses_to_csv.py results_cleaned/
```

Converts each `*_optimal_response.json` produced by `LLM_judge.py` into a sibling `*_optimal_response.csv` file.

### Step 4: Build Validation Datasets

Create balanced datasets (human vs AI) for each platform:

```bash
# Bluesky
python build_validation_data.py --folder=results_cleaned/bluesky

# Twitter
python build_validation_data.py --folder=results_cleaned/twitter

# Reddit
python build_validation_data.py --folder=results_cleaned/reddit
```

**Output:**
- `*_ml_validation_data.csv` (ML-selected responses)
- `*_cosine_validation_data.csv` (Cosine-selected responses)
- `*_random_validation_data.csv` (Baseline)

### Step 5: Run Validation

```bash
conda activate shap_env

python validate_text.py --input_dir=results/results_bluesky/ --validation=all
python validate_text.py --input_dir=results/results_twitter/ --validation=all
python validate_text.py --input_dir=results/results_reddit/ --validation=all
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

### Step 6: Feature Analysis

Compute linguistic features and train Random Forest:

```bash
# Compute and cache features
python features_analysis.py compute_features results

# Train RF on ground truth labels
python features_analysis.py evaluate results labels

# Optional: Train RF on BERT predictions
python features_analysis.py evaluate results bert_prediction
```

### Step 7: Generate Analysis Plots

```bash
python generate_SOTA_plots.py results
python generate_config_optimal_plots.py results
```

Creates publication-ready figures for research papers.

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
├── configs/                      # Model configuration YAMLs
│   ├── llama3.1_base.yaml
│   ├── mistral_base.yaml
│   └── ...
├── src/                          # Core simulation engine
│   ├── agent.py                  # User agent (loads posts, generates responses)
│   ├── model.py                  # LLM wrapper (loading, fine-tuning)
│   ├── simulate.py               # Simulation orchestration
│   └── config_utils.py, model_utils.py, globals.py
├── bluesky_data/                 # Bluesky user data
│   └── personas.pkl
├── run_simulation.py             # Main entry point for generation
├── LLM_judge.py                  # ML-based response selection
├── build_validation_data.py      # Create human vs AI datasets
├── validate_text.py              # BERT + Empath validation
├── features_analysis.py          # Random Forest feature analysis
├── feature_utils.py              # Feature extraction functions
├── post_process.py               # Aggregate validation results
├── optimal_responses_to_csv.py   # Convert optimal_response.json → CSV
├── parse_reddit_data.py          # Parse raw Reddit JSON
├── aggregate_reddit_data.py      # Generate Reddit personas
├── plotting_utils.py             # Shared plotting utilities
├── generate_SOTA_plots.py        # Main research figures
├── generate_config_optimal_plots.py  # Configuration comparisons
└── utils.py                      # Validator class (BERT, Empath)
```

## Usage Examples

### Generate responses for a single configuration

```bash
python run_simulation.py \
    --config_dir=configs \
    --config_file=llama3.1_base.yaml \
    --n_users=50 \
    --output_dir=results_test
```

### Force reprocess all LLM Judge results

```bash
python LLM_judge.py results --force-reprocess --include-advanced
```

### Validate with only BERT (skip Empath)

```bash
python validate_text.py --input_dir=results/results_bluesky/ --validation=bert
```

### Validate with only Empath (skip BERT)

```bash
python validate_text.py --input_dir=results/results_bluesky/ --validation=empath
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

```
[Add citation information once published]
```

## License

[Add license information]

## Contact

[Add contact information]

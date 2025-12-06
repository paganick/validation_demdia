# Consolidated Postprocessing - Usage Guide

## Overview

The new `run_postprocessing_consolidated.py` script replaces the multi-step postprocessing workflow with a single command that generates consolidated outputs.

**Before:** 44 files per configuration
**After:** 4 files per configuration (90% reduction!)

## Quick Start

### Test on Single Model (Recommended First)

```bash
# Test on meta-llama model from bluesky (fastest, skips BERT)
python run_postprocessing_consolidated.py \
    --input results/results_bluesky/meta-llama \
    --dataset bluesky \
    --output test_output \
    --skip-bert \
    --model meta-llama
```

This will:
- Process all Llama configurations in that directory
- Generate 4 files per config in `test_output/bluesky/meta-llama/<config>/`
- Skip BERT validation (fast for testing)
- Show progress and summary

### Full Processing

```bash
# Process entire bluesky dataset with BERT
python run_postprocessing_consolidated.py \
    --input results/results_bluesky \
    --dataset bluesky \
    --output results_consolidated
```

## Command-Line Options

```
--input PATH          Input directory with simulation outputs (required)
--dataset NAME        Dataset name: bluesky, twitter, or reddit (required)
--output PATH         Output directory (default: results_consolidated)
--config PATH         Custom pipeline config YAML (default: pipeline_config.yaml)
--skip-bert           Skip BERT validation (faster for testing)
--model NAME          Process only this model directory
```

## Output Structure

```
test_output/
└── bluesky/
    └── meta-llama/
        ├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU/
        │   ├── responses.json              # All selection methods
        │   ├── validation_results.json     # BERT results for all methods
        │   ├── features.parquet            # Features for all methods
        │   └── statistics.json             # Stats for all methods
        ├── Llama-3.1-8B__ft__ctx1__style10__no_OPPU/
        │   └── ... (same 4 files)
        └── ...
```

## Verifying Output

### Check Generated Files

```bash
# List generated structure
ls -R test_output/bluesky/meta-llama/ | head -50

# Check one configuration
ls -lh test_output/bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU/
```

### Load and Inspect Results

```python
import json
import pandas as pd
from pathlib import Path

# Path to a configuration
config_dir = Path('test_output/bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU')

# 1. Load responses
with open(config_dir / 'responses.json') as f:
    responses = json.load(f)
    print(f"Config: {responses['config_name']}")
    print(f"Responses: {len(responses['responses'])}")
    print(f"Selection methods: {list(responses['responses'][0]['selected_indices'].keys())}")

# 2. Load validation results
with open(config_dir / 'validation_results.json') as f:
    validation = json.load(f)
    for method in ['random', 'ml', 'cosine']:
        if method in validation['results']:
            acc = validation['results'][method]['bert_metrics']['accuracy']
            print(f"{method} accuracy: {acc:.3f}")

# 3. Load features
features = pd.read_parquet(config_dir / 'features.parquet')
print(f"\nFeatures shape: {features.shape}")
print(f"Selection methods: {features['selection_method'].unique()}")
print(f"\nFeature columns:")
print(features.columns.tolist())

# 4. Load statistics
with open(config_dir / 'statistics.json') as f:
    stats = json.load(f)
    for method in stats['statistics']:
        corrs = stats['statistics'][method]['correlation_stats']
        print(f"\n{method} feature correlations: {len(corrs)}")
```

## Current Implementation Status

### ✅ Working Features:
- Configuration detection and processing
- Consolidated response format
- Feature extraction (basic features: word_count, char_count, avg_word_length)
- Statistics computation (correlations, importance placeholders)
- Parallel processing of multiple configurations
- Progress tracking and error handling

### ⚠️ Mock/Placeholder Features:
- **BERT Validation**: Currently returns mock results
  - Integration point: `run_bert_validation_simple()`
  - Replace with: `validation.utils.Validator.bert_validate()`

- **Feature Extraction**: Basic features only
  - Integration point: `extract_features_simple()`
  - Add: Empath categories, sentiment, linguistic features
  - From: `validation/feature_utils.py` and `validation/features_analysis.py`

- **Statistics**: Placeholders
  - Integration point: `compute_statistics_simple()`
  - Add: Median run agreement, Empath significant features
  - From: `validation/features_analysis.py`

### 🔧 To Extend:

#### 1. Add Full BERT Validation

Replace the mock function in `run_postprocessing_consolidated.py`:

```python
def run_bert_validation_full(validation_df: pd.DataFrame) -> Dict[str, Any]:
    """Full BERT validation using existing infrastructure."""
    from transformers import BertTokenizer
    from validation.utils import Validator

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Run full BERT validation
    trainer, report, cm, results, shap_data, shap_stats = Validator.bert_validate(
        validation_df,
        tokenizer
    )

    return {
        'metrics': report,
        'confusion_matrix': cm.tolist(),
        'trainer_results': results,
        'shap_values': shap_data
    }
```

Then replace the call:
```python
# OLD:
bert_results = run_bert_validation_simple(validation_df)

# NEW:
bert_results = run_bert_validation_full(validation_df)
```

#### 2. Add Full Feature Extraction

```python
def extract_features_full(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature extraction."""
    from validation.feature_utils import extract_all_features

    # Use existing feature extraction
    features_df = extract_all_features(validation_df)

    return features_df
```

#### 3. Add Full Statistics

```python
def compute_statistics_full(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, Any]:
    """Full statistical analysis."""
    from validation.features_analysis import compute_feature_stats

    stats = compute_feature_stats(features_df, labels_df)

    return stats
```

## Comparison: Old vs New Workflow

### Old Workflow (44 files per config):
```bash
# Multiple scripts, many files
python LLM_judge.py
python build_validation_data.py
python validate_text.py
python features_analysis.py
python post_process.py

# Result: 44 files scattered across directories
results/bluesky/meta-llama/
├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU__random_response.json
├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU__optimal_response.json
├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU__random_validation_data.csv
├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU__ml_validation_data.csv
├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU__cosine_validation_data.csv
├── ... (39 more files)
```

### New Workflow (4 files per config):
```bash
# Single command
python run_postprocessing_consolidated.py \
    --input results/results_bluesky \
    --dataset bluesky

# Result: 4 organized files
test_output/bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU/
├── responses.json              # All methods, all data
├── validation_results.json     # All BERT results
├── features.parquet            # All features (5-10x smaller!)
└── statistics.json             # All stats
```

## Performance

- **90% fewer files**: 6,000 → 540 files for full dataset
- **5-10x smaller**: Parquet format vs CSV
- **Faster loading**: 4 files vs 44 files per config
- **Easier navigation**: Clean directory structure

## Troubleshooting

### Import Errors

If you get import errors from the `validation` module loading BERT models:

```bash
# Option 1: Skip BERT for testing
python run_postprocessing_consolidated.py --skip-bert ...

# Option 2: Fix imports (if __init__.py loads models)
# Comment out heavy imports in validation/__init__.py
```

### Memory Issues

For large datasets:

```bash
# Process one model at a time
python run_postprocessing_consolidated.py \
    --input results/results_bluesky/meta-llama \
    --model meta-llama \
    ...
```

### Missing Dependencies

```bash
pip install pandas pyarrow transformers datasets
```

## Next Steps

1. **Test the script**: Run on one model with `--skip-bert`
2. **Verify outputs**: Check the 4 generated files
3. **Integrate full validation**: Replace mock functions
4. **Run on full dataset**: Process all configurations
5. **Update plots**: Use new consolidated format

## Documentation

- **`POSTPROCESSING_REDESIGN.md`** - Complete design
- **`POSTPROCESSING_SUMMARY.md`** - Quick overview
- **`validation/consolidated_outputs.py`** - Data structures
- **`run_postprocessing_consolidated.py`** - Main script

---

**Questions?** Check the inline comments in `run_postprocessing_consolidated.py` - they mark all integration points with `TODO` comments.

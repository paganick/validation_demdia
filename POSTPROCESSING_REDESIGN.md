# Postprocessing Pipeline Redesign

## Problem

Current postprocessing generates **44 files per configuration**:
- 13 files × 3 selection methods (random, ml, cosine) = 39 files
- Plus 5 shared files (responses, comparisons, etc.)

For 45 configurations per dataset × 3 datasets = **~6,000 files total**!

This causes:
- Slow file I/O
- Difficult to navigate
- Lots of duplication
- Hard to manage

## Solution: Consolidated Output Format

**New structure:** 4 files per configuration (11x reduction!)

```
results/bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU/
├── responses.json              # All selection methods (random, ml, cosine)
├── validation_results.json     # All BERT/validation for all methods
├── features.parquet            # All features for all methods
└── statistics.json             # All stats (confusion, correlations, etc.)
```

**Benefits:**
- **6,000 → 540 files** (90% reduction!)
- Faster I/O (fewer files to open)
- Easy to navigate (4 files per config)
- No duplication (selection methods in same file)
- Easy to add new configs or models

## File Descriptions

###  1. responses.json
Contains all response data and selections for all methods.

**Structure:**
```json
{
  "config_name": "Llama-3.1-70B__ft__ctx1__style10__no_OPPU",
  "model": "meta-llama/Llama-3.1-70B",
  "metadata": {
    "fine_tuned": true,
    "retrieve_context": true,
    "n_style_examples": 10
  },
  "responses": [
    {
      "user": "user123",
      "prompt_id": "user123___msg456",
      "candidates": ["response1", "response2", "response3"],
      "selected_indices": {
        "random": 0,
        "ml": 1,
        "cosine": 2
      },
      "scores": {
        "ml": [0.3, 0.5, 0.2],
        "cosine": [0.7, 0.2, 0.9]
      }
    }
  ]
}
```

### 2. validation_results.json
Contains all BERT validation results for all methods.

**Replaces:**
- `{method}_validation_data_bert_report.json` (×3)
- `{method}_validation_data_confusion_matrix.csv` (×3)
- `{method}_validation_data_bert_shap_analysis.json` (×3)
- `{method}_validation_data_trainer_results.json` (×3)

**Structure:**
```json
{
  "config_name": "Llama-3.1-70B__ft__ctx1__style10__no_OPPU",
  "model": "meta-llama/Llama-3.1-70B",
  "results": {
    "random": {
      "selection_method": "random",
      "bert_metrics": {
        "accuracy": 0.85,
        "f1_score": 0.82,
        "precision": 0.84,
        "recall": 0.80
      },
      "confusion_matrix": [[45, 5], [8, 42]],
      "predictions": [...],  // optional
      "shap_values": {...}   // optional
    },
    "ml": { ... },
    "cosine": { ... }
  }
}
```

### 3. features.parquet
Contains all features for all methods in efficient binary format.

**Replaces:**
- `{method}_validation_data_features.csv` (×3)
- `{method}_validation_data_features_with_text_and_label.csv` (×3)
- `{method}_validation_data_labelled.csv` (×3)

**Structure:** Parquet file with columns:
```
selection_method | user | prompt_id | response | label | feature1 | feature2 | ...
random          | user1| prompt1   | text...  | 0     | 0.5      | 0.3      | ...
ml              | user1| prompt1   | text...  | 1     | 0.6      | 0.4      | ...
cosine          | user1| prompt1   | text...  | 1     | 0.7      | 0.5      | ...
```

**Benefits of Parquet:**
- 5-10x smaller than CSV
- 10-100x faster to read
- Preserves column types
- Built-in compression

### 4. statistics.json
Contains all statistical analyses for all methods.

**Replaces:**
- `{method}_validation_data_labelled_from_labels_feature_correlation_stats.csv` (×3)
- `{method}_validation_data_labelled_from_labels_feature_importance_stats.csv` (×3)
- `{method}_validation_data_empath_significant_features.csv` (×3)
- `{method}_validation_data_from_labels_median_run_agreement.csv` (×3)
- `{method}_validation_data_from_labels_median_run_results.csv` (×3)

**Structure:**
```json
{
  "config_name": "Llama-3.1-70B__ft__ctx1__style10__no_OPPU",
  "model": "meta-llama/Llama-3.1-70B",
  "statistics": {
    "random": {
      "correlation_stats": [...],
      "importance_stats": [...],
      "empath_significant": [...],
      "agreement_stats": [...]
    },
    "ml": { ... },
    "cosine": { ... }
  }
}
```

## Usage Examples

### Loading Results

```python
import json
import pandas as pd
from pathlib import Path
from validation.consolidated_outputs import ConsolidatedValidation, load_consolidated_features

# Load validation results
config_dir = Path('results/bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU')

# 1. Load validation
with open(config_dir / 'validation_results.json') as f:
    validation = json.load(f)

# Access ML method results
ml_accuracy = validation['results']['ml']['bert_metrics']['accuracy']
ml_confusion = validation['results']['ml']['confusion_matrix']

print(f"ML Accuracy: {ml_accuracy}")
print(f"ML Confusion Matrix: {ml_confusion}")

# 2. Load features (all methods)
features_all = pd.read_parquet(config_dir / 'features.parquet')
print(f"Total feature rows: {len(features_all)}")

# Load features for specific method
features_ml = features_all[features_all['selection_method'] == 'ml']
print(f"ML feature rows: {len(features_ml)}")

# 3. Load statistics
with open(config_dir / 'statistics.json') as f:
    stats = json.load(f)

# Get significant Empath features for cosine method
cosine_empath = stats['statistics']['cosine']['empath_significant']
print(f"Significant Empath features (cosine): {len(cosine_empath)}")
```

### Comparing Selection Methods

```python
import json
import pandas as pd

config_dir = Path('results/bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU')

# Load validation
with open(config_dir / 'validation_results.json') as f:
    validation = json.load(f)

# Compare accuracy across methods
methods = ['random', 'ml', 'cosine']
accuracies = {
    method: validation['results'][method]['bert_metrics']['accuracy']
    for method in methods
}

print("Accuracy by method:")
for method, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method}: {acc:.3f}")

# Compare confusion matrices
for method in methods:
    cm = validation['results'][method]['confusion_matrix']
    print(f"\n{method} confusion matrix:")
    print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}, TP={cm[1][1]}")
```

### Working with Features

```python
import pandas as pd

# Load features
features = pd.read_parquet('results/bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU/features.parquet')

# Analyze feature differences by method
for method in ['random', 'ml', 'cosine']:
    method_features = features[features['selection_method'] == method]

    print(f"\n{method} statistics:")
    print(f"  Mean feature1: {method_features['feature1'].mean():.3f}")
    print(f"  Mean feature2: {method_features['feature2'].mean():.3f}")

# Compare same prompt across methods
prompt_id = 'user123___msg456'
prompt_features = features[features['prompt_id'] == prompt_id]

print(f"\nFeatures for {prompt_id} across methods:")
print(prompt_features[['selection_method', 'feature1', 'feature2']])
```

## Implementation Status

✅ **Completed:**
- Consolidated data structures (`validation/consolidated_outputs.py`)
- Configuration system (`pipeline_config.yaml`)
- Basic conversion pipeline (`run_postprocessing.py`)

⏳ **TODO:**
- Extend `run_postprocessing.py` to run validation and generate consolidated outputs
- Add validation logic (BERT, features, statistics)
- Test on sample configuration
- Migrate existing results (optional)

## Next Steps

1. **Extend run_postprocessing.py** to generate consolidated outputs
2. **Test on bluesky** dataset with one configuration
3. **Verify results** match old format
4. **Run on all datasets** when validated
5. **Update plotting scripts** to use new format

## Migration Strategy

**For existing results:**
```bash
# Option 1: Keep old results as backup, run new pipeline
mv results/results_bluesky results/results_bluesky_old
python run_postprocessing.py --input simulation_outputs/bluesky --dataset bluesky

# Option 2: Migrate existing postprocessing results
python migrate_postprocessing_results.py results/results_bluesky
```

**For new simulations:**
```bash
# Run simulation (generates *_optimal_response.json)
python run_simulation.py --dataset bluesky --model llama

# Run consolidated postprocessing
python run_postprocessing.py --input results/bluesky --dataset bluesky
```

Result: Clean 4-file structure automatically!

## Configuration

Edit `pipeline_config.yaml` to control output:

```yaml
storage:
  use_consolidated_format: true      # NEW format (4 files)
  keep_legacy_csv: false              # Don't generate old CSV files
  save_feature_cache: false           # Don't save intermediate files

processing:
  selection_methods: [random, ml, cosine]
  validation_methods: [bert, features, empath]
  consolidate_validation: true        # Consolidate per config

outputs:
  save_confusion_matrices: true       # Include in validation_results.json
  save_feature_importance: true       # Include in statistics.json
  save_predictions: false             # Skip individual predictions
```

---

**File Reduction Summary:**
```
Old: 44 files × 45 configs × 3 datasets = 5,940 files
New: 4 files × 45 configs × 3 datasets = 540 files

🎉 90% reduction!
```

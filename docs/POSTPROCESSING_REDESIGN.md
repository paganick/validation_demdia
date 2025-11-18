# Postprocessing Pipeline Redesign

## Problem Statement

Current postprocessing generates 40-50 files per configuration:
- Duplicate data across random/ml/cosine variants
- Same user/message data repeated in every file
- Scattered validation results
- Hard to manage and navigate

## Proposed Solution

### 1. Consolidated Data Structure

**Before:**
```
results/bluesky/llama-3.1-8B/
├── config1_random_response.json
├── config1_optimal_response.json
├── config1_response_comparisons.csv
├── config1_responses_features.csv
├── config1_ml_validation_data.csv
├── config1_cosine_validation_data.csv
└── ... (40+ more files)
```

**After:**
```
results/bluesky/
├── shared/
│   ├── users.json              # User metadata (personas) - stored ONCE
│   └── prompts.json            # Original messages/prompts - stored ONCE
│
├── responses/
│   └── llama-3.1-8B_base_ctx0_style0.json
│       {
│         "metadata": {...},
│         "responses": [
│           {
│             "user": "user123",
│             "reply_to_id": "msg456",
│             "candidates": ["resp1", "resp2", ...],
│             "selected": {
│               "random": 0,           # Index in candidates
│               "ml": 2,
│               "cosine": 1
│             },
│             "scores": {
│               "ml_probabilities": [0.3, 0.5, 0.7, ...],
│               "cosine_similarities": [0.6, 0.8, 0.4, ...]
│             }
│           }
│         ]
│       }
│
└── validation/
    └── results.json
        {
          "llama-3.1-8B_base_ctx0_style0": {
            "random": {
              "bert": {...},
              "features": {...},
              "metrics": {...}
            },
            "ml": {...},
            "cosine": {...}
          }
        }
```

### 2. Benefits

1. **Reduced duplication**: Users/prompts stored once
2. **Better organization**: Clear separation of responses vs validation
3. **Easier analysis**: All validation results in one file
4. **More compact**: ~3-5 files instead of 40-50

### 3. Migration Strategy

1. Create data structure utilities
2. Add backward compatibility layer
3. Migrate existing scripts incrementally
4. Provide migration tool for old data

### 4. Pipeline Configuration

```yaml
# pipeline_config.yaml
storage:
  use_consolidated_format: true
  keep_legacy_csv: false          # Don't generate CSV files

processing:
  selection_methods: [random, ml, cosine]
  validation_methods: [bert, features]

outputs:
  save_comparison_csv: false      # Skip comparison tables
  save_feature_cache: false       # Don't save intermediate features
  consolidate_validation: true    # One validation file per dataset
```

## Implementation Plan

1. ✅ Create design document
2. Create consolidated data structure utilities
3. Create pipeline configuration system
4. Refactor LLM_judge.py to use new structure
5. Refactor build_validation_data.py
6. Refactor validate_text.py
7. Create unified pipeline runner
8. Test and document

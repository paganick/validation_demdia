# Quick Start: Migration & Testing

## Summary

✅ Successfully migrated **Bluesky** dataset to the new consolidated format!

**Results:**
- 45 configurations migrated
- 59 unique users
- 1,179 unique prompts
- New location: `results_consolidated/bluesky/`

## Migrate All Datasets

Use the unified migration script for all datasets:

```bash
# Migrate individual datasets
python migrate_results_simple.py bluesky   # ✅ Already done!
python migrate_results_simple.py twitter
python migrate_results_simple.py reddit

# Or migrate all at once
python migrate_results_simple.py all
```

## New Data Structure

```
results_consolidated/
├── bluesky/           ✅ MIGRATED
│   ├── shared/
│   │   ├── users.json      (59 users)
│   │   └── prompts.json    (1,179 prompts)
│   └── responses/
│       └── *.json          (45 files)
├── twitter/           ⏳ TODO
│   └── ...
└── reddit/            ⏳ TODO
    └── ...
```

## What Changed?

**Before (Legacy):**
```
results/results_bluesky/meta-llama/
├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU__optimal_response.json
├── Llama-3.1-70B__ft__ctx1__style10__no_OPPU__ml_validation_data.csv
├── ... (many files per config)
```

**After (Consolidated):**
```
results_consolidated/bluesky/
├── shared/
│   ├── users.json          # All users (shared across configs)
│   └── prompts.json        # All prompts (shared across configs)
└── responses/
    └── Llama-3.1-70B__ft__ctx1__style10__no_OPPU_.json  # All selections
```

## Load New Format (Python)

```python
import json

# Load shared data (once for all configs)
with open('results_consolidated/bluesky/shared/users.json') as f:
    users = json.load(f)

with open('results_consolidated/bluesky/shared/prompts.json') as f:
    prompts = json.load(f)

# Load a specific configuration
with open('results_consolidated/bluesky/responses/Llama-3.1-70B__ft__ctx1__style10__no_OPPU_.json') as f:
    config = json.load(f)

print(f"Config: {config['config_name']}")
print(f"Model: {config['model']}")
print(f"Responses: {len(config['responses'])}")

# Access response data
for resp in config['responses']:
    # Get selected responses
    candidates = resp['candidates']
    selected = resp['selected_indices']

    if 'random' in selected:
        random_resp = candidates[selected['random']]
    if 'ml' in selected:
        ml_resp = candidates[selected['ml']]
    if 'cosine' in selected:
        cosine_resp = candidates[selected['cosine']]
```

## Test Plotting

Update your plotting scripts to use the new paths:

```python
# OLD
data_file = "results/results_bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU__optimal_response.json"

# NEW
data_file = "results_consolidated/bluesky/responses/Llama-3.1-70B__ft__ctx1__style10__no_OPPU_.json"
users_file = "results_consolidated/bluesky/shared/users.json"
prompts_file = "results_consolidated/bluesky/shared/prompts.json"
```

## Verify Data Integrity

```bash
# Count original files
find results/results_bluesky -name "*_optimal_response.json" | wc -l

# Count migrated files
find results_consolidated/bluesky/responses -name "*.json" | wc -l

# Should be the same number!
```

## Benefits

1. ✅ **35% reduction in file size** (no duplication)
2. ✅ **Easier to query** (single JSON per config)
3. ✅ **Cleaner structure** (shared vs. config-specific data)
4. ✅ **Same data** (all selections and scores preserved)

## Documentation

- `MIGRATION_TESTING_GUIDE.md` - Detailed migration guide
- `NEW_FORMAT_USAGE.md` - Usage examples and API
- `migrate_results_simple.py` - Migration script (works for all datasets)

## Next Steps

1. ✅ Bluesky migrated and verified
2. ⏳ Migrate Twitter: `python migrate_results_simple.py twitter`
3. ⏳ Migrate Reddit: `python migrate_results_simple.py reddit`
4. ⏳ Update plotting scripts
5. ⏳ Verify plots are identical

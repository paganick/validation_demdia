# Migration Testing Guide

This guide shows how to migrate existing results to the new consolidated format and verify everything works correctly.

## Overview

You have existing results in:
- `results/results_bluesky/`
- `results/results_twitter/`
- `results/results_reddit/`

These will be migrated to a new consolidated structure:
```
results_consolidated/
├── bluesky/
│   ├── shared/
│   │   ├── users.json
│   │   └── prompts.json
│   └── responses/
│       ├── config1.json
│       ├── config2.json
│       └── ...
├── twitter/
│   └── ...
└── reddit/
    └── ...
```

## Quick Start: Test with Bluesky

### Step 1: Migrate Bluesky Dataset

```bash
# Migrate bluesky results to new format
python migrate_results_simple.py bluesky
```

This will:
- Convert all response JSON files to consolidated format
- Extract and save shared users.json
- Extract and save shared prompts.json
- Create organized directory structure

### Step 2: Verify Migration

```bash
# Check the new structure
ls -lh results_consolidated/bluesky/shared/
ls -lh results_consolidated/bluesky/responses/ | head -10

# Count files
echo "Original JSON files:"
find results/results_bluesky -name "*_optimal_response.json" -o -name "*_random_response.json" | wc -l

echo "Migrated JSON files:"
find results_consolidated/bluesky/responses -name "*.json" | wc -l
```

### Step 3: Test Plotting Pipeline

Now you can test the new plotting pipeline using the consolidated format:

```bash
# Use the new consolidated data for plotting
# This should produce identical plots to before

# If you have a plotting script, update it to use:
# results_consolidated/bluesky/ instead of results/results_bluesky/
```

## Full Migration (All Datasets)

Once you've verified bluesky works, you can migrate the other datasets:

```bash
# Migrate Twitter
python migrate_results_simple.py twitter

# Migrate Reddit
python migrate_results_simple.py reddit

# Or migrate all three at once!
python migrate_results_simple.py all
```

## Verification Checklist

- [ ] All JSON files migrated (count matches)
- [ ] users.json contains all users from original data
- [ ] prompts.json contains all prompts from original data
- [ ] Response files are accessible and valid JSON
- [ ] Plots generated from new format match old plots

## What Gets Migrated

### From Legacy Format:
- `*_optimal_response.json` or `*_random_response.json`
  - User information
  - Prompt text and IDs
  - Response candidates
  - Selection methods (random, ML, cosine)
  - Scores (probabilities, similarities)

### To Consolidated Format:
- `shared/users.json` - All user metadata (username, persona, platform)
- `shared/prompts.json` - All prompts (prompt_id, user, text)
- `responses/config_name.json` - Response data for each configuration

## Troubleshooting

### No files found
Make sure you're using the correct input path:
```bash
ls results/results_bluesky/*/*.json | head
```

### Import errors
Make sure you're in the project root and have installed dependencies:
```bash
pip install pandas
```

### Data format issues
Check if the legacy JSON files have the expected structure:
```bash
python -c "import json; print(json.load(open('results/results_bluesky/meta-llama/Llama-3.1-70B__ft__ctx1__style10__no_OPPU___optimal_response.json'))[0].keys())"
```

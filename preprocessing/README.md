# Preprocessing Scripts

Scripts for preparing raw social media data for simulation.

## Overview

These scripts transform raw data dumps into the structured format required by the simulation pipeline (pandas DataFrames with 20 messages per user).

## Scripts

### `parse_reddit_data.py`
Parses raw Reddit JSON exports into structured CSV format.

**Usage:**
```bash
python preprocessing/parse_reddit_data.py \\
    --comments reddit_comments.json \\
    --submissions reddit_submissions.json \\
    --output reddit_parsed.csv
```

### `aggregate_reddit_data.py`
Aggregates parsed Reddit data and generates persona descriptions.

**Usage:**
```bash
python preprocessing/aggregate_reddit_data.py \\
    --input reddit_parsed.csv \\
    --output data/reddit/personas.pkl \\
    --min_messages 20
```

## Workflow

```
Raw Data → Parse → Aggregate → Personas.pkl (20 msgs/user) → Simulation
```

### Step-by-Step Example (Reddit)

```bash
# 1. Parse raw Reddit exports
python preprocessing/parse_reddit_data.py \\
    --comments raw_data/reddit_comments.json \\
    --submissions raw_data/reddit_submissions.json \\
    --output intermediate/reddit_parsed.csv

# 2. Aggregate and select 20 messages per user
python preprocessing/aggregate_reddit_data.py \\
    --input intermediate/reddit_parsed.csv \\
    --output data/reddit/personas.pkl \\
    --messages_per_user 20

# 3. Verify output
python -c "import pandas as pd; df = pd.read_pickle('data/reddit/personas.pkl'); \\
           print(f'Users: {df.username.nunique()}, Messages: {len(df)}, Avg: {len(df)/df.username.nunique():.1f}')"
```

## Output Format

The final `.pkl` file contains a DataFrame with **exactly 20 messages per user**:

| Column | Type | Description |
|--------|------|-------------|
| `username` | str | Anonymized user ID |
| `message` | str | Post/comment text |
| `reply_to` | str | Parent message for context |
| `persona` | str | Auto-generated persona description |
| `training` | int | 1=train, 0=test (splits the 20 messages) |

## Current Datasets

- **Reddit**: 492 users × 20 messages = 9,840 total
- **Twitter/X**: 250 users × 20 messages = 5,000 total
- **Bluesky**: 59 users × 20 messages = 1,180 total

## Utilities (in `scripts/`)

### `read_reddidt_data.py`
Simple utility to convert pickle to CSV for inspection.

```bash
python scripts/read_reddidt_data.py
```

### `explore_reddit_data.py`
Jupyter notebook for data exploration.

## Adapting for Other Platforms

To add support for new platforms:

1. Create `parse_[platform]_data.py` following the Reddit pattern
2. Ensure output has the same column structure
3. Select exactly 20 messages per user
4. Split into training/test sets

## Requirements

All preprocessing scripts use standard dependencies from `requirements.txt`.

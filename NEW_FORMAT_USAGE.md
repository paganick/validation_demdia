# Using the New Consolidated Data Format

## Migration Complete! 🎉

The bluesky dataset has been successfully migrated to the new consolidated format:

```
results_consolidated/bluesky/
├── shared/
│   ├── users.json          (59 users, 61 KB)
│   └── prompts.json        (1,179 prompts, 619 KB)
└── responses/
    └── *.json              (45 configuration files)
```

**Statistics:**
- ✅ 45 configurations migrated
- ✅ 59 unique users
- ✅ 1,179 unique prompts
- ✅ All response data preserved (random, ML, cosine selections + scores)

## How to Use the New Format

### Loading Shared Data

```python
import json

# Load all users
with open('results_consolidated/bluesky/shared/users.json') as f:
    users = json.load(f)

# Access user data
for user in users:
    print(f"User: {user['username']}")
    print(f"Platform: {user['platform']}")
    print(f"Persona: {user['persona'][:100]}...")  # First 100 chars
    print()

# Load all prompts
with open('results_consolidated/bluesky/shared/prompts.json') as f:
    prompts = json.load(f)

# Create prompt lookup
prompts_dict = {p['prompt_id']: p for p in prompts}
```

### Loading Response Data

```python
import json

# Load a specific configuration
config_file = 'results_consolidated/bluesky/responses/Llama-3.1-70B__ft__ctx1__style10__no_OPPU_.json'
with open(config_file) as f:
    data = json.load(f)

print(f"Config: {data['config_name']}")
print(f"Model: {data['model']}")
print(f"Metadata: {data['metadata']}")
print(f"Number of responses: {len(data['responses'])}")

# Access individual responses
for response in data['responses']:
    user = response['user']
    prompt_id = response['prompt_id']
    candidates = response['candidates']  # List of all possible responses
    selected = response['selected_indices']  # {'random': idx, 'ml': idx, 'cosine': idx}
    scores = response['scores']  # {'ml': [probs...], 'cosine': [sims...]}

    # Get the selected responses
    if 'random' in selected:
        random_response = candidates[selected['random']]
    if 'ml' in selected:
        ml_response = candidates[selected['ml']]
    if 'cosine' in selected:
        cosine_response = candidates[selected['cosine']]
```

### Example: Comparing Selection Methods

```python
import json
from collections import Counter

config_file = 'results_consolidated/bluesky/responses/Llama-3.1-70B__ft__ctx1__style10__no_OPPU_.json'
with open(config_file) as f:
    data = json.load(f)

# Count how often different selection methods agree
agreements = Counter()

for response in data['responses']:
    selected = response['selected_indices']

    # Check if ML and cosine selected the same response
    if 'ml' in selected and 'cosine' in selected:
        if selected['ml'] == selected['cosine']:
            agreements['ml_cosine'] += 1

    # Check if random matches ML
    if 'random' in selected and 'ml' in selected:
        if selected['random'] == selected['ml']:
            agreements['random_ml'] += 1

print(f"ML and Cosine agreement: {agreements['ml_cosine']} times")
print(f"Random and ML agreement: {agreements['random_ml']} times")
```

### Example: Analyzing Scores

```python
import json
import numpy as np

config_file = 'results_consolidated/bluesky/responses/Llama-3.1-70B__ft__ctx1__style10__no_OPPU_.json'
with open(config_file) as f:
    data = json.load(f)

ml_confidences = []

for response in data['responses']:
    if 'ml' in response['scores']:
        probs = response['scores']['ml']
        # Get the highest probability (confidence)
        confidence = max(probs)
        ml_confidences.append(confidence)

print(f"Average ML confidence: {np.mean(ml_confidences):.3f}")
print(f"Median ML confidence: {np.median(ml_confidences):.3f}")
print(f"Min/Max ML confidence: {np.min(ml_confidences):.3f} / {np.max(ml_confidences):.3f}")
```

## Migrate Other Datasets

To migrate Twitter and Reddit:

```bash
# Migrate Twitter
python migrate_results_simple.py  # Edit the script to change dataset

# Or create separate scripts for each:
```

```python
#!/usr/bin/env python3
# migrate_twitter.py
# (Same as migrate_results_simple.py but with:)
# input_dir = Path("results/results_twitter")
# dataset = "twitter"
```

```python
#!/usr/bin/env python3
# migrate_reddit.py
# (Same as migrate_results_simple.py but with:)
# input_dir = Path("results/results_reddit")
# dataset = "reddit"
```

## Advantages of New Format

1. **Reduced Duplication**: User and prompt data stored once, not repeated in every config file
2. **Easier Access**: Single JSON per configuration instead of multiple files
3. **Better Organization**: Clear separation of shared vs. configuration-specific data
4. **Efficient Storage**: Reduces file size by ~30-40% for large datasets
5. **Easier to Query**: Can load just users/prompts without loading all response data
6. **Consistent Structure**: Same format across all datasets (bluesky, twitter, reddit)

## File Size Comparison

**Old Format (results/results_bluesky):**
- 45+ JSON files scattered across model directories
- Total: ~2.0 GB
- User/prompt data duplicated in every file

**New Format (results_consolidated/bluesky):**
- 2 shared files + 45 response files
- Total: ~1.3 GB (35% reduction)
- User/prompt data stored once

## Next Steps

1. ✅ Bluesky migrated and verified
2. ⏳ Migrate Twitter dataset
3. ⏳ Migrate Reddit dataset
4. ⏳ Update plotting scripts to use new format
5. ⏳ Test that plots are identical

## Testing Plotting

When you're ready to test plotting with the new format, update your plotting scripts to:

```python
# OLD:
old_data_path = "results/results_bluesky/model_name/config_name_optimal_response.json"

# NEW:
new_data_path = "results_consolidated/bluesky/responses/config_name.json"
users_path = "results_consolidated/bluesky/shared/users.json"
prompts_path = "results_consolidated/bluesky/shared/prompts.json"
```

The data structure is similar, so most plotting code should work with minimal changes!

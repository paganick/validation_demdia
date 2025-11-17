# Simulation Package

Core simulation code for generating synthetic social media responses using large language models with various personalization techniques.

## Overview

This package simulates how different LLMs respond to social media messages, testing various personalization approaches (personas, style examples, context retrieval, fine-tuning).

## Main Components

### `run_simulation.py` (Entry Point - in project root)
Main script to run simulations.

**Usage:**
```bash
# Single configuration
python run_simulation.py \
    --config configs/llama3.1_base.yaml \
    --data_file data/bluesky/personas.pkl \
    --n_users 59 \
    --n_responses_per_user 20

# All configurations in a directory
python run_simulation.py \
    --config_dir configs \
    --data_file data/reddit/personas.pkl \
    --n_users 492
```

### `agent.py`
Defines the `Agent` class representing individual users.

**Key features:**
- Loads user's historical messages
- Samples style examples for prompting
- Manages personalized model adapters (OPPU)
- Generates responses using provided LLM

### `model.py`
Handles LLM loading, fine-tuning, and text generation.

**Supports:**
- Base models (Llama, Mistral, DeepSeek, etc.)
- Fine-tuned models (LoRA adapters)
- 8-bit/16-bit quantization
- Generation parameter tuning

### `simulate.py`
Core simulation loop with file locking for parallel execution.

**Features:**
- Parallel-safe file locking
- Resume from checkpoints
- Progress tracking
- Automatic caching of results

### `model_utils.py`
Helper functions for BM25 retrieval, conversation formatting, preprocessing.

### `config_utils.py`
YAML/JSON configuration file loading.

## Configuration Options

Configs specify the simulation parameters:

```yaml
model: meta-llama/Llama-3.1-8B  # HuggingFace model name
finetuned: false                 # Use fine-tuned version?
retrieve_context: false          # Use BM25 context retrieval?
n_style_examples: 10             # Number of style examples (0, 10)
OPPU: false                      # Use personalized adapters?
with_persona: true               # Include persona description?
```

## Simulation Pipeline

```
1. Load Data → 2. Load Model → 3. For Each User:
   a. Sample test messages
   b. Build prompt (persona + examples + context)
   c. Generate 20 candidate responses
   d. Save results
```

**Note:** The simulation generates **20 candidate responses** for each user message to enable response selection strategies.

## Output Format

Results are saved as JSON:

```json
[
  {
    "user": "user123",
    "persona": "A tech enthusiast who...",
    "model": "meta-llama/Llama-3.1-8B",
    "fine_tuned": false,
    "retrieve_context": false,
    "n_style_examples": 10,
    "with_persona": true,
    "OPPU": false,
    "reply_to": "What do you think about AI?",
    "original_message": "I think AI is fascinating...",
    "response": "AI is revolutionizing...",
    "all_valid_responses": ["AI is revolutionizing...", ...]
  }
]
```

## Parallelization

The simulation supports running multiple configs in parallel:

```bash
# On different GPUs/machines
python run_simulation.py --config configs/llama3.1_base.yaml --data_file data/reddit/personas.pkl &
python run_simulation.py --config configs/mistral_base.yaml --data_file data/reddit/personas.pkl &
```

File locking prevents conflicts when writing to the same output file.

## Reproducibility

For deterministic results:

```bash
python run_simulation.py \
    --config configs/llama3.1_base.yaml \
    --data_file data/bluesky/personas.pkl \
    --seed 42
```

See `tests/test_seed_utils.py` for seed-fixing implementation.

## Performance & Runtime

### Computational Requirements

**Total generations per dataset:**
- Bluesky: 59 users × 20 responses × 20 candidates = **23,600 generations**
- Twitter/X: 250 users × 20 responses × 20 candidates = **100,000 generations**
- Reddit: 492 users × 20 responses × 20 candidates = **196,800 generations**

### Estimated Runtime (Order of Magnitude)

Approximate runtimes on **NVIDIA A100 GPU**:

| Dataset | Total Generations | Estimated Runtime |
|---------|------------------|-------------------|
| Bluesky | 23,600 | ~13-33 hours |
| Twitter/X | 100,000 | ~55-139 hours (2-6 days) |
| Reddit  | 196,800 | ~109-273 hours (4.5-11 days) |

**⚠️ Important Notes:**
- Runtime varies **significantly** depending on:
  - **Model** (Llama vs Mistral vs DeepSeek vs larger models)
  - **Configuration** (with/without retrieval, style examples, fine-tuning)
  - **GPU** (A100 vs V100 vs RTX 4090, etc.)
  - **Generation parameters** (temperature, max_tokens, etc.)
  - **System load** and other factors
- These are rough estimates only - **benchmark on your hardware** before large runs
- Fine-tuning adds ~30-60 minutes per model (one-time cost)
- OPPU (personalized adapters) adds training time per user

### Hardware Requirements

- **GPU**: CUDA-capable GPU with 24GB+ VRAM recommended for 8B models
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for models and results
- **Python**: 3.8+

## Requirements

Core dependencies:
- PyTorch 2.0+
- transformers 4.30+
- peft 0.4+
- datasets 2.12+

See `requirements.txt` for full dependencies.

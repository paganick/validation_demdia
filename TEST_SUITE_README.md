# Test Suite Documentation

This test suite provides minimal, reproducible tests for the simulation pipeline. It's designed to catch import errors, runtime errors, and verify core functionality without requiring hours of computation.

## 📋 Overview

The test suite consists of:

1. **Quick Test** (`run_test_quick.py`) - Tests only Llama-3.1-8B (3 configs, ~2-5 min)
2. **Full Test** (`run_test_full.py`) - Tests all models (9 configs, ~10-20 min)
3. **Reproducibility Test** (`verify_reproducibility.py`) - Verifies deterministic results
4. **Seed Utils** (`test_seed_utils.py`) - Utilities for fixing random seeds

## 🚀 Quick Start

### Prerequisites

Ensure you have your data file ready (e.g., `bluesky_data/personas.pkl`).

### Run Quick Test (Recommended First)

```bash
# Test only Llama-3.1-8B (fastest)
python run_test_quick.py --data_file bluesky_data/personas.pkl

# Expected runtime: 2-5 minutes
# Expected output: 3 configuration results
```

### Run Full Test

```bash
# Test all models (Llama, DeepSeek, Mistral)
python run_test_full.py --data_file bluesky_data/personas.pkl

# Expected runtime: 10-20 minutes
# Expected output: 9 configuration results
```

### Verify Reproducibility

```bash
# Run same config twice and verify identical outputs
python verify_reproducibility.py --data_file bluesky_data/personas.pkl

# Expected: Byte-for-byte identical outputs
```

## 📁 Test Configurations

All test configs are in `test_configs/` directory:

### Quick Test (Llama-3.1-8B only)
- `llama3.1_base.yaml` - Base model, no examples
- `llama3.1_base_10examples.yaml` - With 10 style examples
- `llama3.1_base_withcontext_10examples.yaml` - With context retrieval + 10 examples

### Full Test (All models)
**Llama-3.1-8B** (3 configs)
- Base, 10 examples, context+10 examples

**DeepSeek-R1-Distill-Llama-8B** (3 configs)
- Base, 10 examples, context+10 examples

**Mistral-7B-v0.1** (3 configs)
- Base, 10 examples, context+10 examples

**Note:** Fine-tuned configurations are excluded from tests (they require pre-trained models).

## 🎲 Reproducibility & Seeds

All tests use **fixed random seed = 42** for deterministic results.

### What's Fixed
- Python's `random` module
- NumPy random operations
- PyTorch random operations (CPU & CUDA)
- Hash-based randomization (PYTHONHASHSEED)

### Running with Custom Seed

```bash
# Quick test with custom seed
python run_test_quick.py --data_file data.pkl --seed 123

# Main simulation with seed
python run_simulation.py --config test_configs/llama3.1_base.yaml \
    --data_file data.pkl --seed 42
```

### Verifying Determinism

The reproducibility test runs the **same configuration twice** and compares:
1. File hashes (byte-level comparison)
2. JSON structure (entry count)
3. Content (field-by-field comparison)

```bash
python verify_reproducibility.py --data_file bluesky_data/personas.pkl
```

If outputs differ, it indicates:
- Non-deterministic model sampling
- Unseeded random operations
- GPU non-determinism
- Race conditions in parallel code

## 🧪 Test Parameters

### Default Test Settings
- **Users:** 1 (vs. 250 in production)
- **Responses per user:** 5 (vs. 20 in production)
- **Total samples:** ~1% of production

### Customizing Test Size

```bash
# Test with more users/responses
python run_test_quick.py --data_file data.pkl \
    --n_users 5 \
    --n_responses 10

# Minimal test (fastest)
python run_test_quick.py --data_file data.pkl \
    --n_users 1 \
    --n_responses 1
```

## 📊 Test Output

### Quick Test Output
```
test_results/
├── meta-llama__Llama-3.1-8B__noft__ctx0__style0__no_OPPU__random_response.json
├── meta-llama__Llama-3.1-8B__noft__ctx0__style10__no_OPPU__random_response.json
├── meta-llama__Llama-3.1-8B__noft__ctx1__style10__no_OPPU__random_response.json
└── quick_test_report.json
```

### Test Report Format
```json
{
  "test_type": "quick",
  "seed": 42,
  "n_users": 1,
  "n_responses": 5,
  "successful": 3,
  "failed": 0,
  "results": [...],
  "failures": []
}
```

## ✅ What the Tests Catch

### Import/Dependency Errors
- Missing packages
- Incorrect import paths
- Module loading failures

### Runtime Errors
- Model loading issues
- Data processing errors
- Memory errors
- GPU/CUDA errors

### Core Functionality
- User sampling works
- Response generation works
- Context retrieval works (when enabled)
- Style examples work (when enabled)
- Output file creation works

### Data Validation
- Correct number of users
- Correct responses per user
- All required fields present
- Valid JSON structure

## 🐛 Debugging with Tests

### If a Test Fails

1. **Check the error message** in test report JSON
2. **Run config manually** for detailed output:
   ```bash
   python run_simulation.py \
       --config test_configs/llama3.1_base.yaml \
       --data_file bluesky_data/personas.pkl \
       --n_users 1 \
       --n_responses_per_user 3 \
       --seed 42
   ```
3. **Check test report** for stack traces:
   ```bash
   cat test_results/quick_test_report.json | jq '.failures'
   ```

### Common Issues

**Import Error: No module named 'transformers'**
- Solution: Install dependencies from `environment.yml`

**CUDA out of memory**
- Solution: Reduce batch size or test on CPU

**Model not found**
- Solution: Check Hugging Face model availability
- Check internet connection for model download

**Data file not found**
- Solution: Verify path to pickle file
- Ensure data file exists

## 🔄 Integration with CI/CD

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Quick Tests
  run: python run_test_quick.py --data_file data/test_data.pkl

- name: Verify Reproducibility
  run: python verify_reproducibility.py --data_file data/test_data.pkl
```

## 📝 Adding New Tests

### Adding a New Configuration

1. Create config in `test_configs/`:
   ```yaml
   model: your-model-name
   finetuned: false
   retrieve_context: false
   n_style_examples: 0
   OPPU: false
   ```

2. Add to test script:
   ```python
   # In run_test_quick.py or run_test_full.py
   test_configs = [
       # ... existing configs
       "your_new_config.yaml",
   ]
   ```

### Testing a New Model

1. Create 3 configs (base, +examples, +context)
2. Add to `run_test_full.py`
3. Run full test suite
4. Verify reproducibility

## 🎯 Best Practices

1. **Run quick test first** - Catches most issues in 2-5 minutes
2. **Fix failing tests before full test** - Saves time
3. **Verify reproducibility** - Ensures debuggability
4. **Use fixed seeds in testing** - Makes issues reproducible
5. **Keep test data small** - 1 user is enough for most issues
6. **Archive test outputs** - Useful for regression testing

## 🔍 Performance Benchmarks

Expected runtimes on NVIDIA A100:

| Test | Configs | Users | Responses | Time |
|------|---------|-------|-----------|------|
| Quick | 3 | 1 | 5 | 2-5 min |
| Full | 9 | 1 | 5 | 10-20 min |
| Reproducibility | 2 runs | 1 | 3 | 3-6 min |

On CPU, expect 2-5x longer.

## 📚 Related Files

- `run_simulation.py` - Main simulation script (now supports `--seed`)
- `src/simulate.py` - Core simulation logic
- `src/agent.py` - Agent implementation
- `src/model.py` - Model loading and fine-tuning

## 🆘 Getting Help

If tests consistently fail:

1. Check the test report JSON for details
2. Run with verbose logging
3. Test individual components (model loading, data loading)
4. Verify GPU/CUDA setup
5. Check disk space (models can be large)

## 📖 Further Reading

- Main README - Full simulation pipeline documentation
- `environment.yml` - Required dependencies
- `configs/` - Production configurations (250 users, 20 responses)

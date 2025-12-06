# Validation of Demographic Diversity in AI-Generated Social Media

Simulation and validation framework for evaluating how well large language models can generate authentic, personalized social media responses.

## 📋 Overview

This project evaluates different personalization techniques for LLM-based social media simulation across three platforms (Bluesky, Twitter/X, Reddit) and multiple models (Llama, Mistral, DeepSeek).

**Key Research Questions:**
- Can LLMs generate responses that match individual writing styles?
- What personalization techniques work best? (personas, examples, retrieval, fine-tuning)
- How well do generated responses pass human/automated validation?

**Paper:** [Computational Turing Test Reveals Systematic Differences Between Human and AI Language](https://arxiv.org/abs/2511.04195)

## 🗂️ Project Structure

```
validation_demdia/
├── data/                   # Datasets (59 Bluesky, 250 Twitter, 492 Reddit users)
├── preprocessing/          # Data preparation scripts
├── simulation/            # Core LLM simulation code
├── configs/               # Simulation configurations
├── validation/            # Analysis and validation tools
├── plotting/              # Visualization scripts
├── tests/                 # Test suite (quick & full tests)
├── scripts/               # Utility scripts
├── results/               # Simulation outputs (gitignored)
├── run_simulation.py      # Main entry point
├── requirements.txt       # Python dependencies
└── environment.yml        # Conda environment
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate demdia_env

# Or using pip
pip install -r requirements.txt
```

### 2. Prepare Data

Place your datasets in the `data/` directory:
```
data/
├── bluesky/personas.pkl   (59 users × 20 messages)
├── twitter/personas.pkl   (250 users × 20 messages)
└── reddit/personas.pkl    (492 users × 20 messages)
```

See `data/README.md` for format details.

### 3. Run Simulation

```bash
# Quick test (1 user, 5 responses, ~5 minutes)
python tests/run_test_quick.py --data_file data/bluesky/personas.pkl

# Full simulation (all users)
python run_simulation.py \
    --config configs/llama3.1_base.yaml \
    --data_file data/bluesky/personas.pkl \
    --n_users 59 \
    --n_responses_per_user 20
```

### 4. Validate Results

```bash
# Run validation pipeline
python validation/validate_text.py \
    --input_dir results/results_bluesky/ \
    --validation all

# Analyze features
python validation/features_analysis.py \
    compute_features results/results_bluesky/
```

## 📊 Datasets

| Platform | Users | Messages | Total |
|----------|-------|----------|-------|
| Bluesky  | 59    | 20/user  | 1,180 |
| Twitter/X| 250   | 20/user  | 5,000 |
| Reddit   | 492   | 20/user  | 9,840 |
| **Total**| **801**| **20/user** | **16,020** |

## 🎯 Simulation Configurations

### Models Tested (9 total)

- **Llama-3.1-8B** (base & instruct)
- **Llama-3.1-70B** (base)
- **Mistral-7B-v0.1** (base & instruct)
- **DeepSeek-R1-Distill-Llama-8B**
- **Apertus-8B-2509**
- **Gemma-3-4B-it**
- **Qwen-2.5-7B-Instruct**

### Personalization Techniques
- ✅ **No personas**
- ✅ **Persona descriptions** - Generated user profiles (Baseline)
- ✅ **Style examples** - Few-shot prompting (10 examples)
- ✅ **Context retrieval** - BM25-based history retrieval
- ✅ **Fine-tuning** - LoRA adapters on user data

### Example Configuration

```yaml
# configs/llama3.1_base_10examples.yaml
model: meta-llama/Llama-3.1-8B
finetuned: false
retrieve_context: false
n_style_examples: 10
OPPU: false
with_persona: true
```

## ⚡ Performance

**Runtime estimates** (NVIDIA A100, per configuration):

| Dataset   | Generations | Estimated Time |
|-----------|------------|----------------|
| Bluesky   | 23,600     | 13-33 hours    |
| Twitter/X | 100,000    | 2-6 days       |
| Reddit    | 196,800    | 4.5-11 days    |

*Note: Varies significantly by model, GPU, and configuration. See `simulation/README.md` for details.*

## 🧪 Testing

Comprehensive test suite for regression testing and validation:

```bash
# Quick test (Llama-3.1-8B only, ~2-5 min)
python tests/run_test_quick.py --data_file data/bluesky/personas.pkl

# Full test (all models, ~10-20 min)
python tests/run_test_full.py --data_file data/bluesky/personas.pkl

# Verify reproducibility
python tests/verify_reproducibility.py --data_file data/bluesky/personas.pkl
```

See `tests/TEST_SUITE_README.md` for detailed testing documentation.

## 📈 Validation & Analysis

### 1. Text Validation
```bash
python validation/validate_text.py \
    --input_dir results/results_bluesky/ \
    --validation all
```

Runs:
- **BERT validation** - Binary classification (human vs. AI)
- **Empath validation** - Linguistic feature analysis

### 2. Feature Analysis
```bash
python validation/features_analysis.py compute_features results/
python validation/features_analysis.py evaluate results/ labels
```

Extracts and analyzes:
- Stylistic features (word count, emojis, links)
- Linguistic complexity
- Sentiment and toxicity
- Feature importance for human/AI classification

### 3. LLM Judging
```bash
python validation/LLM_judge.py results/ --include-advanced
```

Uses LLMs to evaluate response quality and similarity.

## 📊 Visualization

```bash
# Generate plots
python plotting/generate_SOTA_plots.py
python plotting/generate_config_optimal_plots.py
```

Creates:
- Model performance comparisons
- Configuration optimization plots
- Feature importance heatmaps
- Response similarity distributions

## 🔬 Reproducibility

All simulations use fixed random seeds for reproducibility:

```bash
python run_simulation.py \
    --config configs/llama3.1_base.yaml \
    --data_file data/bluesky/personas.pkl \
    --seed 42
```

The test suite includes reproducibility verification to ensure deterministic results.

## 📁 Output Structure

```
results/
├── results_bluesky/
│   ├── meta-llama__Llama-3.1-8B__noft__ctx0__style0__no_OPPU__random_response.json
│   ├── ... (validation data, features, plots)
├── results_twitter/
└── results_reddit/
```

Each JSON file contains:
- Generated responses (20 candidates per message)
- Metadata (model, config, persona)
- Original messages and context

## 🛠️ Development

### Code Organization

- **`simulation/`** - Core simulation engine
- **`validation/`** - Validation and feature extraction
- **`preprocessing/`** - Data preparation
- **`plotting/`** - Visualization
- **`tests/`** - Test suite with golden outputs

### Running Tests Before Commits

```bash
# Run quick test
python tests/run_test_quick.py --data_file data/bluesky/personas.pkl

# Compare with golden outputs
python tests/compare_with_golden.py --test_type quick
```

### Contributing

1. Run tests before committing
2. Update golden outputs if behavior changes intentionally
3. Document any new configurations
4. Follow PEP 8 style guide

## 📚 Documentation

- **`README.md`** (this file) - Project overview
- **`data/README.md`** - Dataset documentation
- **`preprocessing/README.md`** - Data preparation guide
- **`simulation/README.md`** - Simulation details
- **`tests/TEST_SUITE_README.md`** - Testing guide
- **`PROJECT_STRUCTURE.md`** - Directory organization

## 🔧 Requirements

### Hardware
- CUDA-capable GPU (24GB+ VRAM for 8B models)
- 32GB+ RAM recommended
- 100GB+ storage

### Software
- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- See `requirements.txt` for full list

## 📄 License

MIT License - This code is provided for research and educational purposes. You are free to use, modify, and distribute this code, including for commercial purposes, with proper attribution. See the `LICENSE` file for full details.

## 📧 Contact

**Nicolò Pagan**
Email: nicolo.pagan@uzh.ch
University of Zurich

## 🙏 Acknowledgments

- Models: Meta (Llama), Mistral AI, DeepSeek, Apertus, Qwen, Gemma (Google)
- Datasets: Bluesky, Twitter/X, Reddit (public posts only)
- Co-authors: Petter Törnberg, Christopher A. Bail, Anikó Hannák, Christopher Barrie

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{pagan2025computationalturingtestreveals,
      title={Computational Turing Test Reveals Systematic Differences Between Human and AI Language},
      author={Nicolò Pagan and Petter Törnberg and Christopher A. Bail and Anikó Hannák and Christopher Barrie},
      year={2025},
      eprint={2511.04195},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.04195},
}
```

**Paper:** https://arxiv.org/abs/2511.04195

---

**Last Updated:** 2025-01-17

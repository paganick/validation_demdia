# Proposed Project Structure

## New Directory Organization

```
validation_demdia/
├── README.md                          # Main documentation
├── environment.yml                    # Conda environment
├── requirements.txt                   # Pip requirements (to be created)
│
├── data/                              # Data directory (gitignored, with README)
│   ├── README.md                      # Data download instructions
│   ├── bluesky/
│   │   └── personas.pkl              # Bluesky dataset
│   ├── twitter/
│   │   └── personas.pkl              # Twitter dataset
│   └── reddit/
│       └── personas.pkl              # Reddit dataset
│
├── preprocessing/                     # Data preprocessing scripts
│   ├── README.md                      # Preprocessing docs
│   ├── parse_reddit_data.py
│   ├── aggregate_reddit_data.py
│   ├── read_reddidt_data.py
│   └── explore_reddit_data.py
│
├── simulation/                        # Core simulation code
│   ├── __init__.py
│   ├── run_simulation.py             # Main entry point
│   ├── agent.py                       # Agent class
│   ├── model.py                       # Model loading/finetuning
│   ├── simulate.py                    # Simulation logic
│   ├── model_utils.py                 # Helper functions
│   └── config_utils.py                # Config loading
│
├── configs/                           # Simulation configurations
│   ├── README.md                      # Config documentation
│   ├── llama3.1_*.yaml
│   ├── deepseekR1_*.yaml
│   └── mistral_*.yaml
│
├── validation/                        # Validation and analysis
│   ├── utils.py                       # Validator class
│   ├── validate_text.py
│   ├── features_analysis.py
│   ├── feature_utils.py
│   ├── compute_cosine_similarities.py
│   ├── LLM_judge.py
│   ├── build_validation_data.py
│   └── post_process.py
│
├── plotting/                          # Visualization (separate)
│   ├── plotting_utils.py
│   ├── generate_SOTA_plots.py
│   └── generate_config_optimal_plots.py
│
├── tests/                             # Test suite
│   ├── README.md -> ../TEST_SUITE_README.md
│   ├── test_configs/
│   ├── run_test_quick.py
│   ├── run_test_full.py
│   ├── verify_reproducibility.py
│   ├── test_seed_utils.py
│   ├── save_golden_outputs.py
│   └── compare_with_golden.py
│
├── scripts/                           # Utility scripts
│   ├── check_run.py                   # Diagnostic utilities
│   └── convert_emojiis.py
│
└── results/                           # Output directory (gitignored)
    ├── results_bluesky/
    ├── results_twitter/
    └── results_reddit/
```

## Benefits

1. **Clear separation** - Preprocessing vs Simulation vs Validation vs Plotting
2. **Easy navigation** - Each directory has clear purpose
3. **Better imports** - Clean module structure
4. **Data management** - Explicit data directory with READMEs
5. **Easy for newcomers** - Obvious where to start
6. **Testing separate** - Tests don't clutter main code

## Migration Plan

1. Create new directories
2. Move files to appropriate locations
3. Update imports
4. Create __init__.py files
5. Update documentation
6. Test everything still works

# Work Summary - December 10, 2025

## Completed Work

### 1. Fixed Critical Simulation Issues

#### Issue 1: Config Key Bug (context_retrieval → retrieve_context)
- **Problem**: All 45 config files used `context_retrieval:` but the code expected `retrieve_context:`
- **Impact**: Filename collisions caused data loss - configs with different context settings generated identical filenames, so jobs overwrote each other's outputs
- **Fix**: Renamed key in all config files from `context_retrieval` to `retrieve_context`
- **Files Changed**: All 45 config files in `configs/`

#### Issue 2: Gemma Model CUDA Errors
- **Problem**: "CUDA error: device-side assert triggered" during generation
- **Root Cause**: Gemma models require specific tokenizer settings and dtype
- **Fix** (in `simulation/model.py`):
  - Use `torch.bfloat16` dtype (Gemma's preferred format)
  - Set `add_bos_token=True` for tokenizer
  - Don't override Gemma's built-in pad token
- **New Script**: Created `sbatch_scripts/run_gemma_simulations.sh` with proper resource allocation
- **Result**: All 9 Gemma outputs completed successfully

#### Issue 3: Llama-3.1-70B OOM Errors
- **Problem**: Jobs killed at 73% model loading with 32GB RAM
- **Root Cause**: 70B model requires ~140GB for weights in float16
- **Fix**: Created `sbatch_scripts/run_llama70b_simulations.sh`
  - Memory: 256GB (was 32GB)
  - GPUs: 2 (was 1)
  - CPUs: 8 (was 4)
- **Result**: All 9 Llama-70B outputs completed successfully

#### Issue 4: Apertus Model dtype Issues
- **Problem**: dtype mismatch errors when loading Apertus models
- **Fix** (in `simulation/model.py`):
  - Use `torch.float32` dtype for Apertus models
  - Add `trust_remote_code=True` parameter
- **Result**: All 9 Apertus outputs completed successfully

### 2. Simulation Completion Status

All simulations are now complete with **81 total output files**:

- **DeepSeek-R1-Distill-Llama-8B**: 9 files ✓
- **Gemma-3-4B**: 9 files ✓
- **Llama-3.1-8B**: 9 files ✓
- **Llama-3.1-8B-Instruct**: 9 files ✓
- **Llama-3.1-70B**: 9 files ✓
- **Mistral-7B**: 9 files ✓
- **Mistral-7B-Instruct**: 9 files ✓
- **Qwen2.5-7B-Instruct**: 9 files ✓
- **Apertus-8B-2509**: 9 files ✓

Output locations:
- `reference_outputs/deepseek-ai/` - 9 files
- `reference_outputs/google/` - 9 files (Gemma)
- `reference_outputs/meta-llama/` - 27 files (8B, 8B-Instruct, 70B)
- `reference_outputs/mistralai/` - 18 files
- `reference_outputs/Qwen/` - 9 files
- `reference_outputs/swiss-ai/` - 9 files (Apertus)

### 3. Git Commit

Created commit `c39cc25` with all fixes:
- Updated all 45 config files
- Fixed `simulation/model.py` with model-specific handling
- Added 3 new specialized SBATCH scripts
- Commit message documents all issues and fixes

---

## Next Steps (TODO)

### 1. Run Postprocessing Pipeline

Need to run the postprocessing on all 81 simulation outputs:

```bash
# Check what the postprocessing script does
cat sbatch_scripts/run_postprocessing.sh

# Run postprocessing (may need to update script)
sbatch sbatch_scripts/run_postprocessing.sh
```

**Questions to address:**
- What does postprocessing do exactly?
- What outputs does it generate?
- Where should processed results be saved?

### 2. Save Results as Golden Outputs

After postprocessing completes, save current results as "golden outputs" for comparison:

**Proposed approach:**
```bash
# Create golden outputs directory
mkdir -p golden_outputs

# Copy all reference outputs
cp -r reference_outputs/* golden_outputs/

# Copy postprocessed outputs (once generated)
# cp -r postprocessed_outputs/* golden_outputs/
```

**Questions to address:**
- What exactly should be included in golden outputs?
- Just raw simulation outputs or also postprocessed results?
- Should we create a specific commit/tag for golden outputs?

### 3. Re-run Simulations

Run all simulations again with identical configurations:

```bash
# Re-run all simulations
# (Need to clear reference_outputs or use different output directory)
sbatch sbatch_scripts/run_all_simulations.sh
sbatch sbatch_scripts/run_llama70b_simulations.sh
sbatch sbatch_scripts/run_gemma_simulations.sh
sbatch sbatch_scripts/run_apertus_simulations.sh
```

**Questions to address:**
- Should we clear `reference_outputs/` or create a new directory like `reference_outputs_run2/`?
- Do we need to set a different random seed for the second run?

### 4. Compare Results

After second run completes, compare outputs:

**Proposed approach:**
- Create comparison script to:
  - Check if outputs are identical (for reproducibility)
  - Calculate difference metrics if outputs differ
  - Generate comparison report

**Questions to address:**
- What kind of comparison is expected?
  - Exact match (reproducibility test)?
  - Statistical similarity (distribution comparison)?
  - Content similarity (semantic comparison)?
- What format should the comparison report take?

---

## Files Created/Modified

### Modified Files:
- `simulation/model.py` - Added model-specific handling for Gemma, Apertus, and Llama-70B
- All 45 files in `configs/` - Fixed `context_retrieval` → `retrieve_context`

### New Files:
- `sbatch_scripts/run_gemma_simulations.sh` - Gemma-specific SLURM script
- `sbatch_scripts/run_llama70b_simulations.sh` - Llama-70B-specific SLURM script (256GB RAM, 2 GPUs)
- `sbatch_scripts/run_apertus_simulations.sh` - Apertus-specific SLURM script
- `WORK_SUMMARY.md` - This file

---

## Technical Notes

### Resource Requirements by Model:
- **Standard models** (8B): 32GB RAM, 1 GPU, 4 CPUs
- **Llama-70B**: 256GB RAM, 2 GPUs, 8 CPUs (requires high-memory nodes)
- **Gemma**: 64GB RAM, 1 GPU, 4 CPUs
- **Apertus**: Standard allocation

### Special Model Handling in code:
- **Apertus/swiss-ai**: Use `torch.float32`, `trust_remote_code=True`
- **Gemma**: Use `torch.bfloat16`, `add_bos_token=True`
- **Llama-70B**: Standard float16 but needs much more RAM
- **All others**: Standard `torch.float16` loading

### Config File Structure:
Each config specifies:
- `model`: HuggingFace model identifier
- `finetuned`: boolean (whether to use fine-tuned version)
- `retrieve_context`: boolean (whether to include context in generation)
- `n_style_examples`: int (0 or 10, number of style examples)
- `OPPU`: boolean (whether to use OPPU technique)

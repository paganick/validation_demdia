# SLURM Batch Scripts for Running Simulations

This directory contains SLURM sbatch scripts for running simulations and postprocessing on a computing cluster.

## Scripts

### 1. `run_all_simulations.sh`

Runs all non-finetuned model configurations across all three datasets in parallel.

**What it does:**
- Launches 108 parallel jobs (36 configs × 3 datasets)
- Each job runs on a separate GPU node
- Processes 1 user per dataset with 20 test messages
- Uses fixed seed (42) for reproducibility
- Outputs go to `reference_outputs/`

**Usage:**
```bash
# Create logs directory
mkdir -p logs

# Submit job array
sbatch sbatch_scripts/run_all_simulations.sh

# Check job status
squeue -u $USER

# Monitor specific job
tail -f logs/sim_JOBID_TASKID.out
```

**Resource requirements per job:**
- Time: 6 hours
- Memory: 32GB
- CPUs: 4
- GPUs: 1

### 2. `run_postprocessing.sh`

Runs the complete postprocessing pipeline on all simulation outputs.

**What it does:**
- Validates simulation outputs
- Computes linguistic features
- Runs evaluation metrics
- Should be run AFTER simulations complete

**Usage:**
```bash
# Wait for all simulations to complete, then:
sbatch sbatch_scripts/run_postprocessing.sh

# Monitor progress
tail -f logs/postprocessing_JOBID.out
```

**Resource requirements:**
- Time: 4 hours
- Memory: 64GB
- CPUs: 8
- GPUs: None

## Workflow

1. **Run simulations** (in parallel):
   ```bash
   sbatch sbatch_scripts/run_all_simulations.sh
   ```

2. **Wait for completion**:
   ```bash
   # Check if all jobs finished
   squeue -u $USER

   # Count completed outputs (should be 108)
   find reference_outputs/meta-llama -name "*.json" | wc -l
   ```

3. **Run postprocessing**:
   ```bash
   sbatch sbatch_scripts/run_postprocessing.sh
   ```

## Output Structure

```
reference_outputs/
└── meta-llama/
    ├── <model>__<config>__<dataset>__random_response.json
    ├── ... (108 files total)
    └── ...
```

## Monitoring Jobs

```bash
# View all your jobs
squeue -u $USER

# View detailed job info
scontrol show job JOBID

# Cancel all array jobs
scancel ARRAY_JOB_ID

# Cancel specific array task
scancel ARRAY_JOB_ID_TASK_ID
```

## Troubleshooting

**Jobs not starting:**
- Check cluster queue: `squeue`
- Check your limits: `sacctmgr show user $USER`

**Jobs failing:**
- Check error logs: `logs/sim_*_*.err`
- Verify conda environment: `conda list`
- Test locally first: `python3 run_simulation.py --config configs/llama3.1_base.yaml ...`

**Out of memory:**
- Increase `--mem` in sbatch script
- Check actual usage: `sacct -j JOBID --format=JobID,MaxRSS,Elapsed`

## Estimated Runtime

- **Simulations**: ~2-4 hours per job (depends on model size and GPU)
- **Total wall time**: Limited by slowest job
- **Postprocessing**: ~1-2 hours

With sufficient GPUs available, all 108 simulations can complete in 2-4 hours.

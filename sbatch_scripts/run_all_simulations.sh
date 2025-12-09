#!/bin/bash
#SBATCH --job-name=demdia_sim
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-107

# Run simulations for all non-finetuned configs across all datasets
# Total: 36 configs × 3 datasets = 108 jobs (all run in parallel on different nodes)

# Activate conda environment
source ~/.bashrc
conda activate demdia_env

# Navigate to project directory
cd /home/nicpag/data/demdia_val

# Create logs directory
mkdir -p logs

# Get list of non-finetuned configs
mapfile -t CONFIGS < <(ls configs/*.yaml | grep -v finetuned | sort)

# Dataset array
DATASETS=("bluesky" "twitter" "reddit")

# Calculate config and dataset indices
NUM_CONFIGS=${#CONFIGS[@]}
NUM_DATASETS=${#DATASETS[@]}

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))

CONFIG=${CONFIGS[$CONFIG_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

echo "========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "========================================="

# Run simulation with 1 user for testing
python3 run_simulation.py \
    --config "$CONFIG" \
    --data_file "data/${DATASET}/personas.pkl" \
    --n_users 1 \
    --n_responses_per_user 20 \
    --seed 42 \
    --output_dir reference_outputs

echo "Completed simulation for $CONFIG on $DATASET"

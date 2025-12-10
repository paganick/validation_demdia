#!/bin/bash
#SBATCH --job-name=llama70b_sim
#SBATCH --output=logs/llama70b_%A_%a.out
#SBATCH --error=logs/llama70b_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --array=0-5

# Run simulations for Llama-3.1-70B models with high memory allocation
# 70B models require ~140GB for loading model weights in float16
# Using 256GB mem + 2 GPUs to handle the large model
# Running only jobs 0-5 (base_10examples and base_withcontext_10examples configs)

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate demdia_env

# Navigate to project directory
cd /home/nicpag/data/demdia_val

# Create logs directory
mkdir -p logs

# Get list of non-finetuned 70B configs (excluding finetuned ones)
mapfile -t CONFIGS < <(ls configs/llama3.1-70B*.yaml | grep -v finetuned | sort)

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
echo "Memory: 256GB, GPUs: 2"
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

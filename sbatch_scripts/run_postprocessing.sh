#!/bin/bash
#SBATCH --job-name=demdia_post
#SBATCH --output=logs/postprocessing_%j.out
#SBATCH --error=logs/postprocessing_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Run full postprocessing pipeline on reference outputs
# This should run AFTER all simulations complete

# Activate conda environment
source ~/.bashrc
conda activate demdia_env

# Navigate to project directory
cd /home/nicpag/data/demdia_val

# Create logs directory
mkdir -p logs

echo "========================================="
echo "Starting postprocessing pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

# Check if reference outputs exist
if [ ! -d "reference_outputs/meta-llama" ]; then
    echo "ERROR: reference_outputs/meta-llama directory not found!"
    echo "Please run simulations first."
    exit 1
fi

# Count output files
NUM_FILES=$(find reference_outputs/meta-llama -name "*.json" | wc -l)
echo "Found $NUM_FILES simulation output files"

if [ "$NUM_FILES" -eq 0 ]; then
    echo "ERROR: No simulation outputs found!"
    exit 1
fi

# Run postprocessing pipeline
# Note: This will process all results in reference_outputs/
echo "Running validation pipeline..."
python validation/validate_text.py \
    --input_dir reference_outputs/ \
    --validation all

echo "Computing features..."
python validation/features_analysis.py \
    compute_features reference_outputs/

echo "Running evaluation..."
python validation/features_analysis.py \
    evaluate reference_outputs/ labels

echo "========================================="
echo "Postprocessing completed successfully!"
echo "========================================="

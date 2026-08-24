#!/bin/bash
#SBATCH --job-name=cs224n_a2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

#SBATCH --output=/anvil/scratch/x-ezhou1/physics_datasets/CS224_2024Spring/Assignment2/cs224n_a2_%j.out
#SBATCH --error=/anvil/scratch/x-ezhou1/physics_datasets/CS224_2024Spring/Assignment2/cs224n_a2_%j.err

set -euo pipefail

WORKDIR="/anvil/scratch/x-ezhou1/physics_datasets/CS224_2024Spring/Assignment2"
PYTHON="/anvil/scratch/x-ezhou1/physics_datasets/CS224_2024Spring/envs/cs224n_a2/bin/python"

cd "${WORKDIR}"

echo "========================================"
echo "CS224N Assignment 2 full training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: ${PYTHON}"
echo "========================================"

"${PYTHON}" --version
"${PYTHON}" -c "import torch; print('PyTorch:', torch.__version__)"

echo ""
echo "Starting full training..."
echo ""

"${PYTHON}" run.py

echo ""
echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
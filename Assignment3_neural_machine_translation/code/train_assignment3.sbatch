#!/bin/bash

#SBATCH --account=eng260004-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=cs224n_a3
#SBATCH --output=/anvil/scratch/x-ezhou1/physics_datasets/CS224_2024Spring/Assignment3/logs/train_%j.out
#SBATCH --error=/anvil/scratch/x-ezhou1/physics_datasets/CS224_2024Spring/Assignment3/logs/train_%j.err

set -euo pipefail

ASSIGNMENT_ROOT="/anvil/scratch/x-ezhou1/physics_datasets/CS224_2024Spring/Assignment3"
ENV_PREFIX="${ASSIGNMENT_ROOT}/envs/cs224n_a3"

source /home/x-ezhou1/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

cd "${ASSIGNMENT_ROOT}"

export PYTHONUNBUFFERED=1
export NLTK_DATA="/home/x-ezhou1/nltk_data"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "========================================"
echo "CS224N Assignment 3 training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
echo "========================================"

nvidia-smi

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
assert torch.cuda.is_available(), 'GPU is not available inside the Slurm job'
"

echo
echo "Starting full training..."
sh run.sh train

echo
echo "Training completed. Starting test evaluation..."
sh run.sh test

echo
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
#!/bin/bash
#
#SBATCH --partition=testing
#SBATCH --qos=debug
#SBATCH --job-name=real_test
#SBATCH --output=logs/test.txt
#SBATCH --gres=gpu:1
#SBATCH -c8
#
#SBATCH --mail-user=julien.rolland@universite-paris-saclay.fr
#SBATCH --mail-type=ALL

source .venv/bin/activate
srun python src/train.py

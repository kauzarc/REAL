#!/bin/bash
#
#SBATCH --partition=testing
#SBATCH --qos=debug
#SBATCH --job-name=real_test
#SBATCH --output=logs/test_ddp_spawn.txt
#SBATCH --gres=gpu:2
#SBATCH -c8
#SBATCH --mem=128G
#
#SBATCH --mail-user=julien.rolland@universite-paris-saclay.fr
#SBATCH --mail-type=ALL

source .venv/bin/activate
srun python src/train.py train=ddp_spawn

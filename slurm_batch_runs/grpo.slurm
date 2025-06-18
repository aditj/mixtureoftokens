#!/bin/bash
#SBATCH --job-name=gsm_mix_hidden_state
#SBATCH --partition=hpc-low
#SBATCH --gres=gpu:h100:1            # Request 1× H100 GPU
#SBATCH --cpus-per-task=4            # Adjust as needed
#SBATCH --mem=32G                    # Adjust memory as needed
#SBATCH --time=6:00:00               # HH:MM:SS
#SBATCH --array=0-1                 # 16 experiments (see arrays below)
#SBATCH --output=generation_comparison/std/%x_%A_%a.out   # Stdout/stderr per-array-task

# Optional: load modules / activate environment
source venv/bin/activate
cd DeepSeekRL-Extended
python main.py

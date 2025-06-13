#!/bin/bash
#SBATCH --job-name=gsm_mix
#SBATCH --partition=hpc-low
#SBATCH --gres=gpu:h100:1            # Request 1× H100 GPU
#SBATCH --cpus-per-task=4            # Adjust as needed
#SBATCH --mem=32G                    # Adjust memory as needed
#SBATCH --time=1:00:00               # HH:MM:SS
#SBATCH --array=0-8                  # 16 experiments (see arrays below)
#SBATCH --output=generation_comparison/std/%x_%A_%a.out   # Stdout/stderr per-array-task

# Optional: load modules / activate environment
source venv/bin/activate
experiment_name="answer_directly_nucleus_dirichlet"

# ---------------------------------------------------------------------------
# Define the grid of hyper-parameters
# ---------------------------------------------------------------------------

T_E_vals=(100  100  100  200  200  200)
K_vals=( 1  2  4  1  2  4)

N_examples=500
Temperature=0.6

T_E=${T_E_vals[$SLURM_ARRAY_TASK_ID]}
K=${K_vals[$SLURM_ARRAY_TASK_ID]}
T_total=600 #${T_totals[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID with T_e=$T_E, k=$K on GPU $CUDA_VISIBLE_DEVICES"

# Adjust other CLI flags as desired; T_total is kept constant at 600 here
python simple.py \
  --T_e $T_E \
  --k $K \
  --T_total $T_total \
  --num_examples $N_examples \
  --temperature $Temperature \
  --experiment_name $experiment_name

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID" 
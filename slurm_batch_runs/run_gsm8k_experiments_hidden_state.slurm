#!/bin/bash
#SBATCH --job-name=gsm_mix_hidden_state
#SBATCH --partition=hpc-low
#SBATCH --gres=gpu:h100:1            # Request 1× H100 GPU
#SBATCH --cpus-per-task=4            # Adjust as needed
#SBATCH --mem=32G                    # Adjust memory as needed
#SBATCH --time=4:00:00               # HH:MM:SS
#SBATCH --array=0-11                 # 16 experiments (see arrays below)
#SBATCH --output=generation_comparison/std/%x_%A_%a.out   # Stdout/stderr per-array-task

# Optional: load modules / activate environment
source venv/bin/activate
experiment_name="answer_directly_hidden_state_element_wise_max"

# ---------------------------------------------------------------------------
# Define the grid of hyper-parameters
# ---------------------------------------------------------------------------

T_E_vals=(200  200  200  200  200  200  200  200)
K_vals=(1  1  1  1  2  2  2  2)
layer_offset_vals=(1  3  5  7  1  3  5  7)
N_examples=500
Temperature=0.6

T_E=${T_E_vals[$SLURM_ARRAY_TASK_ID]}
K=${K_vals[$SLURM_ARRAY_TASK_ID]}
layer_offset=${layer_offset_vals[$SLURM_ARRAY_TASK_ID]}
T_total=600 #${T_totals[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID with T_e=$T_E, k=$K on GPU $CUDA_VISIBLE_DEVICES"

# Adjust other CLI flags as desired; T_total is kept constant at 600 here
python simple_hidden_state.py \
  --T_e $T_E \
  --k $K \
  --T_total $T_total \
  --num_examples $N_examples \
  --temperature $Temperature \
  --experiment_name $experiment_name \
  --layer_offset $layer_offset

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID" 
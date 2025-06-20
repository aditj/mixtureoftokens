#!/bin/bash
#SBATCH --job-name=gsm_mix_hidden_state
#SBATCH --partition=hpc-low
#SBATCH --gres=gpu:h100:1            # Request 1× H100 GPU
#SBATCH --cpus-per-task=4            # Adjust as needed
#SBATCH --mem=32G                    # Adjust memory as needed
#SBATCH --time=12:00:00               # HH:MM:SS
#SBATCH --array=0-8                 # 16 experiments (see arrays below)
#SBATCH --output=generation_comparison/std/%x_%A_%a.out   # Stdout/stderr per-array-task

# Optional: load modules / activate environment
source venv/bin/activate

T_E_vals=(400 400 200 200 400 400 200 200)
K_vals=(2 4 2 4 8 16 8 16)
T_totals=(1000 800 800 800 1200 1000 1000 1000)

Temperature=0.6

T_E=${T_E_vals[$SLURM_ARRAY_TASK_ID]}
K=${K_vals[$SLURM_ARRAY_TASK_ID]}
T_total=1000 #${T_totals[$SLURM_ARRAY_TASK_ID]}
experiment_name="uniform"
echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID with T_e=$T_E, k=$K on GPU $CUDA_VISIBLE_DEVICES"

# Adjust other CLI flags as desired; T_total is kept constant at 600 here
python mixture_grpo.py \
  --T_e $T_E \
  --k $K \
  --slurm_id $SLURM_ARRAY_TASK_ID \
  --experiment_name $experiment_name

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID" 
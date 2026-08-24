#!/bin/bash
#SBATCH -J hecktor_stragglers
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/stragglers_%j.log

# Finish the ~14 leftover radiomics patients on a DEDICATED compute node (no login-node
# contention, which is what caused the 600s timeouts). Each patient is still hard-capped
# by the per-patient timeout in step 13, so nothing can hang. Resumes from the jsonl.

set -euo pipefail
module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/radiomics

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs outputs/eda

python -u scripts/13_finish_stragglers.py \
  --pred-root /N/scratch/$USER/hecktor_nnunet/results/Dataset501_HECKTOR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
  --data-root "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data" \
  --out outputs/eda/radiomics_pred.csv \
  --timeout 300

echo "STRAGGLERS DONE"

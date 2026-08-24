#!/bin/bash
#SBATCH -J hecktor_radiomics
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -o logs/radiomics_%j.log

# NOTE: previous run hit BOTH the 8h wall clock AND out-of-memory kills (8 workers
# on large CT/PET volumes exceeded 64G). Fixes:
#   - mem raised 64G -> 120G, workers reduced 8 -> 4  (more RAM per worker)
#   - time raised 8h -> 24h
#   - script now writes each patient to .jsonl IMMEDIATELY, so a kill loses nothing
#     and simply re-running this job RESUMES from where it stopped.

set -euo pipefail
module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/radiomics

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs outputs/eda

# grep -v filters the harmless repeated GLCM notice that spammed the last log
python -u scripts/11_extract_radiomics.py \
  --pred-root /N/scratch/$USER/hecktor_nnunet/results/Dataset501_HECKTOR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
  --data-root "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data" \
  --out outputs/eda/radiomics_pred.csv \
  --workers 4 \
  2>&1 | grep -v "GLCM is symmetrical"

echo "RADIOMICS DONE"

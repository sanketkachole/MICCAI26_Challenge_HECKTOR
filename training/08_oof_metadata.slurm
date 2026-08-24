#!/bin/bash
#SBATCH -J hecktor_oof_meta
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/oof_meta_%j.log

set -euo pipefail

module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs outputs/eda

python scripts/07_oof_pred_metadata.py \
  --pred-root /N/scratch/$USER/hecktor_nnunet/results/Dataset501_HECKTOR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
  --data-root "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data" \
  --out outputs/eda/case_metadata_pred.csv

echo "OOF METADATA DONE"

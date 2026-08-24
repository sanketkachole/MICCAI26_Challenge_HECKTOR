#!/bin/bash
#SBATCH -J hecktor_serr
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/serr_%j.log
set -euo pipefail
module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR
cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs
python -u scripts/37_survival_errors.py \
  --clinical "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data/HECKTOR_2026_training_data.csv" \
  --meta outputs/eda/case_metadata_pred.csv \
  --radiomics outputs/eda/radiomics_pred.csv
echo "SERR DONE"

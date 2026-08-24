#!/bin/bash
#SBATCH -J hecktor_prog_rad
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/prognosis_rad_%j.log

set -euo pipefail
module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs

python -u scripts/16_prognosis_radiomics.py \
  --clinical "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data/HECKTOR_2026_training_data.csv" \
  --meta outputs/eda/case_metadata_pred.csv \
  --radiomics outputs/eda/radiomics_pred.csv \
  --splits outputs/eda/splits_2026.json

echo "PROGNOSIS RADIOMICS DONE"

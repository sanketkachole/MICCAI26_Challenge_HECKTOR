#!/bin/bash
#SBATCH -J sk_cc
#SBATCH -A r01753
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o logs/cc_%j.log
set -euo pipefail
module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR
cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs outputs/eda
python -u scripts/51_crop_check.py --stage all \
  --data-root "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data" \
  --extent outputs/eda/disease_extent.csv \
  --work /N/scratch/$USER/cropcheck \
  --results /N/scratch/$USER/hecktor_nnunet/results \
  --n-large 8 --n-typical 8
echo "CC DONE"

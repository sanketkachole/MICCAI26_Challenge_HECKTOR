#!/bin/bash
#SBATCH -J hecktor_tune_nodes
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/tune_nodes_%j.log

set -euo pipefail
module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs outputs/eda

# -u = unbuffered so progress shows in the log immediately
python -u scripts/09_tune_nodes.py \
  --pred-root /N/scratch/$USER/hecktor_nnunet/results/Dataset501_HECKTOR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
  --data-root "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data" \
  --clinical "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data/HECKTOR_2026_training_data.csv"

echo "TUNE NODES DONE"

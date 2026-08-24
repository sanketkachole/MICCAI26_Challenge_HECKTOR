#!/bin/bash
#SBATCH -J hecktor_smoke_prep
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/smoke_prep_%j.log

set -euo pipefail

module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs

export nnUNet_raw=/N/scratch/$USER/hecktor_nnunet_smoke/raw
export nnUNet_preprocessed=/N/scratch/$USER/hecktor_nnunet_smoke/preprocessed
export nnUNet_results=/N/scratch/$USER/hecktor_nnunet_smoke/results

rm -rf /N/scratch/$USER/hecktor_nnunet_smoke
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

python scripts/01_prepare_nnunet.py \
  --data-root "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data" \
  --nnunet-raw "$nnUNet_raw" \
  --dataset-id 599 \
  --limit 2 \
  --smoke-split \
  --link

nnUNetv2_plan_and_preprocess \
  -d 599 \
  --verify_dataset_integrity \
  -c 3d_fullres

cp "$nnUNet_raw/Dataset599_HECKTOR/splits_final.json" \
   "$nnUNet_preprocessed/Dataset599_HECKTOR/splits_final.json"

echo "SMOKE PREPROCESS DONE"

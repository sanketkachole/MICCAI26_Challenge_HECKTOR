#!/bin/bash
#SBATCH -J hecktor_prep_full
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/prep_full_%j.log

set -euo pipefail

module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs outputs/eda

export nnUNet_raw=/N/scratch/$USER/hecktor_nnunet/raw
export nnUNet_preprocessed=/N/scratch/$USER/hecktor_nnunet/preprocessed
export nnUNet_results=/N/scratch/$USER/hecktor_nnunet/results

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

python scripts/01_prepare_nnunet.py \
  --data-root "/N/project/Sanket_Slate_Project/1_datasets/MICCAI_HECKTOR/HECKTOR 2026 Training Data" \
  --splits "outputs/eda/splits_2026.json" \
  --nnunet-raw "$nnUNet_raw" \
  --dataset-id 501 \
  --link

nnUNetv2_plan_and_preprocess \
  -d 501 \
  --verify_dataset_integrity \
  -c 3d_fullres

cp "$nnUNet_raw/Dataset501_HECKTOR/splits_final.json" \
   "$nnUNet_preprocessed/Dataset501_HECKTOR/splits_final.json"

echo "FULL PREPROCESS DONE"

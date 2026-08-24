#!/bin/bash
#SBATCH -J hecktor_bestcfg
#SBATCH -A r01753
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/bestcfg_%j.log

set -euo pipefail

module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs

export nnUNet_raw=/N/scratch/$USER/hecktor_nnunet/raw
export nnUNet_preprocessed=/N/scratch/$USER/hecktor_nnunet/preprocessed
export nnUNet_results=/N/scratch/$USER/hecktor_nnunet/results

nnUNetv2_find_best_configuration 501 -c 3d_fullres -tr nnUNetTrainer_250epochs
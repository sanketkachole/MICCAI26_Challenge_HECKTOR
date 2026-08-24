#!/bin/bash
#SBATCH -J hecktor_smoke_train
#SBATCH -A r01753
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/smoke_train_%j.log

set -euo pipefail

module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs

export nnUNet_raw=/N/scratch/$USER/hecktor_nnunet_smoke/raw
export nnUNet_preprocessed=/N/scratch/$USER/hecktor_nnunet_smoke/preprocessed
export nnUNet_results=/N/scratch/$USER/hecktor_nnunet_smoke/results

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

nnUNetv2_train 599 3d_fullres 0 -tr nnUNetTrainer_1epoch --npz

echo "SMOKE TRAIN DONE"

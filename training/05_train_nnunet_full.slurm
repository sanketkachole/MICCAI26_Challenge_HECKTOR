#!/bin/bash
#SBATCH -J hecktor_train
#SBATCH -A r01753
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH -o logs/train_f%a_%j.log
#SBATCH --array=0-4

set -euo pipefail

module load conda/26.3.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/envs/HECKTOR

cd /N/project/Sanket_Slate_Project/4_modelling/MICCAI_HECKTOR/HECKTOR2026
mkdir -p logs

export nnUNet_raw=/N/scratch/$USER/hecktor_nnunet/raw
export nnUNet_preprocessed=/N/scratch/$USER/hecktor_nnunet/preprocessed
export nnUNet_results=/N/scratch/$USER/hecktor_nnunet/results

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

nnUNetv2_train 501 3d_fullres "$SLURM_ARRAY_TASK_ID" -tr nnUNetTrainer_250epochs --npz

echo "TRAINING DONE fold=$SLURM_ARRAY_TASK_ID"

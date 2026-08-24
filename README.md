\# GASP — Multimodal Geometry-Aware Segmentation, Staging, and Prognostication of Head and Neck Cancer



Geometry-aware framework for joint tumor and nodal segmentation, TN staging, and recurrence-free survival prediction from head and neck PET/CT. Submitted to the \*\*MICCAI 2026 HECKTOR challenge\*\*.



Sanket Kachole and Spyridon Bakas — Division of Computational Pathology, Indiana University School of Medicine. Corresponding author: spbakas@iu.edu



\- Challenge: https://hecktor26.grand-challenge.org/

\- Model weights (Zenodo): https://doi.org/10.5281/zenodo.22073791



\## Method in brief



An nnU-Net 3D full-resolution ensemble (5 folds, PlainConvUNet, patch 96×160×160) segments the primary tumor (GTVp) and nodal disease (GTVn) from co-registered CT and FDG-PET. A cleanup stage merges nearby nodal components and discards candidates below fixed gates on volume (1.5 mL), peak SUV (3.0), and largest diameter (8.0 mm). A geometry stage converts the cleaned mask into interpretable measurements — GTVp volume, node count, largest nodal diameter, total nodal volume, and laterality — which drive three lightweight heads: a size-focused logistic classifier for N stage, a label-audited logistic classifier for T stage, and a Cox proportional hazards model for recurrence-free survival. All downstream models are fitted on out-of-fold segmentation predictions, never on expert masks.



Large, near-whole-body scans are cropped to a 400 mm head-and-neck slab before segmentation and the prediction is pasted back onto the original grid, which keeps peak memory within the challenge runtime; ordinary scans bypass the crop and run unchanged.



\*\*Official test set (400 patients, three withheld centers):\*\* weighted score \*\*0.5654\*\* — mean Dice \*\*0.652\*\*, RFS C-index \*\*0.558\*\*, T balanced accuracy \*\*0.461\*\*, N balanced accuracy \*\*0.564\*\*.



\## 1. Setup



```bash

conda create -n hecktor python=3.11 \&\& conda activate hecktor

pip install -r requirements.txt

```



Segmentation training and the tabular experiments ran on an HPC cluster (Slurm; GPU partition for nnU-Net, CPU partitions for the tabular work). The container was built and verified locally on a single NVIDIA RTX 6000 Ada (48 GB). The submitted container targets the Grand Challenge runtime: 1× NVIDIA T4 (16 GB VRAM), 8 vCPU, 32 GB RAM, 25 minutes per case, no network access.



Two extra environments are needed only to reproduce the survival ablations:



```bash

\# pyradiomics feature extraction

conda create -n radiomics python=3.10 \&\& pip install pyradiomics



\# ICARE comparison (pinned sklearn/sksurv pair)

conda create -n icare python=3.10

pip install git+https://github.com/Lrebaud/ICARE.git

pip install "scikit-learn==1.3.2" "scikit-survival==0.22.2" "numpy<2"

```



\## 2. Usage



\### Path A — Inference from released weights (no training)



1\. Download the weight bundle from Zenodo (https://doi.org/10.5281/zenodo.22073791) and extract it so that `model/nnunet\_results/` and the `.joblib` files sit beside `model/staging\_config.json`.

2\. Build and package the container:



```bash

bash scripts/build.sh          # docker build

bash scripts/save.sh           # docker save + tar the model folder

bash scripts/test\_run.sh       # run on a case placed under test/input

```



`inference.py` reads CT, PET, and `ehr.json` from `/input`, runs the nnU-Net ensemble → nodal cleanup → geometry stage → three heads, and writes `output.mha`, `t-stage.json`, `n-stage.json`, and `rfs.json` to `/output`.



\### Path B — Train from scratch



Scripts in `training/` reproduce the full pipeline. Run in numeric order; paths are argparse defaults to override per environment. Slurm submitters (`.slurm`) accompany the heavier steps.


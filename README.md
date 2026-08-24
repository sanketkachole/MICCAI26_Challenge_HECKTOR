# GASP - Multimodal Geometry-Aware Segmentation, Staging, and Prognostication of Head and Neck Cancer

Geometry-aware framework for joint tumor and nodal segmentation, TN staging, and recurrence-free survival prediction from head and neck PET/CT. Submitted to the **MICCAI 2026 HECKTOR challenge**.

Sanket Kachole and Spyridon Bakas - Division of Computational Pathology, Indiana University School of Medicine. Corresponding author: spbakas@iu.edu

- Challenge: https://hecktor26.grand-challenge.org/
- Model weights (Zenodo): https://doi.org/10.5281/zenodo.22073791

## Method in brief

An nnU-Net 3D full-resolution ensemble (5 folds, PlainConvUNet, patch 96x160x160) segments the primary tumor (GTVp) and nodal disease (GTVn) from co-registered CT and FDG-PET. A cleanup stage merges nearby nodal components and discards candidates below fixed gates on volume (1.5 mL), peak SUV (3.0), and largest diameter (8.0 mm). A geometry stage converts the cleaned mask into interpretable measurements - GTVp volume, node count, largest nodal diameter, total nodal volume, and laterality - which drive three lightweight heads: a size-focused logistic classifier for N stage, a label-audited logistic classifier for T stage, and a Cox proportional hazards model for recurrence-free survival. All downstream models are fitted on out-of-fold segmentation predictions, never on expert masks.

Large, near-whole-body scans are cropped to a 400 mm head-and-neck slab before segmentation and the prediction is pasted back onto the original grid, which keeps peak memory within the challenge runtime; ordinary scans bypass the crop and run unchanged.

**Official test set (400 patients, three withheld centers):** weighted score **0.5654** - mean Dice **0.652**, RFS C-index **0.558**, T balanced accuracy **0.461**, N balanced accuracy **0.564**.

## 1. Setup

```bash
conda create -n hecktor python=3.11 && conda activate hecktor
pip install -r requirements.txt
```



Two extra environments are needed only to reproduce the survival ablations:

```bash
# pyradiomics feature extraction
conda create -n radiomics python=3.10 && pip install pyradiomics

# ICARE comparison (pinned sklearn/sksurv pair)
conda create -n icare python=3.10
pip install git+https://github.com/Lrebaud/ICARE.git
pip install "scikit-learn==1.3.2" "scikit-survival==0.22.2" "numpy<2"
```

## 2. Usage

### Path A - Inference from released weights (no training)

1. Download the weight bundle from Zenodo (https://doi.org/10.5281/zenodo.22073791) and extract it so that `model/nnunet_results/` and the `.joblib` files sit beside `model/staging_config.json`.
2. Build and package the container:

```bash
bash scripts/build.sh          # docker build
bash scripts/save.sh           # docker save + tar the model folder
bash scripts/test_run.sh       # run on a case placed under test/input
```

`inference.py` reads CT, PET, and `ehr.json` from `/input`, runs the nnU-Net ensemble, nodal cleanup, geometry stage, and the three heads, and writes `output.mha`, `t-stage.json`, `n-stage.json`, and `rfs.json` to `/output`.

### Path B - Train from scratch

Scripts in `training/` reproduce the full pipeline. Run in numeric order; paths are argparse defaults to override per environment. Slurm submitters (`.slurm`) accompany the heavier steps.

```
01_prepare_nnunet.py            convert to nnU-Net dataset format
05_train_nnunet_full.slurm      5-fold 3d_fullres, 250 epochs
07_oof_pred_metadata.py         geometry measurements from out-of-fold masks
09_tune_nodes.py                nodal cleanup threshold sweep
35_train_final_staging.py       final N and T classifiers (with CV check)
04_prognosis.py                 Cox survival model
50_measure_disease_extent.py    measures the 400 mm crop margin
51_crop_check.py                validates the crop on large-FOV cases
```

Then apply the staging-model changes to the container brain:

```bash
python patch_inference.py
```

Radiomics extraction (`11_extract_radiomics.py`, `13_finish_stragglers.py`) is needed only to reproduce the negative ablations in the paper; the submitted system does not use radiomics.

## 3. Repository layout

```
inference.py            container entry point (crop, segment, geometry, heads)
patch_inference.py      applies the staging-model changes to inference.py
Dockerfile              container definition
requirements.txt        python dependencies
scripts/                build.sh, save.sh, test_run.sh
model/                  staging_config.json + README (weights via Zenodo)
training/               full reproduction pipeline (.py and .slurm)
test/                   README + ehr.example.json (no patient data)
```

### Training scripts

Core pipeline: `01_prepare_nnunet` (dataset prep), `03_staging` / `03b_staging_pred` (early staging on GT vs predicted masks), `07_oof_pred_metadata` (geometry from out-of-fold masks), `09_tune_nodes` (cleanup sweep), `35_train_final_staging` (final N and T models), `04_prognosis` (Cox model), plus `50_measure_disease_extent`, `51_crop_check`, and `crop_utils` for the memory-safe crop.

Ablations and recorded negative results (kept for transparency, not needed for the main result): `15_train_T_radiomics`, `16_prognosis_radiomics`, `22_icare_vs_cox`, `23_hpv_nrule`, `24_error_analysis`, `25_n_ordinal`, `27_t_cleanup`, `29_survival_ensemble`, `31_container_ensemble`, `33_survival_stability`, `37_survival_errors`, `39_dissemination`, `41_eval_dissemination`.

## 4. Model weights

DOI: https://doi.org/10.5281/zenodo.22073791

Bundle contents: `nnunet_results/` (5 fold checkpoints for `Dataset501_HECKTOR`), `n_classifier.joblib`, `t_classifier.joblib`, `staging_Tmodel.joblib`, `prognosis_model.joblib`, and `staging_config.json`. Extract into `model/`.

The HECKTOR imaging data is not redistributed here and must be obtained from the challenge organisers under their data use agreement.

## Caveats

- Nodal cleanup thresholds were tuned on the training cohort only and applied unchanged to every center, with no per-center adaptation.
- T staging encodes anatomical invasion that tumor volume and nodal burden do not capture, so accuracy on this task remains limited.
- The prognostic signal available from segmentation geometry and routine clinical variables appears close to saturation: seven alternative survival approaches did not beat a simple Cox model under patient-weighted validation.
- This is a challenge submission and is not intended for unsupervised clinical use.

## Citation

Kachole, S., Bakas, S. GASP: Multimodal Geometry-Aware Segmentation, Staging, and Prognostication of Head and Neck Cancer. MICCAI 2026 HECKTOR Challenge.

## Acknowledgements

Container scaffolding is derived from the organisers' starter kit: https://github.com/BioMedIA-MBZUAI/HECKTOR2026/

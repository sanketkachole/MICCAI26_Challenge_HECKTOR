# Model weights

The trained weights are distributed through Zenodo, not this repository, because of their size.

**Zenodo DOI:** https://doi.org/10.5281/zenodo.22073791

## Contents of the Zenodo bundle

| File | Description |
|---|---|
| `nnunet_results/` | 5-fold nnU-Net checkpoints for `Dataset501_HECKTOR` (3d_fullres) |
| `n_classifier.joblib` | Size-focused logistic model for N stage |
| `t_classifier.joblib` | Label-audited logistic model for T stage |
| `staging_Tmodel.joblib` | Earlier T-staging model (kept for compatibility) |
| `prognosis_model.joblib` | Cox proportional hazards survival model |
| `staging_config.json` | Configuration read by the inference container |

## How to use

Download the bundle from the Zenodo link above and extract its contents into this `model/` folder so that `nnunet_results/` and the `.joblib` files sit directly beside `staging_config.json`. The container reads everything from here at `/opt/ml/model/` during inference.

The HECKTOR imaging data is **not** included and must be obtained from the challenge organisers under their data use agreement.

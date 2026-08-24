#!/usr/bin/env python3
r"""
HECKTOR 2026 - inference pipeline (runs INSIDE the container).
Given one patient's CT + PET (+ clinical row), it produces the 4 outputs:
  <out>/images/head-neck-tumor-segmentation/<PID>.mha   (label 0/1/2, original CT grid)
  <out>/t-stage.json   "T2"
  <out>/n-stage.json   "N1"
  <out>/rfs.json        42.0   (risk score, higher = worse)

Pipeline: nnU-Net 5-fold ensemble -> node clean-up -> geometry -> N-rule + T-model
          -> prognosis Cox risk.

NOTE ON I/O: the exact Grand Challenge input/output paths + how the clinical data
arrives come from the challenge's docker-template branch. The functions here are
self-contained; wire the GC paths in main() once we confirm the template.
"""
import argparse, json, os, re, subprocess, sys, glob
from pathlib import Path
import numpy as np

try:
    import SimpleITK as sitk
except Exception:
    sitk = None
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

MIDLINE_DEADZONE_MM = 6.0
N_ORDINAL = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}


# ============================================================ PURE HELPERS ===
def cleanup_nodes(label_arr, pet_arr, spacing_zyx, suv_thr=2.5, diam_mm=10.0, min_vox=5):
    """Remove false-positive node blobs; keep only the largest GTVp component."""
    sz, sy, sx = spacing_zyx
    out = label_arr.copy()
    # --- GTVn (label 2): keep a component only if SUV>=thr OR diameter>=diam_mm ---
    gtvn = (label_arr == 2)
    if gtvn.any():
        lab, n = ndi.label(gtvn, structure=np.ones((3, 3, 3), int))
        for c in range(1, n + 1):
            comp = lab == c
            idx = np.argwhere(comp); nvox = idx.shape[0]
            extent = (idx.max(0) - idx.min(0) + 1) * np.array([sz, sy, sx])
            maxdim = float(extent.max())
            suvmax = float(pet_arr[comp].max()) if pet_arr is not None else 999.0
            keep = (suvmax >= suv_thr) or (maxdim >= diam_mm)
            if nvox < min_vox and maxdim < diam_mm and suvmax < suv_thr:
                keep = False
            if not keep:
                out[comp] = 0
    # --- GTVp (label 1): keep the single largest component ---
    gtvp = (out == 1)
    if gtvp.any():
        labp, npn = ndi.label(gtvp, structure=np.ones((3, 3, 3), int))
        if npn > 1:
            sizes = ndi.sum(np.ones_like(labp), labp, index=range(1, npn + 1))
            big = int(np.argmax(sizes)) + 1
            for k in range(1, npn + 1):
                if k != big:
                    out[labp == k] = 0
    return out


def analyze_gtvn(label_arr, spacing_zyx, phys_x_of_index, midline_x):
    sz, sy, sx = spacing_zyx
    voxvol_ml = (sz * sy * sx) / 1000.0
    gtvn = (label_arr == 2)
    if not gtvn.any():
        return dict(n_nodes=0, max_node_mm=0.0, total_vol_ml=0.0, laterality="none")
    lab, n = ndi.label(gtvn, structure=np.ones((3, 3, 3), int))
    max_mm, tot_ml, sides = 0.0, 0.0, set()
    for c in range(1, n + 1):
        idx = np.argwhere(lab == c)
        extent = idx.max(0) - idx.min(0) + 1
        max_mm = max(max_mm, float((extent * np.array([sz, sy, sx])).max()))
        tot_ml += idx.shape[0] * voxvol_ml
        cen = idx.mean(0); px = phys_x_of_index(cen[0], cen[1], cen[2])
        if px > midline_x + MIDLINE_DEADZONE_MM: sides.add("R")
        elif px < midline_x - MIDLINE_DEADZONE_MM: sides.add("L")
    lat = "bilateral" if len(sides) >= 2 else ("unilateral" if len(sides) == 1 else "midline")
    return dict(n_nodes=int(n), max_node_mm=round(max_mm, 1),
                total_vol_ml=round(tot_ml, 3), laterality=lat)


def rule_N(n_nodes, max_mm, laterality, n3_mm, n1_mm):
    if n_nodes == 0: return "N0"
    if max_mm > n3_mm: return "N3"
    if n_nodes == 1 and laterality in ("unilateral", "midline") and max_mm <= n1_mm: return "N1"
    return "N2"


def apply_encoders(row, cols, enc):
    """Single-row feature vector using saved staging encoders (train/inference consistent)."""
    vals = []
    for c in cols:
        v = row.get(c, np.nan)
        if c in enc:
            cats = enc[c]
            code = cats.index(str(v)) if str(v) in cats else np.nan
            vals.append(float(code) if code == code else np.nan)
        else:
            try: vals.append(float(v))
            except (TypeError, ValueError): vals.append(np.nan)
    return np.array(vals, float).reshape(1, -1)


def prog_features(row, cmap, spec):
    """Build the prognosis raw feature row (mirrors step 04 build_raw_features)."""
    X = {}
    for g in spec["GEOM"]:
        X[g] = float(row.get(g, np.nan)) if row.get(g, None) is not None else np.nan
    rn = str(row.get("rule_N", "")).upper()[:2]
    X["N_ordinal"] = spec["N_ORDINAL"].get(rn, np.nan)
    X["Age"] = float(row.get(cmap.get("Age"), np.nan)) if row.get(cmap.get("Age")) is not None else np.nan
    for c in spec["CAT_CLIN"]:
        src = cmap.get(c)
        X[c] = row.get(src, np.nan)
    import pandas as pd
    return pd.DataFrame([X])


def apply_prep(state, X):
    import pandas as pd
    Z = pd.DataFrame(index=X.index)
    for c in [c for c in state["feat_order"] if not c.endswith("_missing")]:
        s = X[c] if c in X else pd.Series(np.nan, index=X.index)
        if c in state["cat_maps"]:
            cat = pd.Categorical(s.astype(str), categories=state["cat_maps"][c])
            s = pd.Series(cat.codes, index=X.index).replace({-1: np.nan})
        else:
            s = pd.to_numeric(s, errors="coerce")
        if c + "_missing" in state["feat_order"]:
            Z[c + "_missing"] = s.isna().astype(float)
        s = s.fillna(state["medians"][c])
        Z[c] = (s - state["mu"][c]) / state["sd"][c]
    return Z[state["feat_order"]]


def predict_risk_from_state(state, X):
    return float(state["cph"].predict_partial_hazard(apply_prep(state, X)).values[0])


# ============================================================ ITK / RUNTIME ==
def load_image_any(path):
    return sitk.ReadImage(str(path))


def run_nnunet(ct_img, pet_img, pid, workdir, results_dir):
    """Write CT/PET as nnU-Net input, run 5-fold ensemble predict, return label sitk image."""
    ind = Path(workdir) / "nnin"; outd = Path(workdir) / "nnout"
    ind.mkdir(parents=True, exist_ok=True); outd.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct_img,  str(ind / f"{pid}_0000.nii.gz"))
    sitk.WriteImage(pet_img, str(ind / f"{pid}_0001.nii.gz"))
    env = dict(os.environ, nnUNet_results=str(results_dir),
               nnUNet_raw=str(Path(workdir) / "raw"),
               nnUNet_preprocessed=str(Path(workdir) / "prep"))
    cmd = ["nnUNetv2_predict", "-i", str(ind), "-o", str(outd),
           "-d", "Dataset501_HECKTOR", "-c", "3d_fullres",
           "-f", "0", "1", "2", "3", "4",
           "-tr", "nnUNetTrainer_250epochs", "-p", "nnUNetPlans"]
    subprocess.run(cmd, check=True, env=env)
    return sitk.ReadImage(str(outd / f"{pid}.nii.gz"))


def process_patient(ct_img, pet_img, pid, clinical_row, models_dir, results_dir, workdir):
    staging_cfg = json.load(open(Path(models_dir) / "staging_config.json"))
    import joblib
    Tclf = joblib.load(Path(models_dir) / "staging_Tmodel.joblib")
    prog = joblib.load(Path(models_dir) / "prognosis_model.joblib")

    # 1) segmentation ensemble
    pred = run_nnunet(ct_img, pet_img, pid, workdir, results_dir)
    larr = sitk.GetArrayFromImage(pred)
    parr = sitk.GetArrayFromImage(pet_img)
    sp = pred.GetSpacing(); spacing_zyx = (sp[2], sp[1], sp[0])

    # 2) node clean-up (remove FP blobs)
    if parr.shape != larr.shape:  # safety: PET must align to label grid
        parr = sitk.GetArrayFromImage(sitk.Resample(pet_img, pred, sitk.Transform(),
                                                    sitk.sitkLinear, 0.0, pet_img.GetPixelID()))
    cleaned = cleanup_nodes(larr, parr, spacing_zyx)
    out_img = sitk.GetImageFromArray(cleaned.astype(np.uint8)); out_img.CopyInformation(pred)

    # 3) geometry
    size = pred.GetSize()
    def phys_x(iz, iy, ix):
        return pred.TransformContinuousIndexToPhysicalPoint((float(ix), float(iy), float(iz)))[0]
    midline = pred.TransformContinuousIndexToPhysicalPoint((size[0]/2., size[1]/2., size[2]/2.))[0]
    g = analyze_gtvn(cleaned, spacing_zyx, phys_x, midline)
    gtvp_ml = float((cleaned == 1).sum()) * (sp[0]*sp[1]*sp[2]) / 1000.0

    feat_row = {"gtvp_ml": gtvp_ml, "n_gtvn": g["n_nodes"], "max_node_mm": g["max_node_mm"],
                "gtvn_total_ml": g["total_vol_ml"], "laterality": g["laterality"]}
    feat_row.update(clinical_row or {})

    # 4) N (rule) + T (model)
    nr = staging_cfg["N_rule"]
    n_stage = rule_N(g["n_nodes"], g["max_node_mm"], g["laterality"], nr["N3_MM"], nr["N1_MAX_MM"])
    Xt = apply_encoders(feat_row, staging_cfg["T_features"], staging_cfg["T_encoders"])
    t_stage = str(Tclf.predict(Xt)[0])

    # 5) prognosis
    feat_row["rule_N"] = n_stage
    Xp = prog_features(feat_row, prog["clinical_colmap"], prog["feature_spec"])
    risk = predict_risk_from_state(prog["state"], Xp)

    return out_img, t_stage, n_stage, risk


def write_outputs(out_dir, pid, seg_img, t_stage, n_stage, risk):
    seg_dir = Path(out_dir) / "images" / "head-neck-tumor-segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(seg_img, str(seg_dir / f"{pid}.mha"))
    json.dump(t_stage, open(Path(out_dir) / "t-stage.json", "w"))
    json.dump(n_stage, open(Path(out_dir) / "n-stage.json", "w"))
    json.dump(float(risk), open(Path(out_dir) / "rfs.json", "w"))


if __name__ == "__main__":
    print("This module is imported by the container entrypoint; wire GC I/O in the entrypoint.")
    
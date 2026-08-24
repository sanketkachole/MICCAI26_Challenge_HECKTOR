#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 07: build predicted-mask metadata for OOF staging retraining.
Reads the 782 out-of-fold nnU-Net validation masks, applies the SAME node clean-up
as the container, and recomputes geometry with a ROBUST body-centroid midline
(fixes laterality for off-center patients). Writes case_metadata_pred.csv.

USAGE (BigRed CPU):
  python 07_oof_pred_metadata.py \
     --pred-root /N/scratch/$USER/hecktor_nnunet/results/Dataset501_HECKTOR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
     --data-root "/N/.../HECKTOR 2026 Training Data/HECKTOR 2026 Training Data" \
     --out outputs/eda/case_metadata_pred.csv
"""
import argparse, glob, os, re
from pathlib import Path
import numpy as np, pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

MIDLINE_DEADZONE_MM = 6.0
# Winning settings from the BigRed sweep (09_tune_nodes.py): balanced accuracy 0.657
MERGE_MM = 2.0      # close gaps this size to merge split node pieces
MIN_ML = 1.5        # drop any node smaller than this volume outright
SUV_THR = 3.0       # "both" mode: keep only if SUV is at least this...
DIAM_MM = 8.0       # ...AND max diameter is at least this (stricter than either/or)

def cleanup_nodes(label_arr, pet_arr, spacing_zyx, suv_thr=SUV_THR, diam_mm=DIAM_MM,
                  min_ml=MIN_ML, merge_mm=MERGE_MM):
    """Remove false-positive GTVn components. The keep/discard DECISION for each node
    is measured on the MERGED group (so a real node split into pieces by segmentation
    noise is judged as one node, matching how it was validated in the tuning sweep).
    The OUTPUT mask only ever removes or keeps ORIGINAL predicted voxels -- it never
    adds synthetic bridging voxels, so segmentation Dice is not affected by merging.
    The closing operation is restricted to a small box around the nodes (not the whole
    scan), which is what actually costs the time on a full-body volume."""
    sz, sy, sx = spacing_zyx
    voxvol_ml = (sz*sy*sx)/1000.0
    out = label_arr.copy()
    gtvn = (label_arr == 2)
    if gtvn.any():
        idx0 = np.argwhere(gtvn)
        pad = max(1, int(np.ceil(merge_mm / min(sz, sy, sx)))) + 1
        lo = np.maximum(idx0.min(0) - pad, 0)
        hi = np.minimum(idx0.max(0) + pad, np.array(gtvn.shape))
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        gtvn_c = gtvn[sl]
        pet_c = pet_arr[sl] if pet_arr is not None else None
        g = gtvn_c
        if merge_mm > 0:
            r = (max(1,int(round(merge_mm/sz))), max(1,int(round(merge_mm/sy))), max(1,int(round(merge_mm/sx))))
            g = ndi.binary_closing(gtvn_c, structure=np.ones((r[0]*2+1,r[1]*2+1,r[2]*2+1), int))
        lab, n = ndi.label(g, structure=np.ones((3,3,3), int))
        drop_mask_c = np.zeros_like(gtvn_c)
        for c in range(1, n+1):
            comp = lab == c                       # the MERGED group (may include bridge voxels)
            idx = np.argwhere(comp); nvox = idx.shape[0]
            vol_ml = nvox * voxvol_ml
            maxdim = float(((idx.max(0)-idx.min(0)+1)*np.array([sz,sy,sx])).max())
            suvmax = float(pet_c[comp].max()) if pet_c is not None else 0.0
            keep = (vol_ml >= min_ml) and (suvmax >= suv_thr) and (maxdim >= diam_mm)
            if not keep:
                drop_mask_c |= (comp & gtvn_c)
        # translate the crop's drop-mask back to full-volume coords and zero those voxels
        full_drop = np.zeros_like(gtvn)
        full_drop[sl] = drop_mask_c
        out[full_drop] = 0
    gtvp = (out == 1)
    if gtvp.any():
        labp, npn = ndi.label(gtvp, structure=np.ones((3,3,3), int))
        if npn > 1:
            sizes = ndi.sum(np.ones_like(labp), labp, index=range(1, npn+1))
            big = int(np.argmax(sizes))+1
            for k in range(1, npn+1):
                if k != big: out[labp == k] = 0
    return out

def body_midline_x_mm(ct_arr, sx):
    """Body center-of-mass x (mm). Fast/downsampled to avoid hanging on huge full-body CTs."""
    step0 = max(1, ct_arr.shape[0]//64)
    step1 = max(1, ct_arr.shape[1]//128)
    step2 = max(1, ct_arr.shape[2]//128)
    small = ct_arr[::step0, ::step1, ::step2]
    body = small > -500
    counts = body.sum(axis=(0,1)).astype(np.float64)
    if counts.sum() == 0:
        return (ct_arr.shape[2] / 2.0) * sx
    x_idx = np.arange(small.shape[2])
    mean_idx_small = float((x_idx * counts).sum() / counts.sum())
    return mean_idx_small * step2 * sx

def analyze(label_arr, spacing_zyx, midline_x_mm, merge_mm=MERGE_MM):
    """Count/size nodes for STAGING. Uses the SAME merge grouping as cleanup_nodes,
    so a real node that segmentation split into pieces is counted as one -- this
    must match the tuning sweep's logic or the retrained N-rule will be miscalibrated.
    Closing is restricted to a small box around the nodes for speed on large scans."""
    sz, sy, sx = spacing_zyx
    voxvol_ml = (sz*sy*sx)/1000.0
    gtvp_ml = float((label_arr == 1).sum()) * voxvol_ml
    gtvn = (label_arr == 2)
    if not gtvn.any():
        return dict(gtvp_ml=round(gtvp_ml,3), n_gtvn=0, max_node_mm=0.0,
                    gtvn_total_ml=0.0, laterality="none")
    idx0 = np.argwhere(gtvn)
    pad = max(1, int(np.ceil(merge_mm / min(sz, sy, sx)))) + 1
    lo = np.maximum(idx0.min(0) - pad, 0)
    hi = np.minimum(idx0.max(0) + pad, np.array(gtvn.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    gtvn_c = gtvn[sl]
    z0, y0, x0 = lo   # offset to translate cropped indices back to full-volume mm
    g = gtvn_c
    if merge_mm > 0:
        r = (max(1,int(round(merge_mm/sz))), max(1,int(round(merge_mm/sy))), max(1,int(round(merge_mm/sx))))
        g = ndi.binary_closing(gtvn_c, structure=np.ones((r[0]*2+1,r[1]*2+1,r[2]*2+1), int))
    lab, n = ndi.label(g, structure=np.ones((3,3,3), int))
    max_mm, tot_ml, sides = 0.0, 0.0, set()
    for c in range(1, n+1):
        idx = np.argwhere(lab == c)
        max_mm = max(max_mm, float(((idx.max(0)-idx.min(0)+1)*np.array([sz,sy,sx])).max()))
        tot_ml += idx.shape[0]*voxvol_ml
        cx_mm = float((idx[:,2].mean() + x0))*sx
        if cx_mm > midline_x_mm + MIDLINE_DEADZONE_MM: sides.add("R")
        elif cx_mm < midline_x_mm - MIDLINE_DEADZONE_MM: sides.add("L")
    lat = "bilateral" if len(sides)>=2 else ("unilateral" if len(sides)==1 else "midline")
    return dict(gtvp_ml=round(gtvp_ml,3), n_gtvn=int(n), max_node_mm=round(max_mm,1),
                gtvn_total_ml=round(tot_ml,3), laterality=lat)

def build_file_index(data_root):
    """Scan data_root ONCE and return {patient_id: path} for CT and PET files.
    Much faster than a recursive glob per case (which was the other slow spot)."""
    ct_index, pet_index = {}, {}
    for root, _dirs, files in os.walk(data_root):
        pid = os.path.basename(root)
        for b in files:
            if not b.endswith(".nii.gz"):
                continue
            if re.search(r"CT\.nii\.gz$", b, re.I):
                ct_index[pid] = os.path.join(root, b)
            elif re.search(r"(PT|PET)\.nii\.gz$", b, re.I):
                pet_index[pid] = os.path.join(root, b)
    return ct_index, pet_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/eda/case_metadata_pred.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    masks = sorted(glob.glob(os.path.join(args.pred_root, "fold_*/validation/*.nii.gz")))
    print(f"[found] {len(masks)} OOF predicted masks", flush=True)

    print("[index] scanning data-root for CT/PET once...", flush=True)
    ct_index, pet_index = build_file_index(args.data_root)
    print(f"[index] {len(ct_index)} CT / {len(pet_index)} PET", flush=True)
    if len(ct_index) == 0:
        print("[FATAL] no CT files found under --data-root. Check the path.", flush=True)
        return

    import time
    rows = []
    for i, mpath in enumerate(masks, 1):
        pid = os.path.basename(mpath).replace(".nii.gz", "")
        t0 = time.time()
        try:
            mimg = sitk.ReadImage(mpath); marr = sitk.GetArrayFromImage(mimg)
            sp = mimg.GetSpacing(); spacing_zyx = (sp[2], sp[1], sp[0])
            petp = pet_index.get(pid)
            if petp:
                pet = sitk.ReadImage(petp)
                if pet.GetSize() != mimg.GetSize() or not np.allclose(pet.GetSpacing(), sp, atol=1e-3):
                    pet = sitk.Resample(pet, mimg, sitk.Transform(), sitk.sitkLinear, 0.0, pet.GetPixelID())
                parr = sitk.GetArrayFromImage(pet)
            else:
                parr = None
            cleaned = cleanup_nodes(marr, parr, spacing_zyx)
            ctp = ct_index.get(pid)
            if ctp:
                ct = sitk.ReadImage(ctp)
                if ct.GetSize() != mimg.GetSize() or not np.allclose(ct.GetSpacing(), sp, atol=1e-3):
                    ct = sitk.Resample(ct, mimg, sitk.Transform(), sitk.sitkLinear, -1000.0, ct.GetPixelID())
                mid = body_midline_x_mm(sitk.GetArrayFromImage(ct), sp[0])
            else:
                mid = (marr.shape[2]/2.0)*sp[0]
            g = analyze(cleaned, spacing_zyx, mid)
            g["PatientID"] = pid; g["center"] = pid.split("-")[0]
            rows.append(g)
        except Exception as e:
            rows.append({"PatientID": pid, "center": pid.split("-")[0], "error": str(e)[:100]})
        dt = time.time() - t0
        if i <= 5 or dt > 5 or i % 50 == 0 or i == len(masks):
            print(f"  ...{i}/{len(masks)}  ({pid}: {dt:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[write] {args.out}  ({df.shape[0]} rows)", flush=True)
    if "error" in df: print(f"[warn] rows with error: {df['error'].notna().sum()}", flush=True)

if __name__ == "__main__":
    main()
    
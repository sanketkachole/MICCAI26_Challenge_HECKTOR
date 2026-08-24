#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 39 (Phase B): lesion DISSEMINATION features.

Phase A finding: nearly all our prognostic signal is "how much tumour is there"
(volume, TLG, Energy are collinear). The one genuinely untried axis is
"how far apart / how spread out is the disease".

Computes per patient, from the SAME predicted masks the container uses:
  Dmax_mm            - max distance between any two lesion voxels (disease spread)
  Dmax_norm          - Dmax / body length (removes scanner field-of-view effects)
  n_lesions          - connected components across GTVp+GTVn after cleanup
  centroid_spread_mm - mean pairwise distance between lesion centroids
  dist_p_to_n_max_mm - primary centroid to farthest node
  frac_largest       - largest lesion / total lesion volume (bulky vs scattered)
  gtvn_to_gtvp_ratio - nodal burden relative to primary
  suv_hetero         - SUVmax/SUVmean over all lesions (metabolic heterogeneity)

Writes outputs/eda/dissemination.csv. Resume-safe (jsonl), flushed progress.

USAGE (batch, HECKTOR env):
  python 39_dissemination.py \
    --pred-root .../nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
    --data-root ".../HECKTOR 2026 Training Data" \
    --out outputs/eda/dissemination.csv
"""
import argparse, os, re, glob, sys, json, time
import multiprocessing as mp
import numpy as np
print("[boot] importing SimpleITK/scipy...", flush=True)
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
print("[boot] imports done", flush=True)

# identical cleanup to the container
MERGE_MM, MIN_ML, SUV_THR, DIAM_MM = 2.0, 1.5, 3.0, 8.0


def cleanup_nodes(label_arr, pet_arr, spacing_zyx):
    sz, sy, sx = spacing_zyx
    voxvol_ml = (sz * sy * sx) / 1000.0
    out = label_arr.copy()
    gtvn = (label_arr == 2)
    if gtvn.any():
        idx0 = np.argwhere(gtvn)
        pad = max(1, int(np.ceil(MERGE_MM / min(sz, sy, sx)))) + 1
        lo = np.maximum(idx0.min(0) - pad, 0)
        hi = np.minimum(idx0.max(0) + pad, np.array(gtvn.shape))
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        gc = gtvn[sl]
        pc = pet_arr[sl] if pet_arr is not None else None
        r = (max(1, int(round(MERGE_MM / sz))),
             max(1, int(round(MERGE_MM / sy))),
             max(1, int(round(MERGE_MM / sx))))
        g = ndi.binary_closing(gc, structure=np.ones((r[0]*2+1, r[1]*2+1, r[2]*2+1), int))
        lab, n = ndi.label(g, structure=np.ones((3, 3, 3), int))
        drop = np.zeros_like(gc)
        for c in range(1, n + 1):
            comp = lab == c
            idx = np.argwhere(comp)
            vol = idx.shape[0] * voxvol_ml
            maxdim = float(((idx.max(0) - idx.min(0) + 1) * np.array([sz, sy, sx])).max())
            suvmax = float(pc[comp].max()) if pc is not None else 0.0
            if not ((vol >= MIN_ML) and (suvmax >= SUV_THR) and (maxdim >= DIAM_MM)):
                drop |= (comp & gc)
        full = np.zeros_like(gtvn)
        full[sl] = drop
        out[full] = 0
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


def subsample(idx, cap=4000, seed=0):
    """Cap point count so pdist stays fast; random subsample preserves extent well."""
    if idx.shape[0] <= cap:
        return idx
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx.shape[0], cap, replace=False)
    return idx[sel]


def body_length_mm(ct_arr, sz):
    """Approximate scan extent in z where body is present (normalises Dmax)."""
    if ct_arr is None:
        return np.nan
    body = (ct_arr > -500)
    zs = np.where(body.any(axis=(1, 2)))[0]
    if zs.size < 2:
        return float(ct_arr.shape[0]) * sz
    return float(zs[-1] - zs[0] + 1) * sz


def features_for(mask, pet, spacing_zyx, ct_arr):
    sz, sy, sx = spacing_zyx
    vox = np.array([sz, sy, sx])
    voxvol_ml = (sz * sy * sx) / 1000.0
    out = {}

    lesion = (mask > 0)
    out["has_lesion"] = int(lesion.any())
    if not lesion.any():
        return out

    idx = np.argwhere(lesion)
    pts = subsample(idx) * vox
    if pts.shape[0] > 1:
        d = pdist(pts)
        out["Dmax_mm"] = float(d.max())
        out["spread_mean_mm"] = float(d.mean())
    else:
        out["Dmax_mm"] = 0.0
        out["spread_mean_mm"] = 0.0

    bl = body_length_mm(ct_arr, sz)
    out["body_len_mm"] = float(bl) if np.isfinite(bl) else np.nan
    out["Dmax_norm"] = float(out["Dmax_mm"] / bl) if (np.isfinite(bl) and bl > 0) else np.nan

    lab, n = ndi.label(lesion, structure=np.ones((3, 3, 3), int))
    out["n_lesions"] = int(n)
    if n > 0:
        sizes = np.array(ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1)), dtype=float)
        tot = sizes.sum()
        out["frac_largest"] = float(sizes.max() / tot) if tot > 0 else np.nan
        out["total_lesion_ml"] = float(tot * voxvol_ml)
        cents = np.array(ndi.center_of_mass(lesion, lab, index=range(1, n + 1)))
        if cents.ndim == 1:
            cents = cents[None, :]
        if cents.shape[0] > 1:
            dc = pdist(cents * vox)
            out["centroid_spread_mm"] = float(dc.mean())
            out["centroid_max_mm"] = float(dc.max())
        else:
            out["centroid_spread_mm"] = 0.0
            out["centroid_max_mm"] = 0.0

    p = (mask == 1)
    nn = (mask == 2)
    if p.any() and nn.any():
        pc = np.array(ndi.center_of_mass(p)) * vox
        nlab, nn_n = ndi.label(nn, structure=np.ones((3, 3, 3), int))
        ncents = np.array(ndi.center_of_mass(nn, nlab, index=range(1, nn_n + 1)))
        if ncents.ndim == 1:
            ncents = ncents[None, :]
        dists = np.linalg.norm(ncents * vox - pc, axis=1)
        out["dist_p_to_n_max_mm"] = float(dists.max())
        out["dist_p_to_n_mean_mm"] = float(dists.mean())
    else:
        out["dist_p_to_n_max_mm"] = 0.0
        out["dist_p_to_n_mean_mm"] = 0.0

    pv = float(p.sum()) * voxvol_ml
    nv = float(nn.sum()) * voxvol_ml
    out["gtvn_to_gtvp_ratio"] = float(nv / pv) if pv > 0 else np.nan

    if pet is not None:
        vals = pet[lesion]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0 and vals.mean() > 0:
            out["suv_max_all"] = float(vals.max())
            out["suv_mean_all"] = float(vals.mean())
            out["suv_hetero"] = float(vals.max() / vals.mean())
            out["suv_std_all"] = float(vals.std())
    return out


def process_one(task):
    """Worker: all dissemination features for one patient."""
    pid, mpath, ct_path, pet_path = task
    row = {"PatientID": pid, "center": pid.split("-")[0]}
    try:
        mimg = sitk.ReadImage(mpath)
        marr = sitk.GetArrayFromImage(mimg)
        sp = mimg.GetSpacing()
        spacing_zyx = (sp[2], sp[1], sp[0])
        pet = None
        if pet_path:
            pi = sitk.ReadImage(pet_path)
            if pi.GetSize() != mimg.GetSize():
                pi = sitk.Resample(pi, mimg, sitk.Transform(),
                                   sitk.sitkLinear, 0.0, pi.GetPixelID())
            pet = sitk.GetArrayFromImage(pi)
        ct_arr = None
        if ct_path:
            ci = sitk.ReadImage(ct_path)
            if ci.GetSize() != mimg.GetSize():
                ci = sitk.Resample(ci, mimg, sitk.Transform(),
                                   sitk.sitkLinear, -1000.0, ci.GetPixelID())
            ct_arr = sitk.GetArrayFromImage(ci)
        cleaned = cleanup_nodes(marr, pet, spacing_zyx)
        del marr
        row.update(features_for(cleaned, pet, spacing_zyx, ct_arr))
    except Exception as e:
        row["error"] = ("%s: %s" % (type(e).__name__, e))[:120]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/eda/dissemination.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel processes")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    jsonl = args.out.replace(".csv", ".jsonl")

    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                done.add(json.loads(line)["PatientID"])
            except Exception:
                pass
        print("[resume] %d already done" % len(done), flush=True)

    masks = sorted(glob.glob(os.path.join(args.pred_root, "fold_*/validation/*.nii.gz")))
    if args.limit:
        masks = masks[:args.limit]
    print("[masks] %d" % len(masks), flush=True)

    ct_index, pet_index = {}, {}
    for root, _d, files in os.walk(args.data_root):
        pid = os.path.basename(root)
        for b in files:
            if not b.endswith(".nii.gz"):
                continue
            if re.search(r"CT\.nii\.gz$", b, re.I):
                ct_index[pid] = os.path.join(root, b)
            elif re.search(r"(PT|PET)\.nii\.gz$", b, re.I):
                pet_index[pid] = os.path.join(root, b)
    print("[index] %d CT / %d PET" % (len(ct_index), len(pet_index)), flush=True)

    todo = []
    for m in masks:
        pid = os.path.basename(m).replace(".nii.gz", "")
        if pid in done:
            continue
        todo.append((pid, m, ct_index.get(pid), pet_index.get(pid)))
    print("[todo] %d patients across %d workers" % (len(todo), args.workers), flush=True)

    t0 = time.time()
    if todo:
        with mp.Pool(processes=args.workers, maxtasksperchild=16) as pool, \
             open(jsonl, "a") as fout:
            for i, row in enumerate(pool.imap_unordered(process_one, todo), 1):
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                if i <= 3 or i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    eta = (el / i) * (len(todo) - i) / 60.0
                    print("  ...%d/%d (%.1f min elapsed, ~%.0f min left)"
                          % (i, len(todo), el / 60, eta), flush=True)

    import pandas as pd
    rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset="PatientID").sort_values("PatientID")
    df.to_csv(args.out, index=False)
    print("[write] %s (%d rows x %d cols)" % (args.out, df.shape[0], df.shape[1]), flush=True)
    if "error" in df:
        print("[warn] errors: %d" % int(df["error"].notna().sum()), flush=True)


if __name__ == "__main__":
    main()

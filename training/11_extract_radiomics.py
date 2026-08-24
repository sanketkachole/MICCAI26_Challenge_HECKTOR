#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 11: extract radiomics from OOF PREDICTED masks.

For each patient, computes PyRadiomics features for 4 combinations:
    CT x GTVp,  PET x GTVp,  CT x GTVn,  PET x GTVn
using the SAME node cleanup as the container (merge 2mm, SUV>=3 AND size>=8mm,
min 1.5 mL), so features match what the deployed model will actually see.

Output: one row per patient, wide table -> outputs/eda/radiomics_pred.csv

USAGE (Slurm, CPU, radiomics env):
  python -u 11_extract_radiomics.py \
    --pred-root /N/scratch/$USER/hecktor_nnunet/results/Dataset501_HECKTOR/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres \
    --data-root "/N/.../HECKTOR 2026 Training Data" \
    --out outputs/eda/radiomics_pred.csv
"""
import argparse, os, re, glob, sys, time, logging, json
import multiprocessing as mp

# Keep each worker single-threaded: with N processes each spawning N threads,
# the cores get oversubscribed and everything runs slower. Must be set BEFORE numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "SIMPLEITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np, pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

# quiet pyradiomics chatter
logging.getLogger("radiomics").setLevel(logging.ERROR)
from radiomics import featureextractor

# ---- node cleanup params (must match the container / step 07) ----
MERGE_MM = 2.0
MIN_ML   = 1.5
SUV_THR  = 3.0
DIAM_MM  = 8.0


def cleanup_nodes(label_arr, pet_arr, spacing_zyx,
                  suv_thr=SUV_THR, diam_mm=DIAM_MM, min_ml=MIN_ML, merge_mm=MERGE_MM):
    """Same logic as the container: decide keep/drop on the MERGED group, but only
    ever erase ORIGINAL voxels. Cropped to a small box for speed."""
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
        pet_c  = pet_arr[sl] if pet_arr is not None else None
        g = gtvn_c
        if merge_mm > 0:
            r = (max(1,int(round(merge_mm/sz))), max(1,int(round(merge_mm/sy))), max(1,int(round(merge_mm/sx))))
            g = ndi.binary_closing(gtvn_c, structure=np.ones((r[0]*2+1, r[1]*2+1, r[2]*2+1), int))
        lab, n = ndi.label(g, structure=np.ones((3,3,3), int))
        drop_c = np.zeros_like(gtvn_c)
        for c in range(1, n+1):
            comp = lab == c
            idx = np.argwhere(comp)
            vol_ml = idx.shape[0] * voxvol_ml
            maxdim = float(((idx.max(0)-idx.min(0)+1) * np.array([sz,sy,sx])).max())
            suvmax = float(pet_c[comp].max()) if pet_c is not None else 0.0
            keep = (vol_ml >= min_ml) and (suvmax >= suv_thr) and (maxdim >= diam_mm)
            if not keep:
                drop_c |= (comp & gtvn_c)
        full_drop = np.zeros_like(gtvn)
        full_drop[sl] = drop_c
        out[full_drop] = 0
    # GTVp: keep only the largest connected component
    gtvp = (out == 1)
    if gtvp.any():
        labp, npn = ndi.label(gtvp, structure=np.ones((3,3,3), int))
        if npn > 1:
            sizes = ndi.sum(np.ones_like(labp), labp, index=range(1, npn+1))
            big = int(np.argmax(sizes)) + 1
            for k in range(1, npn+1):
                if k != big:
                    out[labp == k] = 0
    return out


def build_extractor(bin_width, interpolator="sitkBSpline"):
    """PyRadiomics extractor. Resample to 2mm isotropic so texture features are
    comparable across centers with different voxel sizes. The interpolator is
    configurable: sitkBSpline is highest quality but on coarse-z scans it can
    allocate a huge intermediate array and OOM; sitkLinear is far lighter and
    is used as a fallback for those cases."""
    settings = {
        "resampledPixelSpacing": [2.0, 2.0, 2.0],
        "interpolator": interpolator,
        "padDistance": 10,
        "geometryTolerance": 1e-3,
        "binWidth": bin_width,
        "label": 1,
    }
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.disableAllFeatures()
    for cls in ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        ex.enableFeatureClassByName(cls)
    return ex


def binary_image_like(ref_img, arr_bool):
    m = sitk.GetImageFromArray(arr_bool.astype(np.uint8))
    m.CopyInformation(ref_img)
    return m


def extract_one(img, mask_img, extractor, prefix):
    """Run pyradiomics for one (image, binary mask) pair -> flat dict."""
    res = extractor.execute(img, mask_img)
    feats = {}
    for k, v in res.items():
        if k.startswith("diagnostics"):
            continue
        try:
            feats[f"{prefix}_{k.replace('original_','')}"] = float(v)
        except (TypeError, ValueError):
            pass
    return feats


def resample_to(img, ref, default, interp=sitk.sitkLinear):
    if img.GetSize() == ref.GetSize() and np.allclose(img.GetSpacing(), ref.GetSpacing(), atol=1e-3):
        return img
    return sitk.Resample(img, ref, sitk.Transform(), interp, default, img.GetPixelID())


def process_one(task):
    """Worker: extract all radiomics for one patient. Returns a flat dict."""
    pid, mpath, ct_path, pet_path, interp = task
    row = {"PatientID": pid, "center": pid.split("-")[0]}
    try:
        ex_ct  = build_extractor(bin_width=25.0, interpolator=interp)   # CT: Hounsfield units
        ex_pet = build_extractor(bin_width=0.3,  interpolator=interp)   # PET: SUV units

        mimg = sitk.ReadImage(mpath)
        marr = sitk.GetArrayFromImage(mimg)
        sp = mimg.GetSpacing()
        spacing_zyx = (sp[2], sp[1], sp[0])

        ct  = resample_to(sitk.ReadImage(ct_path), mimg, -1000.0)
        pet = resample_to(sitk.ReadImage(pet_path), mimg, 0.0) if pet_path else None
        parr = sitk.GetArrayFromImage(pet) if pet is not None else None

        cleaned = cleanup_nodes(marr, parr, spacing_zyx)
        del marr, parr          # big arrays no longer needed -> free before radiomics

        for lbl, tag in ((1, "gtvp"), (2, "gtvn")):
            m_bool = (cleaned == lbl)
            row[f"has_{tag}"] = int(m_bool.any())
            if not m_bool.any():
                continue
            m_img = binary_image_like(mimg, m_bool)
            try:
                row.update(extract_one(ct, m_img, ex_ct, f"ct_{tag}"))
            except Exception as e:
                row[f"err_ct_{tag}"] = str(e)[:80]
            if pet is not None:
                try:
                    row.update(extract_one(pet, m_img, ex_pet, f"pt_{tag}"))
                except Exception as e:
                    row[f"err_pt_{tag}"] = str(e)[:80]
            del m_bool, m_img
        del cleaned, ct, pet, mimg
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:120]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/eda/radiomics_pred.csv")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N patients")
    ap.add_argument("--only", type=str, default="", help="process only this one patient id")
    ap.add_argument("--interp", type=str, default="sitkBSpline",
                    help="resampling interpolator (sitkBSpline default, sitkLinear = low memory)")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # incremental output: one JSON line per patient, written as soon as it finishes.
    # if the job dies (time limit / OOM), rerunning SKIPS whatever is already done.
    jsonl_path = args.out.replace(".csv", ".jsonl")

    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["PatientID"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} patients already done in {jsonl_path}", flush=True)

    masks = sorted(glob.glob(os.path.join(args.pred_root, "fold_*/validation/*.nii.gz")))
    if args.limit:
        masks = masks[:args.limit]
    print(f"[masks] {len(masks)}", flush=True)

    print("[index] scanning data-root once...", flush=True)
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
    print(f"[index] {len(ct_index)} CT / {len(pet_index)} PET", flush=True)
    if not ct_index:
        print("[FATAL] no CT found under --data-root", flush=True); sys.exit(1)

    tasks = []
    for mpath in masks:
        pid = os.path.basename(mpath).replace(".nii.gz", "")
        if args.only and pid != args.only:
            continue
        if pid in done or pid not in ct_index:
            continue
        tasks.append((pid, mpath, ct_index[pid], pet_index.get(pid), args.interp))
    print(f"[tasks] {len(tasks)} remaining across {args.workers} workers", flush=True)

    t_start = time.time()
    if tasks:
        with mp.Pool(processes=args.workers, maxtasksperchild=8) as pool, \
             open(jsonl_path, "a") as fout:
            for i, row in enumerate(pool.imap_unordered(process_one, tasks), 1):
                fout.write(json.dumps(row) + "\n")
                fout.flush()          # persist immediately -> nothing lost on a kill
                if i <= 3 or i % 25 == 0 or i == len(tasks):
                    el = time.time() - t_start
                    eta = (el / i) * (len(tasks) - i) / 60.0
                    print(f"  ...{i}/{len(tasks)}  ({el/60:.1f} min elapsed, ~{eta:.0f} min left)", flush=True)

    # assemble the final CSV from everything on disk
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows).drop_duplicates(subset="PatientID").sort_values("PatientID")
    df.to_csv(args.out, index=False)
    print(f"[write] {args.out}  ({df.shape[0]} rows x {df.shape[1]} cols)", flush=True)
    if "error" in df:
        print(f"[warn] rows with error: {int(df['error'].notna().sum())}", flush=True)


if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""
HECKTOR 2026 - Step 51: does cropping change the segmentation?

Runs the SAME 5-fold nnU-Net ensemble twice on the SAME patients:
  arm A "nocrop" : the full image, exactly as the failed containers did
  arm B "crop"   : cropped to a head-and-neck slab by crop_utils

Then scores both against ground truth and prints the difference.

The point of the test
---------------------
The CT channel is normalised with global statistics from the dataset
fingerprint, so cropping cannot change it. The PET channel is z-scored PER
IMAGE, so cropping DOES change it. This script measures whether that matters.

If crop Dice >= nocrop Dice, the crop is free and we put it in the container.

Case selection
--------------
Half the cases are the largest-FOV patients (these are the ones that killed the
container). Half are median-sized head-and-neck patients (to confirm the crop
does not hurt the normal cases either).

Stages
------
  prep    build both input folders
  predict run nnUNetv2_predict on each arm
  score   paste back and compute Dice
  all     all three (default)

Usage
-----
python 51_crop_check.py --stage all \
    --data-root "/path/to/HECKTOR 2026 Training Data" \
    --extent outputs/eda/disease_extent.csv \
    --work /N/scratch/$USER/cropcheck \
    --n-large 8 --n-typical 8
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_utils  # noqa: E402


# --------------------------------------------------------------- case lookup
def find_case_files(data_root):
    """Return {patient_id: (ct_path, pet_path, label_path)}."""
    root = Path(data_root)
    out = {}
    ct_files = sorted(root.glob("**/*__CT.nii.gz")) or sorted(root.glob("**/*CT.nii.gz"))
    for ct in ct_files:
        d = ct.parent
        pet = None
        for pat in ("*__PT.nii.gz", "*_PT.nii.gz", "*PT.nii.gz", "*PET.nii.gz"):
            hits = list(d.glob(pat))
            if hits:
                pet = hits[0]
                break
        lab = None
        for f in sorted(d.glob("*.nii.gz")):
            if not re.search(r"__?(CT|PT|PET)\.nii\.gz$", f.name, re.IGNORECASE):
                lab = f
                break
        if pet is not None and lab is not None:
            out[d.name] = (str(ct), str(pet), str(lab))
    return out


def pick_cases(extent_csv, n_large, n_typical):
    rows = []
    with open(extent_csv) as f:
        for r in csv.DictReader(f):
            try:
                r["ct_z_extent_mm"] = float(r["ct_z_extent_mm"])
            except ValueError:
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["ct_z_extent_mm"], reverse=True)

    large = [r["PatientID"] for r in rows[:n_large]]

    med = np.median([r["ct_z_extent_mm"] for r in rows])
    typical_sorted = sorted(rows, key=lambda r: abs(r["ct_z_extent_mm"] - med))
    typical = [r["PatientID"] for r in typical_sorted[:n_typical]]

    z = {r["PatientID"]: r["ct_z_extent_mm"] for r in rows}
    return large, typical, z


# ------------------------------------------------------------------- stages
def stage_prep(cases, files, work, boxes_path):
    nocrop_in = work / "in_nocrop"
    crop_in = work / "in_crop"
    nocrop_in.mkdir(parents=True, exist_ok=True)
    crop_in.mkdir(parents=True, exist_ok=True)

    boxes = json.load(open(boxes_path)) if boxes_path.exists() else {}

    for i, pid in enumerate(cases, 1):
        ct_path, pet_path, _ = files[pid]
        t0 = time.time()

        done_nc = (nocrop_in / f"{pid}_0001.nii.gz").exists()
        done_c = (crop_in / f"{pid}_0001.nii.gz").exists() and pid in boxes
        if done_nc and done_c:
            print(f"  [{i}/{len(cases)}] {pid} already prepared", flush=True)
            continue

        ct = sitk.ReadImage(ct_path)
        pet = sitk.ReadImage(pet_path)

        # --- arm A: no crop. PET resampled onto the FULL CT grid (what the
        #     failing container did), so the comparison is honest.
        if not done_nc:
            pet_full = sitk.Resample(pet, ct, sitk.Transform(),
                                     sitk.sitkLinear, 0.0, pet.GetPixelID())
            sitk.WriteImage(ct, str(nocrop_in / f"{pid}_0000.nii.gz"), True)
            sitk.WriteImage(pet_full, str(nocrop_in / f"{pid}_0001.nii.gz"), True)
            del pet_full

        # --- arm B: crop
        if not done_c:
            ct_c, pet_c, index_xyz, size_xyz = crop_utils.crop_pair(ct, pet)
            sitk.WriteImage(ct_c, str(crop_in / f"{pid}_0000.nii.gz"), True)
            sitk.WriteImage(pet_c, str(crop_in / f"{pid}_0001.nii.gz"), True)
            boxes[pid] = {"index_xyz": list(index_xyz), "size_xyz": list(size_xyz)}
            print(f"  [{i}/{len(cases)}] {pid}  {crop_utils.describe(ct, index_xyz, size_xyz)}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
            json.dump(boxes, open(boxes_path, "w"), indent=1)

        del ct, pet

    json.dump(boxes, open(boxes_path, "w"), indent=1)
    print(f"[prep] inputs ready in {nocrop_in} and {crop_in}")


def stage_predict(work, results_dir, arm, folds):
    ind = work / f"in_{arm}"
    outd = work / f"out_{arm}"
    outd.mkdir(parents=True, exist_ok=True)

    todo = [p for p in sorted(ind.glob("*_0000.nii.gz"))
            if not (outd / (p.name.replace("_0000.nii.gz", ".nii.gz"))).exists()]
    if not todo:
        print(f"[predict:{arm}] nothing to do")
        return
    print(f"[predict:{arm}] {len(todo)} case(s) remaining", flush=True)

    env = dict(os.environ,
               nnUNet_results=str(results_dir),
               nnUNet_raw=str(work / "raw"),
               nnUNet_preprocessed=str(work / "prep"))
    cmd = ["nnUNetv2_predict", "-i", str(ind), "-o", str(outd),
           "-d", "Dataset501_HECKTOR", "-c", "3d_fullres",
           "-f", *folds,
           "-tr", "nnUNetTrainer_250epochs", "-p", "nnUNetPlans",
           "--disable_tta", "-npp", "1", "-nps", "1", "--continue_prediction"]
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"[predict:{arm}] done in {(time.time()-t0)/60:.1f} min", flush=True)


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0, sa, sb
    if sa == 0 or sb == 0:
        return 0.0, sa, sb
    return 2.0 * float((a & b).sum()) / (sa + sb), sa, sb


def stage_score(cases, files, work, boxes_path, out_csv):
    boxes = json.load(open(boxes_path))
    rows = []
    for pid in cases:
        _, _, lab_path = files[pid]
        p_nc = work / "out_nocrop" / f"{pid}.nii.gz"
        p_c = work / "out_crop" / f"{pid}.nii.gz"
        if not p_nc.exists() or not p_c.exists():
            print(f"  skip {pid}: missing prediction "
                  f"(nocrop={p_nc.exists()}, crop={p_c.exists()})")
            continue

        gt_img = sitk.ReadImage(str(lab_path))
        gt = sitk.GetArrayFromImage(gt_img)

        nc = sitk.GetArrayFromImage(sitk.ReadImage(str(p_nc)))
        cr_small = sitk.ReadImage(str(p_c))
        cr_img = crop_utils.paste_back(cr_small, gt_img, tuple(boxes[pid]["index_xyz"]))
        cr = sitk.GetArrayFromImage(cr_img)

        r = {"PatientID": pid}
        for lbl, name in ((1, "p"), (2, "n")):
            d_nc, gsz, _ = dice(gt == lbl, nc == lbl)
            d_cr, _, _ = dice(gt == lbl, cr == lbl)
            r[f"dice_{name}_nocrop"] = round(d_nc, 4)
            r[f"dice_{name}_crop"] = round(d_cr, 4)
            r[f"gt_{name}_vox"] = gsz
        # did the crop cut off any ground-truth disease?
        x0, y0, z0 = boxes[pid]["index_xyz"]
        dx, dy, dz = boxes[pid]["size_xyz"]
        inside = np.zeros_like(gt, dtype=bool)
        inside[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx] = True
        lost = int(((gt > 0) & ~inside).sum())
        r["gt_voxels_lost_by_crop"] = lost
        rows.append(r)
        print(f"  {pid:12s} GTVp {r['dice_p_nocrop']:.3f} -> {r['dice_p_crop']:.3f}   "
              f"GTVn {r['dice_n_nocrop']:.3f} -> {r['dice_n_crop']:.3f}   lost={lost}",
              flush=True)
        del gt, nc, cr

    if not rows:
        print("no scored cases")
        return

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mean(key, only_nonempty=None):
        vals = [r[key] for r in rows
                if only_nonempty is None or r[only_nonempty] > 0]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n===================== CROP vs NO CROP =====================")
    print(f"cases scored                 : {len(rows)}")
    print(f"ground-truth voxels lost     : {sum(r['gt_voxels_lost_by_crop'] for r in rows)}"
          "   <-- must be 0")
    print()
    print(f"{'':22s} {'no crop':>9s} {'crop':>9s} {'diff':>9s}")
    for name, label, filt in (("GTVp Dice", "p", "gt_p_vox"),
                              ("GTVn Dice", "n", "gt_n_vox")):
        a = mean(f"dice_{label}_nocrop", filt)
        b = mean(f"dice_{label}_crop", filt)
        print(f"{name:22s} {a:9.4f} {b:9.4f} {b-a:+9.4f}")
    a = (mean("dice_p_nocrop", "gt_p_vox") + mean("dice_n_nocrop", "gt_n_vox")) / 2
    b = (mean("dice_p_crop", "gt_p_vox") + mean("dice_n_crop", "gt_n_vox")) / 2
    print(f"{'MEAN (challenge score)':22s} {a:9.4f} {b:9.4f} {b-a:+9.4f}")
    print("===========================================================")
    print(f"[write] {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["prep", "predict", "score", "all"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--extent", default="outputs/eda/disease_extent.csv")
    ap.add_argument("--work", required=True)
    ap.add_argument("--results", required=True,
                    help="nnUNet_results dir (parent of Dataset501_HECKTOR)")
    ap.add_argument("--n-large", type=int, default=8)
    ap.add_argument("--n-typical", type=int, default=8)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out-csv", default="outputs/eda/crop_check.csv")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    boxes_path = work / "crop_boxes.json"
    folds = args.folds.split(",")

    files = find_case_files(args.data_root)
    large, typical, zext = pick_cases(args.extent, args.n_large, args.n_typical)
    cases = [p for p in (large + typical) if p in files]
    print(f"[cases] {len(cases)}  ({len(large)} large + {len(typical)} typical)")
    for p in cases:
        print(f"    {p:12s} CT z-extent {zext.get(p, float('nan')):7.1f} mm")

    if args.stage in ("prep", "all"):
        stage_prep(cases, files, work, boxes_path)
    if args.stage in ("predict", "all"):
        stage_predict(work, Path(args.results), "crop", folds)     # cheap arm first
        stage_predict(work, Path(args.results), "nocrop", folds)
    if args.stage in ("score", "all"):
        stage_score(cases, files, work, boxes_path, args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
HECKTOR 2026 - Step 50: how far below the top of the CT does the disease reach?

Purpose
-------
Before cropping the image for inference, we must know how big the crop must be
so that no tumour or node is ever cut off. This script measures that directly
from the 782 ground-truth masks.

For each patient it computes, in millimetres:
  ct_z_extent_mm   : total length of the CT scan in z
  depth_top_mm     : top of CT FOV  ->  top of the disease
  depth_bot_mm     : top of CT FOV  ->  bottom of the disease   <-- the number we need
  lesion_span_mm   : top of disease -> bottom of disease

Note on safety
--------------
We anchor at the top of the CT field of view, not the top of the body.
The top of the body is always at or below the top of the FOV, so this
measurement OVER-estimates the slab we need. Sizing the crop from it is safe.

Speed
-----
Reads only the CT header (no pixels) plus the label mask. Masks are mostly
zeros and compress well, so this is fast.

Usage
-----
python 50_measure_disease_extent.py \
    --data-root "/path/to/HECKTOR 2026 Training Data" \
    --out outputs/eda/disease_extent.csv
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def find_cases(data_root):
    """Return [(patient_id, ct_path, label_path), ...]."""
    root = Path(data_root)
    ct_files = sorted(root.glob("**/*__CT.nii.gz"))
    if not ct_files:
        ct_files = sorted(root.glob("**/*CT.nii.gz"))

    cases = []
    for ct_path in ct_files:
        patient_dir = ct_path.parent
        label_path = None
        for f in sorted(patient_dir.glob("*.nii.gz")):
            if not re.search(r"__?(CT|PT|PET)\.nii\.gz$", f.name, re.IGNORECASE):
                label_path = f
                break
        if label_path is not None:
            cases.append((patient_dir.name, str(ct_path), str(label_path)))
    return cases


def ct_z_range(ct_path):
    """Physical z of the two ends of the CT, header only. Returns (z_top, z_bottom, z_extent)."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(ct_path)
    reader.ReadImageInformation()
    size = reader.GetSize()
    origin = np.array(reader.GetOrigin(), dtype=float)
    spacing = np.array(reader.GetSpacing(), dtype=float)
    direction = np.array(reader.GetDirection(), dtype=float).reshape(3, 3)

    def phys_z(k):
        idx = np.array([0.0, 0.0, float(k)])
        return float((origin + direction @ (idx * spacing))[2])

    z0 = phys_z(0)
    z1 = phys_z(size[2] - 1)
    return max(z0, z1), min(z0, z1), abs(z1 - z0)


def lesion_z_range(label_path):
    """Physical z of the top-most and bottom-most lesion voxel. Returns (z_top, z_bot, n_vox)."""
    img = sitk.ReadImage(label_path)
    arr = sitk.GetArrayViewFromImage(img)          # axes are z, y, x
    present = (arr > 0).any(axis=(1, 2))           # which z slices have any lesion
    n_vox = int((arr > 0).sum())
    if not present.any():
        return None, None, 0

    k_lo = int(np.argmax(present))
    k_hi = int(len(present) - 1 - np.argmax(present[::-1]))

    z_a = float(img.TransformIndexToPhysicalPoint((0, 0, k_lo))[2])
    z_b = float(img.TransformIndexToPhysicalPoint((0, 0, k_hi))[2])
    return max(z_a, z_b), min(z_a, z_b), n_vox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/eda/disease_extent.csv")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N cases")
    args = ap.parse_args()

    cases = find_cases(args.data_root)
    if args.limit:
        cases = cases[: args.limit]
    print(f"[cases] {len(cases)}", flush=True)
    if not cases:
        print("ERROR: no cases found. Check --data-root.", file=sys.stderr)
        return 1

    rows = []
    t0 = time.time()
    for i, (pid, ct_path, label_path) in enumerate(cases, 1):
        try:
            ct_top, ct_bot, ct_ext = ct_z_range(ct_path)
            les_top, les_bot, n_vox = lesion_z_range(label_path)
            if les_top is None:
                rows.append(dict(PatientID=pid, ct_z_extent_mm=round(ct_ext, 1),
                                 depth_top_mm="", depth_bot_mm="",
                                 lesion_span_mm="", n_lesion_vox=0))
            else:
                rows.append(dict(
                    PatientID=pid,
                    ct_z_extent_mm=round(ct_ext, 1),
                    depth_top_mm=round(ct_top - les_top, 1),
                    depth_bot_mm=round(ct_top - les_bot, 1),
                    lesion_span_mm=round(les_top - les_bot, 1),
                    n_lesion_vox=n_vox,
                ))
        except Exception as e:
            print(f"  ERROR {pid}: {e}", flush=True)
            continue

        if i % 50 == 0 or i == len(cases):
            el = time.time() - t0
            print(f"  ...{i}/{len(cases)}  ({el/60:.1f} min elapsed)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["PatientID", "ct_z_extent_mm", "depth_top_mm", "depth_bot_mm",
              "lesion_span_mm", "n_lesion_vox"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {out_path}  ({len(rows)} rows)")

    # ---------------- summary ----------------
    d = np.array([r["depth_bot_mm"] for r in rows if r["depth_bot_mm"] != ""], dtype=float)
    z = np.array([r["ct_z_extent_mm"] for r in rows], dtype=float)
    big = d[np.array([r["ct_z_extent_mm"] for r in rows if r["depth_bot_mm"] != ""],
                     dtype=float) > 700.0]

    print("\n================ HOW BIG MUST THE CROP BE ================")
    print(f"cases with a lesion            : {len(d)} of {len(rows)}")
    print(f"CT z-extent  median / max      : {np.median(z):7.1f} / {z.max():7.1f} mm")
    print("\ndepth_bot_mm = top of CT down to the LOWEST disease voxel")
    for q in (50, 90, 95, 99):
        print(f"  p{q:<3d}                        : {np.percentile(d, q):7.1f} mm")
    print(f"  MAX                          : {d.max():7.1f} mm")
    if len(big):
        print(f"\nlarge-FOV cases only (CT z > 700 mm), n={len(big)}")
        print(f"  p95                          : {np.percentile(big, 95):7.1f} mm")
        print(f"  MAX                          : {big.max():7.1f} mm")
    print("\nRECOMMENDED SLAB = MAX + 60 mm safety margin"
          f"  ->  {int(np.ceil((d.max() + 60) / 10.0) * 10):d} mm")
    print("==========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

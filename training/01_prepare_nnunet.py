#!/usr/bin/env python3
"""
HECKTOR 2026 — Step 01: Build nnU-Net v2 raw dataset.

Use Dataset599 for 2-case smoke testing.
Use Dataset501 for the full training set.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

DATASET_NAME_TEMPLATE = "Dataset{dataset_id:03d}_HECKTOR"


def find_cases(data_root: str) -> list[tuple[str, str, str, str]]:
    """Return list of (patient_id, ct_path, pet_path, label_path)."""
    root = Path(data_root)

    ct_files = sorted(root.glob("**/*__CT.nii.gz"))
    if not ct_files:
        ct_files = sorted(root.glob("**/*CT.nii.gz"))

    cases: list[tuple[str, str, str, str]] = []

    for ct_path in ct_files:
        patient_dir = ct_path.parent
        patient_id = patient_dir.name

        pet_path = None
        pet_patterns = [
            f"*{patient_id}*PT*.nii.gz",
            "*__PT.nii.gz",
            "*_PT.nii.gz",
            "*PT.nii.gz",
            f"*{patient_id}*PET*.nii.gz",
            "*PET.nii.gz",
        ]

        for pattern in pet_patterns:
            hits = list(patient_dir.glob(pattern))
            if hits:
                pet_path = hits[0]
                break

        label_path = None
        for file_path in patient_dir.glob("*.nii.gz"):
            if not re.search(r"__?(CT|PT|PET)\.nii\.gz$", file_path.name, re.IGNORECASE):
                label_path = file_path
                break

        if pet_path is not None and label_path is not None:
            cases.append(
                (
                    patient_id,
                    str(ct_path),
                    str(pet_path),
                    str(label_path),
                )
            )

    return cases


def place_file(src: str, dst: str, use_symlink: bool) -> None:
    """Symlink or copy a file."""
    dst_path = Path(dst)
    if dst_path.exists() or dst_path.is_symlink():
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if use_symlink:
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def read_header(path: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Read image size and spacing only."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return tuple(reader.GetSize()), tuple(reader.GetSpacing())


def report_fov(cases: list[tuple[str, str, str, str]]) -> None:
    """Print CT field-of-view statistics."""
    z_extents = []
    xyz_sizes = []

    for patient_id, ct_path, _, _ in cases:
        size, spacing = read_header(ct_path)
        xyz_sizes.append(size)
        z_extents.append(size[2] * spacing[2])

    z_arr = np.asarray(z_extents, dtype=float)

    print("\n[FOV]")
    print(f"  sampled cases       : {len(cases)}")
    print(f"  CT z-extent mm min  : {z_arr.min():.1f}")
    print(f"  CT z-extent mm med  : {np.median(z_arr):.1f}")
    print(f"  CT z-extent mm p90  : {np.percentile(z_arr, 90):.1f}")
    print(f"  CT z-extent mm max  : {z_arr.max():.1f}")
    print("  note: >500 mm usually suggests large/full-body FOV.")


def convert_splits(
    splits_path: str,
    present_patient_ids: set[str],
    smoke_split: bool,
) -> list[dict[str, list[str]]]:
    """Create nnU-Net splits_final.json content."""
    patient_ids = sorted(present_patient_ids)

    if smoke_split:
        if len(patient_ids) < 2:
            raise ValueError("Smoke split needs at least 2 cases.")
        return [{"train": [patient_ids[0]], "val": [patient_ids[1]]}]

    with open(splits_path, "r", encoding="utf-8") as file:
        source = json.load(file)

    nnunet_splits = []
    for fold in sorted(source["folds"], key=lambda item: item["fold"]):
        train = [str(pid) for pid in fold["train"] if str(pid) in present_patient_ids]
        val = [str(pid) for pid in fold["val"] if str(pid) in present_patient_ids]

        if not train or not val:
            raise ValueError(
                f"Bad split after filtering. train={len(train)}, val={len(val)}. "
                "For smoke use --smoke-split."
            )

        nnunet_splits.append({"train": train, "val": val})

    return nnunet_splits


def write_dataset_json(dataset_dir: Path, num_training: int) -> None:
    """Write nnU-Net v2 dataset.json."""
    dataset_json: dict[str, Any] = {
        "channel_names": {
            "0": "CT",
            "1": "PET",
        },
        "labels": {
            "background": 0,
            "GTVp": 1,
            "GTVn": 2,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    out_path = dataset_dir / "dataset.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(dataset_json, file, indent=2)

    print(f"[write] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--nnunet-raw", required=True)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--link", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fov-only", action="store_true")
    parser.add_argument("--smoke-split", action="store_true")
    args = parser.parse_args()

    cases = find_cases(args.data_root)
    if not cases:
        sys.exit(f"No cases found under: {args.data_root}")

    if args.limit > 0:
        cases = cases[: args.limit]

    centers = Counter(patient_id.split("-")[0] for patient_id, _, _, _ in cases)
    print(f"[found] {len(cases)} cases")
    print(f"[centers] {dict(centers)}")

    report_fov(cases)

    if args.fov_only:
        return

    dataset_name = DATASET_NAME_TEMPLATE.format(dataset_id=args.dataset_id)
    dataset_dir = Path(args.nnunet_raw) / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    print(f"\n[build] {dataset_dir}")
    print(f"[mode]  {'symlink' if args.link else 'copy'}")

    present_patient_ids = set()

    for idx, (patient_id, ct_path, pet_path, label_path) in enumerate(cases, start=1):
        place_file(ct_path, str(images_tr / f"{patient_id}_0000.nii.gz"), args.link)
        place_file(pet_path, str(images_tr / f"{patient_id}_0001.nii.gz"), args.link)
        place_file(label_path, str(labels_tr / f"{patient_id}.nii.gz"), args.link)

        present_patient_ids.add(patient_id)

        if idx % 100 == 0 or idx == len(cases):
            print(f"  ...{idx}/{len(cases)}")

    write_dataset_json(dataset_dir, len(cases))

    if args.splits or args.smoke_split:
        if args.smoke_split:
            nnunet_splits = convert_splits(
                splits_path="",
                present_patient_ids=present_patient_ids,
                smoke_split=True,
            )
        else:
            if args.splits is None or not Path(args.splits).exists():
                raise FileNotFoundError(f"Splits file not found: {args.splits}")

            nnunet_splits = convert_splits(
                splits_path=args.splits,
                present_patient_ids=present_patient_ids,
                smoke_split=False,
            )

        splits_out = dataset_dir / "splits_final.json"
        with open(splits_out, "w", encoding="utf-8") as file:
            json.dump(nnunet_splits, file, indent=2)

        print(f"[write] {splits_out}")
        print(f"[splits] {len(nnunet_splits)} fold(s)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
    
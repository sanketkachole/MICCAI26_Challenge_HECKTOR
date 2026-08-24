#!/usr/bin/env python3
"""
HECKTOR 2026 - head-and-neck crop utilities (v2, with size gate).

Why this exists
---------------
Some test CT scans are near-whole-body (up to 1330 mm). nnU-Net resamples
everything to 1.5 x 0.977 x 0.977 mm, so those images become 4-6x more voxels
than a typical head-and-neck case. nnU-Net then holds several full-size copies
of the logits at once (fold accumulation, then a pickled hand-off to the export
worker, then the resample back to the original grid). That is what exceeded the
container's 32 GB of CPU RAM.

This module crops CT and PET to a slab at the top of the scan before nnU-Net
sees them, and pastes the predicted mask back onto the original CT grid.
The model, the weights and the 5 folds are untouched.

Slab size
---------
Measured from the 782 training ground-truth masks (scripts/50_...):
the lowest disease voxel is at most 330.3 mm below the top of the CT.
SLAB_MM = 400 leaves ~70 mm of margin. Validated on the 8 largest scans in the
dataset (scripts/51_...): zero ground-truth voxels lost.

Size gate
---------
Measured Dice effect of the crop (scripts/51_...):
    large cases    GTVp -0.0055  GTVn +0.0069   (patches 3.6-5.3x fewer)
    typical cases  GTVp -0.0189  GTVn -0.0066   (patches only 1.5x fewer)
So we crop only when the image is big enough to be a memory risk. Normal cases
go through untouched, exactly as in the container that passed the Sanity Check.

Anchor
------
We keep everything from the top of the image down to 400 mm below the top of
the BODY. When there is no air above the head these are the same thing. When
there is air (or a raised arm) above the head, the slab automatically extends
so the head and neck are still fully covered.
"""

import numpy as np
import SimpleITK as sitk

SLAB_MM = 400.0          # how far below the top of the body we keep
XY_MARGIN_MM = 25.0      # margin around the body in x and y
BODY_HU = -500           # everything above this is "not air"

# nnUNetPlans 3d_fullres target spacing, in SimpleITK (x, y, z) order
TARGET_SPACING = (0.9765620231628418, 0.9765620231628418, 1.5)

# Crop only if the image would resample to more than this many voxels.
# For reference: the median training case is 73 M voxels and is safe.
GATE_VOXELS = 120e6


def estimate_resampled_voxels(img):
    """How many voxels will this image have after nnU-Net resamples it?"""
    size = img.GetSize()
    sp = img.GetSpacing()
    n = 1.0
    for i in range(3):
        n *= size[i] * sp[i] / TARGET_SPACING[i]
    return float(n)


def _phys_z_of_k(img, k):
    """Physical z of voxel index (0, 0, k)."""
    return float(img.TransformIndexToPhysicalPoint((0, 0, int(k)))[2])


def compute_crop_box(ct_img, slab_mm=SLAB_MM, xy_margin_mm=XY_MARGIN_MM):
    """
    Work out the crop from the CT alone.

    Returns (index_xyz, size_xyz), both in SimpleITK (x, y, z) order, ready for
    sitk.RegionOfInterest.
    """
    arr = sitk.GetArrayViewFromImage(ct_img)          # axes are z, y, x
    nz, ny, nx = arr.shape
    spacing = ct_img.GetSpacing()                     # (x, y, z)

    body_z = (arr > BODY_HU).any(axis=(1, 2))         # slices containing tissue
    if not body_z.any():
        return (0, 0, 0), (nx, ny, nz)                # nothing found, keep all

    # which end of the z axis is superior?
    z_at_0 = _phys_z_of_k(ct_img, 0)
    z_at_last = _phys_z_of_k(ct_img, nz - 1)
    head_at_high_index = z_at_last > z_at_0

    k_body = np.where(body_z)[0]
    k_body_top = int(k_body.max()) if head_at_high_index else int(k_body.min())
    z_body_top = _phys_z_of_k(ct_img, k_body_top)
    z_floor = z_body_top - slab_mm                    # nothing below this is kept

    step = (z_at_last - z_at_0) / max(nz - 1, 1)
    z_of_k = z_at_0 + step * np.arange(nz)
    keep = z_of_k >= z_floor
    if not keep.any():                                # should not happen
        return (0, 0, 0), (nx, ny, nz)
    k0, k1 = int(np.argmax(keep)), int(nz - 1 - np.argmax(keep[::-1]))

    # in-plane bounding box of the body, computed on the retained slab only
    slab = arr[k0:k1 + 1] > BODY_HU
    if slab.any():
        ys = np.where(slab.any(axis=(0, 2)))[0]
        xs = np.where(slab.any(axis=(0, 1)))[0]
        my = int(round(xy_margin_mm / spacing[1]))
        mx = int(round(xy_margin_mm / spacing[0]))
        y0 = max(0, int(ys.min()) - my)
        y1 = min(ny - 1, int(ys.max()) + my)
        x0 = max(0, int(xs.min()) - mx)
        x1 = min(nx - 1, int(xs.max()) + mx)
    else:
        x0, x1, y0, y1 = 0, nx - 1, 0, ny - 1

    return (int(x0), int(y0), int(k0)), \
           (int(x1 - x0 + 1), int(y1 - y0 + 1), int(k1 - k0 + 1))


def prepare_inputs(ct_img, pet_img, slab_mm=SLAB_MM, xy_margin_mm=XY_MARGIN_MM,
                   gate_voxels=GATE_VOXELS):
    """
    Produce the CT and PET that go to nnU-Net.

    PET is always resampled onto the CT grid (nnU-Net requires both channels on
    the same grid). If the image is large it is cropped FIRST, so we never build
    a full-size PET volume for a whole-body scan.

    Returns (ct_out, pet_out, index_xyz, info_string).
    Pass index_xyz to paste_back() afterwards.
    """
    try:
        est = estimate_resampled_voxels(ct_img)
    except Exception:
        est = float("inf")        # fail safe: if we cannot measure it, crop it

    if est <= gate_voxels:
        pet_out = sitk.Resample(pet_img, ct_img, sitk.Transform(),
                                sitk.sitkLinear, 0.0, pet_img.GetPixelID())
        info = (f"CROP: no  (estimated {est/1e6:.0f} M voxels after resampling, "
                f"gate {gate_voxels/1e6:.0f} M)")
        return ct_img, pet_out, (0, 0, 0), info

    index_xyz, size_xyz = compute_crop_box(ct_img, slab_mm, xy_margin_mm)
    ct_out = sitk.RegionOfInterest(ct_img, size_xyz, index_xyz)
    pet_out = sitk.Resample(pet_img, ct_out, sitk.Transform(),
                            sitk.sitkLinear, 0.0, pet_img.GetPixelID())

    nz, ny, nx = sitk.GetArrayViewFromImage(ct_img).shape
    before = nx * ny * nz
    after = size_xyz[0] * size_xyz[1] * size_xyz[2]
    sp = ct_img.GetSpacing()
    info = (f"CROP: yes  {nx}x{ny}x{nz} -> "
            f"{size_xyz[0]}x{size_xyz[1]}x{size_xyz[2]}  "
            f"({before/1e6:.0f}M -> {after/1e6:.0f}M voxels, "
            f"{before/max(after,1):.1f}x smaller; slab {size_xyz[2]*sp[2]:.0f} mm; "
            f"estimated {est/1e6:.0f} M after resampling)")
    return ct_out, pet_out, index_xyz, info


def paste_back(mask_small_img, ct_full_img, index_xyz):
    """
    Put the prediction back onto the full original CT grid.

    The challenge requires the output mask to match the original CT resolution,
    origin and spacing, so we copy that information from the full CT.
    Safe to call even when no crop happened (index is then (0, 0, 0)).
    """
    nz, ny, nx = sitk.GetArrayViewFromImage(ct_full_img).shape
    small = sitk.GetArrayFromImage(mask_small_img).astype(np.uint8)
    dz, dy, dx = small.shape

    if (dz, dy, dx) == (nz, ny, nx) and tuple(index_xyz) == (0, 0, 0):
        out = sitk.GetImageFromArray(small)
        out.CopyInformation(ct_full_img)
        return out

    full = np.zeros((nz, ny, nx), dtype=np.uint8)
    x0, y0, z0 = index_xyz
    full[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx] = small
    out = sitk.GetImageFromArray(full)
    out.CopyInformation(ct_full_img)
    return out


# kept so scripts/51_crop_check.py still works (it always crops, by design)
def crop_pair(ct_img, pet_img, slab_mm=SLAB_MM, xy_margin_mm=XY_MARGIN_MM):
    index_xyz, size_xyz = compute_crop_box(ct_img, slab_mm, xy_margin_mm)
    ct_c = sitk.RegionOfInterest(ct_img, size_xyz, index_xyz)
    pet_c = sitk.Resample(pet_img, ct_c, sitk.Transform(),
                          sitk.sitkLinear, 0.0, pet_img.GetPixelID())
    return ct_c, pet_c, index_xyz, size_xyz


def describe(ct_img, index_xyz, size_xyz):
    nz, ny, nx = sitk.GetArrayViewFromImage(ct_img).shape
    before = nx * ny * nz
    after = size_xyz[0] * size_xyz[1] * size_xyz[2]
    sp = ct_img.GetSpacing()
    return (f"{nx}x{ny}x{nz} -> {size_xyz[0]}x{size_xyz[1]}x{size_xyz[2]}  "
            f"({before/1e6:.0f}M -> {after/1e6:.0f}M voxels, "
            f"{before/max(after,1):.1f}x smaller; slab {size_xyz[2]*sp[2]:.0f} mm)")

"""
HECKTOR 2026 - Inference entry point (Grand Challenge container).
Reads /input/images/ct, /input/images/pet, /input/ehr.json ; writes the 4 outputs.
Pipeline: nnU-Net 5-fold ensemble -> node clean-up -> geometry
          -> N-rule + T-model -> Cox risk (written as -risk so it is RFS-like).

Model files under /opt/ml/model:
  nnunet_results/Dataset501_HECKTOR/...            (the trained folds)
  staging_config.json, staging_Tmodel.joblib
  prognosis_model.joblib

MEMORY FIX (2026-07): some test CTs are near-whole-body (up to 1330 mm).
nnU-Net resamples everything to 1.5 x 0.977 x 0.977 mm, so those images become
4-6x more voxels than a typical head-and-neck case, and nnU-Net holds several
full-size copies of the logits at once. That exceeded the 32 GB RAM limit.
run_segmentation() now crops large images to a 400 mm head-and-neck slab before
nnU-Net sees them, then pastes the mask back onto the original CT grid.
Small images are left completely untouched. See the CROP section below.
"""
import json, os, re, subprocess, sys, tempfile
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
import joblib

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
NNUNET_RESULTS = MODEL_PATH / "nnunet_results"

MIDLINE_DEADZONE_MM = 6.0
N_ORDINAL = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}


# ================================================== HEAD-AND-NECK CROP (NEW) ==
# Slab size: measured on all 782 training ground-truth masks. The lowest disease
# voxel is at most 330.3 mm below the top of the CT, so 400 mm leaves ~70 mm of
# margin. Validated on the 8 largest scans in the dataset: 0 voxels lost, and
# Dice unchanged (GTVn +0.007, GTVp -0.006 on those cases).
#
# SAFETY RULE: if anything at all is unusual (2D input, odd geometry, an error),
# we DO NOT crop and the container behaves exactly as it did before.
SLAB_MM = 400.0
XY_MARGIN_MM = 25.0
BODY_HU = -500
TARGET_SPACING = (0.9765620231628418, 0.9765620231628418, 1.5)  # nnUNetPlans 3d_fullres, (x,y,z)
# Crop only above this many voxels-after-resampling. Median training case = 73 M.
GATE_VOXELS = 120e6


def estimate_resampled_voxels(img):
    """How many voxels will this image have after nnU-Net resamples it?"""
    size = img.GetSize()
    sp = img.GetSpacing()
    if len(size) < 3 or len(sp) < 3:
        raise ValueError("image is %dD, not a 3D volume" % len(size))
    n = 1.0
    for i in range(3):
        n *= size[i] * sp[i] / TARGET_SPACING[i]
    return float(n)


def _phys_z_of_k(img, k):
    return float(img.TransformIndexToPhysicalPoint((0, 0, int(k)))[2])


def compute_crop_box(ct_img, slab_mm=SLAB_MM, xy_margin_mm=XY_MARGIN_MM):
    """Return (index_xyz, size_xyz) for sitk.RegionOfInterest, from the CT alone."""
    arr = sitk.GetArrayViewFromImage(ct_img)          # axes are z, y, x
    if arr.ndim != 3:
        raise ValueError("CT array is %dD, not 3D" % arr.ndim)
    nz, ny, nx = arr.shape
    spacing = ct_img.GetSpacing()                     # (x, y, z)

    body_z = (arr > BODY_HU).any(axis=(1, 2))
    if not body_z.any():
        return (0, 0, 0), (nx, ny, nz)

    z_at_0 = _phys_z_of_k(ct_img, 0)
    z_at_last = _phys_z_of_k(ct_img, nz - 1)
    head_at_high_index = z_at_last > z_at_0

    k_body = np.where(body_z)[0]
    k_body_top = int(k_body.max()) if head_at_high_index else int(k_body.min())
    z_floor = _phys_z_of_k(ct_img, k_body_top) - slab_mm

    step = (z_at_last - z_at_0) / max(nz - 1, 1)
    z_of_k = z_at_0 + step * np.arange(nz)
    keep = z_of_k >= z_floor
    if not keep.any():
        return (0, 0, 0), (nx, ny, nz)
    k0, k1 = int(np.argmax(keep)), int(nz - 1 - np.argmax(keep[::-1]))

    slab = arr[k0:k1 + 1] > BODY_HU
    if slab.any():
        ys = np.where(slab.any(axis=(0, 2)))[0]
        xs = np.where(slab.any(axis=(0, 1)))[0]
        my = int(round(xy_margin_mm / spacing[1]))
        mx = int(round(xy_margin_mm / spacing[0]))
        y0 = max(0, int(ys.min()) - my); y1 = min(ny - 1, int(ys.max()) + my)
        x0 = max(0, int(xs.min()) - mx); x1 = min(nx - 1, int(xs.max()) + mx)
    else:
        x0, x1, y0, y1 = 0, nx - 1, 0, ny - 1

    return (int(x0), int(y0), int(k0)), \
           (int(x1 - x0 + 1), int(y1 - y0 + 1), int(k1 - k0 + 1))


def prepare_nnunet_inputs(ct_img, pet_img):
    """
    Return (ct_out, pet_out, crop_index, info).

    PET is always put on the CT grid (nnU-Net needs both channels on one grid).
    crop_index is None when NOTHING was cropped -- in that case the caller must
    return the nnU-Net output unchanged, exactly like the old container.
    Large images are cropped FIRST, so we never build a full-size PET volume for
    a whole-body scan.
    """
    def _no_crop(reason):
        pet_out = sitk.Resample(pet_img, ct_img, sitk.Transform(),
                                sitk.sitkLinear, 0.0, pet_img.GetPixelID())
        return ct_img, pet_out, None, "CROP: no  (%s)" % reason

    # 1. only ordinary 3D volumes can be cropped in z
    try:
        if ct_img.GetDimension() != 3 or pet_img.GetDimension() != 3:
            return _no_crop("input is %dD, nothing to crop" % ct_img.GetDimension())
    except Exception as e:
        return _no_crop("could not read image dimension (%s)" % e)

    # 2. how big will nnU-Net make this?
    try:
        est = estimate_resampled_voxels(ct_img)
    except Exception as e:
        return _no_crop("could not estimate size (%s)" % e)

    if est <= GATE_VOXELS:
        return _no_crop("est %.0f M voxels after resampling, gate %.0f M"
                        % (est / 1e6, GATE_VOXELS / 1e6))

    # 3. big image -> crop
    try:
        index_xyz, size_xyz = compute_crop_box(ct_img)
        ct_out = sitk.RegionOfInterest(ct_img, size_xyz, index_xyz)
        pet_out = sitk.Resample(pet_img, ct_out, sitk.Transform(),
                                sitk.sitkLinear, 0.0, pet_img.GetPixelID())
    except Exception as e:
        print("[warn] crop failed (%s); running on the full image" % e, flush=True)
        return _no_crop("crop failed, est was %.0f M voxels" % (est / 1e6))

    nx, ny, nz = ct_img.GetSize()
    before = float(nx) * ny * nz
    after = float(size_xyz[0]) * size_xyz[1] * size_xyz[2]
    info = ("CROP: yes  %dx%dx%d -> %dx%dx%d  (%.0fM -> %.0fM voxels, %.1fx smaller; "
            "slab %.0f mm; est %.0f M after resampling)" % (
                nx, ny, nz, size_xyz[0], size_xyz[1], size_xyz[2],
                before / 1e6, after / 1e6, before / max(after, 1.0),
                size_xyz[2] * ct_img.GetSpacing()[2], est / 1e6))
    return ct_out, pet_out, index_xyz, info


def paste_back(mask_small_img, ct_full_img, crop_index):
    """Put the cropped prediction back onto the full original CT grid."""
    nx, ny, nz = ct_full_img.GetSize()
    small = sitk.GetArrayFromImage(mask_small_img).astype(np.uint8)
    if small.ndim == 2:
        small = small[None, ...]
    dz, dy, dx = small.shape

    full = np.zeros((nz, ny, nx), dtype=np.uint8)
    x0, y0, z0 = crop_index
    full[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx] = small
    out = sitk.GetImageFromArray(full)
    out.CopyInformation(ct_full_img)
    return out
# ============================================== END HEAD-AND-NECK CROP =======


def _try_load(path):
    """Load an optional model file; return None if missing or unreadable."""
    try:
        if path.exists():
            return joblib.load(path)
        print("[info] optional model not found: %s" % path.name, flush=True)
    except Exception as e:
        print("[warn] could not load %s: %s" % (path.name, e), flush=True)
    return None

# =============================================================== main pipeline
def _run_pipeline():
    ct_path = get_image_file(INPUT_PATH / "images/ct")
    pet_path = get_image_file(INPUT_PATH / "images/pet")
    ehr = load_json_tolerant(INPUT_PATH / "ehr.json")

    staging_cfg = json.load(open(MODEL_PATH / "staging_config.json"))
    Tclf = joblib.load(MODEL_PATH / "staging_Tmodel.joblib")
    prog = joblib.load(MODEL_PATH / "prognosis_model.joblib")

    # --- Slot 2: learned staging models. OPTIONAL: if either file is missing the
    # container still runs using the rule (N) and the old HistGB model (T). ---
    Nclf = _try_load(MODEL_PATH / "n_classifier.joblib")
    Tclf2 = _try_load(MODEL_PATH / "t_classifier.joblib")

    ct_img = sitk.ReadImage(ct_path)
    pet_img = sitk.ReadImage(pet_path)

    # ---- 1. segmentation (nnU-Net 5-fold ensemble) ----
    pred_img = run_segmentation(ct_path, pet_path)
    larr = sitk.GetArrayFromImage(pred_img)
    parr = sitk.GetArrayFromImage(pet_img)
    if parr.shape != larr.shape:                       # align PET to mask grid if needed
        parr = sitk.GetArrayFromImage(sitk.Resample(pet_img, pred_img, sitk.Transform(),
                                                     sitk.sitkLinear, 0.0, pet_img.GetPixelID()))
    if larr.ndim == 2:
        larr = larr[None, ...]
    if parr.ndim == 2:
        parr = parr[None, ...]

    # geometry reference = CT (always a real volume); pad spacing to 3D just in case
    sp = list(ct_img.GetSpacing())
    if len(sp) < 3:
        sp = sp + [1.0] * (3 - len(sp))
    sx, sy, sz = float(sp[0]), float(sp[1]), float(sp[2])
    spacing_zyx = (sz, sy, sx)

    # ---- 2. node clean-up ----
    cleaned = cleanup_nodes(larr, parr, spacing_zyx)

    # write segmentation on the CT reference (original geometry)
    write_segmentation(OUTPUT_PATH / "images/head-neck-tumor-segmentation", cleaned, ct_path)

    # ---- 3. geometry features (laterality in voxel space -> no ITK transform needed) ----
    X = larr.shape[2]
    midline = (X / 2.0) * sx
    def phys_x(iz, iy, ix):
        return ix * sx
    g = analyze_gtvn(cleaned, spacing_zyx, phys_x, midline)
    gtvp_ml = float((cleaned == 1).sum()) * (sx * sy * sz) / 1000.0
    geom = {"gtvp_ml": gtvp_ml, "n_gtvn": g["n_nodes"], "max_node_mm": g["max_node_mm"],
            "gtvn_total_ml": g["total_vol_ml"], "laterality": g["laterality"]}

    # ---- 4. N + T staging ----
    # The RULE N is always computed: the prognosis (Cox) model was trained with it
    # as an input feature, so we keep feeding prognosis exactly what it expects.
    nr = staging_cfg["N_rule"]
    n_stage_rule = rule_N(g["n_nodes"], g["max_node_mm"], g["laterality"],
                          nr["N3_MM"], nr["N1_MAX_MM"])

    # staging row uses SHORT clinical names; map ehr long keys -> short
    t_src = staging_cfg.get("T_clinical_source", {})  # {short: real_csv_name}
    staging_row = dict(geom)
    for short, real in t_src.items():
        staging_row[short] = ehr.get(real if real else short)

    # --- N OUTPUT: learned size-focused classifier (falls back to the rule) ---
    n_stage = n_stage_rule
    if Nclf is not None:
        try:
            row = {"max_node_mm": float(geom["max_node_mm"]),
                   "gtvn_total_ml": float(geom["gtvn_total_ml"]),
                   "gtvp_ml": float(geom["gtvp_ml"])}
            for lv in Nclf["laterality_levels"]:
                row["lat_" + lv] = 1.0 if geom["laterality"] == lv else 0.0
            Xn = np.array([[row.get(f, np.nan) for f in Nclf["features"]]], dtype=float)
            n_stage = str(Nclf["model"].predict(Xn)[0])
        except Exception as e:
            print("[warn] N classifier failed, using rule: %s" % e, flush=True)
            n_stage = n_stage_rule

    # --- T OUTPUT: clean-label classifier (falls back to the old HistGB model) ---
    t_stage = None
    if Tclf2 is not None:
        try:
            row = {}
            for gf in Tclf2["geom_features"]:
                row[gf] = float(geom[gf])
            for short in Tclf2["clinical_features"]:
                longname = Tclf2.get("clinical_long_names", {}).get(short, short)
                v = ehr.get(longname, staging_row.get(short))
                try:
                    row[short] = float(v)
                except (TypeError, ValueError):
                    row[short] = np.nan
            Xt2 = np.array([[row.get(f, np.nan) for f in Tclf2["features"]]], dtype=float)
            raw = str(Tclf2["model"].predict(Xt2)[0])
            t_stage = raw if raw.upper().startswith("T") else ("T" + raw)
        except Exception as e:
            print("[warn] T classifier failed, using old model: %s" % e, flush=True)
            t_stage = None
    if t_stage is None:
        Xt = apply_encoders(staging_row, staging_cfg["T_features"], staging_cfg.get("T_encoders", {}))
        t_stage = str(Tclf.predict(Xt)[0])

    write_json(OUTPUT_PATH / "t-stage.json", t_stage)
    write_json(OUTPUT_PATH / "n-stage.json", n_stage)

    # ---- 5. prognosis (write -risk so it is RFS-time-like / anti-concordant with risk) ----
    # prognosis reads clinical by their real (long) names via clinical_colmap, so pass ehr as-is
    # NOTE: feed the RULE's N (not the classifier's) - this is what Cox was trained on
    prog_row = dict(geom); prog_row.update(ehr); prog_row["rule_N"] = n_stage_rule
    Xp = prog_features(prog_row, prog["clinical_colmap"], prog["feature_spec"])
    risk = predict_risk_from_state(prog["state"], Xp)
    write_json(OUTPUT_PATH / "rfs.json", float(-risk))
    return 0


def _default_segmentation(ct_path):
    """Write an all-background mask on the CT grid (used if segmentation fails)."""
    ref = sitk.ReadImage(ct_path)
    arr = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
    write_segmentation(OUTPUT_PATH / "images/head-neck-tumor-segmentation", arr, ct_path)


def _ensure_all_outputs():
    """Guarantee all 4 outputs exist so the job never fails on a missing file."""
    try:
        ct_path = get_image_file(INPUT_PATH / "images/ct")
    except Exception:
        ct_path = None
    seg = OUTPUT_PATH / "images/head-neck-tumor-segmentation" / "output.mha"
    if not seg.exists() and ct_path is not None:
        try:
            _default_segmentation(ct_path)
        except Exception:
            pass
    if not (OUTPUT_PATH / "t-stage.json").exists():
        write_json(OUTPUT_PATH / "t-stage.json", "T2")
    if not (OUTPUT_PATH / "n-stage.json").exists():
        write_json(OUTPUT_PATH / "n-stage.json", "N0")
    if not (OUTPUT_PATH / "rfs.json").exists():
        write_json(OUTPUT_PATH / "rfs.json", 0.0)


def run():
    """Top-level entry: run the pipeline, but never crash -> always emit valid outputs."""
    try:
        _run_pipeline()
    except Exception as e:
        print(f"PIPELINE ERROR (writing safe defaults): {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        _ensure_all_outputs()
    return 0


# =============================================================== segmentation
def run_segmentation(ct_path, pet_path):
    """
    Run the 5-fold nnU-Net ensemble and return the mask ON THE ORIGINAL CT GRID.

    Large (near-whole-body) images are cropped to a 400 mm head-and-neck slab
    first, then the mask is pasted back. Everything downstream is unaffected.
    """
    pid = "case"
    work = Path(tempfile.mkdtemp())
    ind = work / "nnin"; outd = work / "nnout"
    ind.mkdir(parents=True); outd.mkdir(parents=True)

    ct_full = sitk.ReadImage(ct_path)
    pet = sitk.ReadImage(pet_path)

    # nnU-Net requires BOTH channels on the exact same grid, and a whole-body
    # image must be cropped or it blows the 32 GB RAM limit. Both happen here.
    ct_in, pet_in, crop_index, crop_info = prepare_nnunet_inputs(ct_full, pet)
    print("[seg] " + crop_info, flush=True)
    del pet

    # nnU-Net expects CASE_0000 (CT) and CASE_0001 (PET) as .nii.gz
    sitk.WriteImage(ct_in,  str(ind / f"{pid}_0000.nii.gz"))
    sitk.WriteImage(pet_in, str(ind / f"{pid}_0001.nii.gz"))
    del ct_in, pet_in

    env = dict(os.environ,
               nnUNet_results=str(NNUNET_RESULTS),
               nnUNet_raw=str(work / "raw"),
               nnUNet_preprocessed=str(work / "prep"))
    cmd = ["nnUNetv2_predict", "-i", str(ind), "-o", str(outd),
           "-d", "Dataset501_HECKTOR", "-c", "3d_fullres",
           "-f", "0", "1", "2", "3", "4",
           "-tr", "nnUNetTrainer_250epochs", "-p", "nnUNetPlans",
           "--disable_tta",          # no test-time mirroring -> much faster, tiny accuracy cost
           "-npp", "1", "-nps", "1"]  # single worker -> lower RAM
    subprocess.run(cmd, check=True, env=env)

    pred_small = sitk.ReadImage(str(outd / f"{pid}.nii.gz"))
    if crop_index is None:
        return pred_small          # nothing was cropped -> old behaviour exactly
    return paste_back(pred_small, ct_full, crop_index)


# =============================================================== pure helpers
# Winning settings from the BigRed sweep: balanced accuracy 0.657 on predicted masks.
MERGE_MM = 2.0
MIN_ML = 1.5
SUV_THR = 3.0
DIAM_MM = 8.0

def cleanup_nodes(label_arr, pet_arr, spacing_zyx, suv_thr=SUV_THR, diam_mm=DIAM_MM,
                  min_ml=MIN_ML, merge_mm=MERGE_MM):
    """Remove false-positive GTVn components. Keep/discard is decided on the MERGED
    group (a real node split by segmentation noise is judged as one node), but the
    output mask only ever removes/keeps ORIGINAL voxels -- never adds bridge voxels,
    so Dice is unaffected. Closing is restricted to a small box around the nodes so
    this stays fast even on a full-body scan."""
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
        lab, n = ndi.label(g, structure=np.ones((3, 3, 3), int))
        drop_mask_c = np.zeros_like(gtvn_c)
        for c in range(1, n + 1):
            comp = lab == c
            idx = np.argwhere(comp); nvox = idx.shape[0]
            vol_ml = nvox * voxvol_ml
            maxdim = float(((idx.max(0) - idx.min(0) + 1) * np.array([sz, sy, sx])).max())
            suvmax = float(pet_c[comp].max()) if pet_c is not None else 0.0
            keep = (vol_ml >= min_ml) and (suvmax >= suv_thr) and (maxdim >= diam_mm)
            if not keep:
                drop_mask_c |= (comp & gtvn_c)
        full_drop = np.zeros_like(gtvn)
        full_drop[sl] = drop_mask_c
        out[full_drop] = 0
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


def analyze_gtvn(label_arr, spacing_zyx, phys_x_of_index, midline_x, merge_mm=MERGE_MM):
    """Count/size nodes the SAME way the retrained N-rule and T-model expect: using
    the merged grouping (a split real node counts as one), cropped for speed."""
    sz, sy, sx = spacing_zyx
    voxvol_ml = (sz * sy * sx) / 1000.0
    gtvn = (label_arr == 2)
    if not gtvn.any():
        return dict(n_nodes=0, max_node_mm=0.0, total_vol_ml=0.0, laterality="none")
    idx0 = np.argwhere(gtvn)
    pad = max(1, int(np.ceil(merge_mm / min(sz, sy, sx)))) + 1
    lo = np.maximum(idx0.min(0) - pad, 0)
    hi = np.minimum(idx0.max(0) + pad, np.array(gtvn.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    gtvn_c = gtvn[sl]
    z0, y0, x0 = lo
    g = gtvn_c
    if merge_mm > 0:
        r = (max(1,int(round(merge_mm/sz))), max(1,int(round(merge_mm/sy))), max(1,int(round(merge_mm/sx))))
        g = ndi.binary_closing(gtvn_c, structure=np.ones((r[0]*2+1,r[1]*2+1,r[2]*2+1), int))
    lab, n = ndi.label(g, structure=np.ones((3, 3, 3), int))
    max_mm, tot_ml, sides = 0.0, 0.0, set()
    for c in range(1, n + 1):
        idx = np.argwhere(lab == c)
        max_mm = max(max_mm, float(((idx.max(0) - idx.min(0) + 1) * np.array([sz, sy, sx])).max()))
        tot_ml += idx.shape[0] * voxvol_ml
        cen = idx.mean(0)
        px = phys_x_of_index(cen[0] + z0, cen[1] + y0, cen[2] + x0)
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
    vals = []
    for c in cols:
        v = row.get(c, np.nan)
        if c in enc and enc[c]:
            cats = enc[c]
            vals.append(float(cats.index(str(v))) if str(v) in cats else np.nan)
        else:
            try: vals.append(float(v))
            except (TypeError, ValueError): vals.append(np.nan)
    return np.array(vals, float).reshape(1, -1)


def prog_features(row, cmap, spec):
    import pandas as pd
    X = {}
    for gcol in spec["GEOM"]:
        val = row.get(gcol, None)
        X[gcol] = float(val) if val is not None else np.nan
    rn = str(row.get("rule_N", "")).upper()[:2]
    X["N_ordinal"] = spec["N_ORDINAL"].get(rn, np.nan)
    age = row.get(cmap.get("Age"))
    X["Age"] = float(age) if age is not None else np.nan
    for c in spec["CAT_CLIN"]:
        X[c] = row.get(cmap.get(c), np.nan)
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


# =============================================================== I/O utilities
def get_image_file(location):
    files = (glob(str(location / "*.mha")) + glob(str(location / "*.nii.gz"))
             + glob(str(location / "*.tif")) + glob(str(location / "*.tiff")))
    if not files:
        raise FileNotFoundError(f"No image file found in {location}")
    return files[0]


def load_json_tolerant(location):
    """Load JSON; tolerate a trailing comma before } or ] (seen in the example)."""
    text = Path(location).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(fixed)


def write_json(location, data):
    location.parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(data, f, indent=2)


def write_segmentation(location, array, reference_path):
    location.mkdir(parents=True, exist_ok=True)
    reference = sitk.ReadImage(reference_path)
    arr = array
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # match the reference image dimension (handles single-slice / 2D cases)
    if reference.GetDimension() == 2 and arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif reference.GetDimension() == 3 and arr.ndim == 2:
        arr = arr[None, ...]
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(reference)
    sitk.WriteImage(img, str(location / "output.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run())

#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 09 (memory-safe): tune node post-processing for N balanced acc.
Processes each OOF predicted mask ONCE, keeps only a tiny per-node summary
(volume, max size, SUVmax, x-position) for a few merge settings, frees the big
arrays, then sweeps min-volume / SUV / merge / rule-cutoffs over the summaries.
Low memory, fast, with flushed progress.
"""
import argparse, glob, os, re, itertools, sys
import numpy as np, pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi
from sklearn.metrics import balanced_accuracy_score

DZ = 6.0                      # midline deadzone (mm)
MERGE_GRID = [0, 2, 4, 6]      # closing radius (mm) to merge split pieces
MODE_GRID = ["either", "both"] # "either": keep if SUV-high OR big-enough (challenge's own rule)
                               # "both":   keep only if SUV-high AND big-enough (stricter)

def summarize_case(mpath, ct_path, pet_path):
    """Return {merge_mm: [(vol_ml,max_mm,suvmax,cx_mm), ...]} and body-midline mm."""
    m = sitk.ReadImage(mpath); marr = sitk.GetArrayFromImage(m).astype(np.uint8)
    sp = m.GetSpacing(); sz, sy, sx = sp[2], sp[1], sp[0]
    voxvol_ml = (sz*sy*sx)/1000.0
    # PET on mask grid
    if pet_path:
        pet = sitk.ReadImage(pet_path)
        if pet.GetSize()!=m.GetSize() or not np.allclose(pet.GetSpacing(), sp, atol=1e-3):
            pet = sitk.Resample(pet, m, sitk.Transform(), sitk.sitkLinear, 0.0, pet.GetPixelID())
        parr = sitk.GetArrayFromImage(pet).astype(np.float32)
    else:
        parr = None
    # body-centroid midline from CT: use a DOWNSAMPLED, vectorized marginal-sum
    # computation (avoids materializing tens of millions of index rows via
    # argwhere on a full-resolution full-body CT, which is what was hanging).
    if ct_path:
        ct = sitk.ReadImage(ct_path)
        if ct.GetSize()!=m.GetSize() or not np.allclose(ct.GetSpacing(), sp, atol=1e-3):
            ct = sitk.Resample(ct, m, sitk.Transform(), sitk.sitkLinear, -1000.0, ct.GetPixelID())
        ctarr = sitk.GetArrayFromImage(ct)
        step = max(1, ctarr.shape[0]//64), max(1, ctarr.shape[1]//128), max(1, ctarr.shape[2]//128)
        small = ctarr[::step[0], ::step[1], ::step[2]]
        body = small > -500
        counts_per_x = body.sum(axis=(0, 1)).astype(np.float64)   # 1D, length = small.shape[2]
        if counts_per_x.sum() > 0:
            x_idx_small = np.arange(small.shape[2])
            mean_idx_small = float((x_idx_small * counts_per_x).sum() / counts_per_x.sum())
            mid = mean_idx_small * step[2] * sx
        else:
            mid = (marr.shape[2]/2.0)*sx
        del ctarr, ct, small, body
    else:
        mid = (marr.shape[2]/2.0)*sx

    gtvn0 = (marr == 2)
    out = {}
    if not gtvn0.any():
        for merge_mm in MERGE_GRID:
            out[merge_mm] = []
        return out, mid

    # Restrict all morphology to a padded bounding box around the nodes, not the
    # whole volume -> binary_closing cost depends on the lesion size, not image size.
    idx0 = np.argwhere(gtvn0)
    lo = idx0.min(0); hi = idx0.max(0) + 1
    max_merge_vox = max(1, int(np.ceil(max(MERGE_GRID) / min(sz, sy, sx)))) + 1
    pad = np.array([max_merge_vox]*3)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.array(marr.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    gtvn_crop = gtvn0[sl]
    parr_crop = parr[sl] if parr is not None else None
    z0, y0, x0 = lo  # to translate cropped voxel coords back to full-volume x (mm) for laterality

    for merge_mm in MERGE_GRID:
        g = gtvn_crop
        if merge_mm > 0:
            r = (max(1,int(round(merge_mm/sz))), max(1,int(round(merge_mm/sy))), max(1,int(round(merge_mm/sx))))
            st = np.ones((r[0]*2+1, r[1]*2+1, r[2]*2+1), int)
            g = ndi.binary_closing(gtvn_crop, structure=st)
        comps = []
        if g.any():
            lab, n = ndi.label(g, structure=np.ones((3,3,3), int))
            for c in range(1, n+1):
                idx = np.argwhere(lab == c); nvox = idx.shape[0]
                maxmm = float(((idx.max(0)-idx.min(0)+1)*np.array([sz,sy,sx])).max())
                suvmax = float(parr_crop[lab==c].max()) if parr_crop is not None else 999.0
                cx = float((idx[:,2].mean() + x0))*sx     # translate back to full-volume x index
                comps.append((nvox*voxvol_ml, maxmm, suvmax, cx))
        out[merge_mm] = comps
    del marr, parr, gtvn0, gtvn_crop, parr_crop
    return out, mid

def features(comps, mid, min_ml, suv_thr, diam_mm, mode):
    nodes = []
    for vol, maxmm, suvmax, cx in comps:
        if vol < min_ml: continue
        big_enough = maxmm >= diam_mm
        suv_high = suvmax >= suv_thr
        keep = (suv_high or big_enough) if mode == "either" else (suv_high and big_enough)
        if not keep: continue
        side = "R" if cx > mid+DZ else ("L" if cx < mid-DZ else "mid")
        nodes.append((maxmm, side))
    if not nodes: return 0, 0.0, "none"
    sides = set(s for _,s in nodes if s in ("L","R"))
    lat = "bilateral" if len(sides)>=2 else ("unilateral" if len(sides)==1 else "midline")
    return len(nodes), max(m for m,_ in nodes), lat

def rule_N(n, maxmm, lat, n3, n1):
    if n==0: return "N0"
    if maxmm>n3: return "N3"
    if n==1 and lat in ("unilateral","midline") and maxmm<=n1: return "N1"
    return "N2"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--save-summ", default="outputs/eda/node_summaries.npz")
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.save_summ), exist_ok=True)

    masks = sorted(glob.glob(os.path.join(args.pred_root,"fold_*/validation/*.nii.gz")))
    print(f"[masks] {len(masks)}", flush=True)

    print("[index] scanning data-root for CT/PET once...", flush=True)
    ct_index, pet_index = {}, {}
    for root, _dirs, files in os.walk(args.data_root):
        pid = os.path.basename(root)
        for b in files:
            if not b.endswith(".nii.gz"):
                continue
            if re.search(r"CT\.nii\.gz$", b, re.I):
                ct_index[pid] = os.path.join(root, b)
            elif re.search(r"(PT|PET)\.nii\.gz$", b, re.I):
                pet_index[pid] = os.path.join(root, b)
    print(f"[index] {len(ct_index)} CT / {len(pet_index)} PET", flush=True)
    if len(ct_index) == 0 and len(pet_index) == 0:
        print("[FATAL] no CT/PET found under --data-root. Check the path.", flush=True)
        sys.exit(1)

    clin = pd.read_csv(args.clinical)
    pidcol=[c for c in clin.columns if c.lower() in ("patientid","id")][0]
    ncol=[c for c in clin.columns if re.sub(r"[^a-z]","",c.lower()) in ("nstage","n")][0]
    clin[pidcol]=clin[pidcol].astype(str).str.strip()
    Ntrue=dict(zip(clin[pidcol], clin[ncol].astype(str).str.upper().str[:2]))

    summaries=[]   # (pid, {merge:comps}, mid, Ntrue)
    import time
    for i,mp in enumerate(masks,1):
        pid=os.path.basename(mp).replace(".nii.gz","")
        if Ntrue.get(pid) not in ("N0","N1","N2","N3"): continue
        t0=time.time()
        try:
            comps, mid = summarize_case(mp, ct_index.get(pid), pet_index.get(pid))
            summaries.append((pid, comps, mid, Ntrue[pid]))
        except Exception as e:
            print(f"  [skip] {pid}: {e}", flush=True)
        dt=time.time()-t0
        if i<=5 or dt>5 or i%50==0:
            print(f"  ...{i}/{len(masks)}  ({pid}: {dt:.1f}s)", flush=True)
    print(f"[loaded] {len(summaries)} usable cases", flush=True)

    grid_minml=[0.3,0.5,0.8,1.0,1.5,2.0,3.0]
    grid_suv=[0.0,1.5,2.0,2.5,3.0,4.0]
    grid_diam=[8.0,10.0,12.0,15.0]
    grid_n3=[45,50,55,60,65]
    grid_n1=[20,25,30,35,40]
    combos=list(itertools.product(MERGE_GRID, MODE_GRID, grid_minml, grid_suv, grid_diam))
    print(f"[sweep] {len(combos)} cleanup settings x {len(grid_n3)*len(grid_n1)} rule cutoffs...", flush=True)
    ytrue=[y for *_,y in summaries]
    best=(-1,None,None)
    for ci,(merge_mm,mode,min_ml,suv,diam) in enumerate(combos,1):
        feats=[features(c[merge_mm], mid, min_ml, suv, diam, mode) for _,c,mid,_ in summaries]
        for n3,n1 in itertools.product(grid_n3,grid_n1):
            pred=[rule_N(f[0],f[1],f[2],n3,n1) for f in feats]
            ba=balanced_accuracy_score(ytrue,pred)
            if ba>best[0]:
                best=(ba,dict(merge_mm=merge_mm,mode=mode,min_ml=min_ml,suv=suv,diam=diam,n3=n3,n1=n1),pred)
                print(f"  [{ci}/{len(combos)}] NEW BEST {best[0]:.3f}  {best[1]}", flush=True)
        if ci % 200 == 0:
            print(f"  [{ci}/{len(combos)}] ... still searching, best-so-far {best[0]:.3f}", flush=True)

    print("\n=== BEST N setting on predicted masks ===", flush=True)
    print(f"balanced accuracy = {best[0]:.3f}", flush=True)
    print(f"params = {best[1]}", flush=True)
    # per-class recall at best
    from collections import Counter
    yt=np.array(ytrue); yp=np.array(best[2])
    for cls in ["N0","N1","N2","N3"]:
        mask=yt==cls
        if mask.sum(): print(f"  {cls}: recall {(yp[mask]==cls).mean():.3f}  (n={mask.sum()})", flush=True)

if __name__=="__main__":
    main()
    
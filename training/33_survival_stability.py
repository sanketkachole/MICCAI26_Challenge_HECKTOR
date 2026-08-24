#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 33: is the npICARE > Cox LOCO advantage STABLE, or noise?

LOCO on 8 centers is only 8 numbers, some from tiny centers (USZ n=11, HMR n=18).
Before shipping npICARE over Cox we check:
  1. PER-CENTER LOCO C-index for Cox vs npICARE (does ICARE win consistently, or is
     it one lucky center?)
  2. REPEATED random 5-fold CV (10 different seeds) -> mean +/- std, so we see whether
     the CV difference is within noise.
  3. Paired comparison: on how many centers / repeats does npICARE beat Cox?

This decides whether we ship npICARE, Cox, or a blend.
Run in `icare` env (or HECKTOR - only needs lifelines+numpy+sklearn).
"""
import argparse, json, re, warnings
print("[boot] importing...", flush=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
print("[boot] imports done", flush=True)

import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "c31", os.path.join(os.path.dirname(os.path.abspath(__file__)), "31_container_ensemble.py"))
_c31 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_c31)
NumpyICARE = _c31.NumpyICARE
BaggedNumpyICARE = _c31.BaggedNumpyICARE
load = _c31.load
prep = _c31.prep
GEOM = _c31.GEOM
CLIN = _c31.CLIN


def fit_cox(Xtr, ttr, etr):
    d = pd.DataFrame(Xtr, columns=[f"f{i}" for i in range(Xtr.shape[1])])
    d["T"] = ttr; d["E"] = etr
    return CoxPHFitter(penalizer=0.5).fit(d, "T", "E")


def pred_cox(cph, Xva):
    d = pd.DataFrame(Xva, columns=[f"f{i}" for i in range(Xva.shape[1])])
    return cph.predict_partial_hazard(d).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()

    df, pet_feats = load(args)
    base = [c for c in GEOM if c in df.columns] + [k for k in CLIN if k in df.columns]
    if "N_ordinal" in df.columns: base += ["N_ordinal"]
    if "HPV_missing" in df.columns: base += ["HPV_missing"]
    pet_avail = [c for c in (pet_feats + ["TLG_gtvp","TLG_gtvn"]) if c in df.columns]
    feats = base + pet_avail
    print(f"[feats] {len(feats)}", flush=True)

    # ---------- 1. per-center LOCO ----------
    print("\n" + "=" * 62, flush=True)
    print("PER-CENTER LOCO (is the ICARE advantage consistent?)", flush=True)
    print("=" * 62, flush=True)
    print(f"{'center':<8}{'n':>5}{'events':>8}{'Cox':>10}{'npICARE':>10}{'winner':>10}", flush=True)
    print("-" * 62, flush=True)
    wins = {"Cox": 0, "npICARE": 0}
    rows = []
    for c in sorted(df.center.unique()):
        tr = df[df.center != c]; va = df[df.center == c]
        if len(va) < 8 or va["event"].sum() < 3:
            print(f"{c:<8}{len(va):>5}{int(va['event'].sum()):>8}{'--':>10}{'--':>10}{'(skip)':>10}", flush=True)
            continue
        Xtr, Xva = prep(tr[feats], va[feats])
        ttr, etr = tr["RFS"].values, tr["event"].values
        tva, eva = va["RFS"].values, va["event"].values
        try:
            ci_cox = float(concordance_index(tva, -pred_cox(fit_cox(Xtr, ttr, etr), Xva), eva))
        except Exception:
            ci_cox = float("nan")
        ci_ic = float(concordance_index(tva, -NumpyICARE().fit(Xtr, ttr, etr).predict(Xva), eva))
        w = "npICARE" if ci_ic > ci_cox else "Cox"
        wins[w] += 1
        rows.append((c, len(va), ci_cox, ci_ic))
        print(f"{c:<8}{len(va):>5}{int(va['event'].sum()):>8}{ci_cox:>10.3f}{ci_ic:>10.3f}{w:>10}", flush=True)
    print("-" * 62, flush=True)
    print(f"center wins -> Cox: {wins['Cox']}, npICARE: {wins['npICARE']}", flush=True)
    if rows:
        mc = np.nanmean([r[2] for r in rows]); mi = np.nanmean([r[3] for r in rows])
        print(f"mean LOCO   -> Cox: {mc:.3f}, npICARE: {mi:.3f}  (diff {mi-mc:+.3f})", flush=True)

    # ---------- 2. repeated random CV ----------
    print("\n" + "=" * 62, flush=True)
    print("REPEATED 5-FOLD CV (10 seeds): is the CV difference within noise?", flush=True)
    print("=" * 62, flush=True)
    cox_scores, ic_scores = [], []
    strat = df["event"].astype(int).astype(str) + "_" + df["center"].astype(str)
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cs, is_ = [], []
        for tr_i, va_i in skf.split(df, strat):
            tr = df.iloc[tr_i]; va = df.iloc[va_i]
            if va["event"].sum() < 3: continue
            Xtr, Xva = prep(tr[feats], va[feats])
            ttr, etr = tr["RFS"].values, tr["event"].values
            tva, eva = va["RFS"].values, va["event"].values
            try:
                cs.append(float(concordance_index(tva, -pred_cox(fit_cox(Xtr,ttr,etr), Xva), eva)))
            except Exception:
                pass
            is_.append(float(concordance_index(tva, -NumpyICARE().fit(Xtr,ttr,etr).predict(Xva), eva)))
        if cs: cox_scores.append(np.mean(cs))
        if is_: ic_scores.append(np.mean(is_))
        print(f"  seed {seed}: Cox={np.mean(cs):.3f}  npICARE={np.mean(is_):.3f}", flush=True)
    print("-" * 62, flush=True)
    print(f"Cox     : {np.mean(cox_scores):.3f} +/- {np.std(cox_scores):.3f}", flush=True)
    print(f"npICARE : {np.mean(ic_scores):.3f} +/- {np.std(ic_scores):.3f}", flush=True)
    diff = np.array(ic_scores) - np.array(cox_scores)
    print(f"diff    : {diff.mean():+.3f} +/- {diff.std():.3f}  "
          f"(npICARE better in {int((diff>0).sum())}/{len(diff)} seeds)", flush=True)

    print("\n[verdict] ship npICARE over Cox only if it wins on MOST centers AND", flush=True)
    print("          the repeated-CV difference is not clearly negative.", flush=True)


if __name__ == "__main__":
    main()

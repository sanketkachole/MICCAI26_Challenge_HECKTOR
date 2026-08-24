#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 22: ICARE vs current Cox for RFS prognosis, Priority 1a.

Tests whether ICARE (binary-weighted ensemble, HECKTOR 2022 winner) beats our current
Cox model on the features WE ALREADY HAVE:
  geometry (gtvp_ml, n_gtvn, max_node_mm, gtvn_total_ml) + clinical + predicted N-ordinal
  + the existing 428-column radiomics bank (radiomics_pred.csv)

Compares several configs with the SAME 5-fold splits + LOCO honesty guards used
everywhere else. ICARE resists overfitting with many features (unlike Cox), so unlike
step 16 we can feed it the FULL radiomics bank without preselection.

Runs in the `icare` env (sklearn 1.3.2 + scikit-survival 0.22.2 + icare).
Uses ICARE's own harrell_cindex (no lifelines dependency).

USAGE:
  python 22_icare_vs_cox.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --radiomics outputs/eda/radiomics_pred.csv \
    --splits outputs/eda/splits_2026.json
"""
import argparse, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from icare.survival import IcareSurvival, BaggedIcareSurvival, harrell_cindex

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]
N_ORD = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}


def det(cols, canon):
    for c in cols:
        cl = re.sub(r"[^a-z]", "", c.lower())
        if canon == "Age" and cl == "age": return c
        if canon == "Gender" and cl in ("gender", "sex"): return c
        if canon == "Tobacco" and "tobacco" in cl: return c
        if canon == "Alcohol" and "alcohol" in cl: return c
        if canon == "Performance" and "performance" in cl: return c
        if canon == "HPV" and "hpv" in cl: return c
        if canon == "Treatment" and "treatment" in cl and "id" not in cl: return c
        if canon == "RFS" and cl in ("rfs", "survival", "time"): return c
        if canon == "Relapse" and cl in ("relapse", "event", "recurrence"): return c
    return None


def make_y(dur, evt):
    return np.array(list(zip(evt.astype(bool), dur.astype(float))),
                    dtype=[('event', '?'), ('time', '<f8')])


def load(args):
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    rfs = det(clin.columns, "RFS"); rel = det(clin.columns, "Relapse")
    keep = {pidc: "PatientID", rfs: "RFS", rel: "event"}
    for k in CLIN:
        v = det(clin.columns, k)
        if v: keep[v] = k
    df = clin[list(keep)].rename(columns=keep)
    df = df.dropna(subset=["RFS", "event"]); df = df[df["RFS"] > 0]

    meta = pd.read_csv(args.meta)
    geomcols = [c for c in GEOM if c in meta.columns]
    mkeep = ["PatientID"] + geomcols + (["laterality"] if "laterality" in meta.columns else [])
    meta = meta[mkeep]
    if "laterality" in meta.columns:
        def rn(r):
            if r["n_gtvn"] == 0: return 0
            if r["max_node_mm"] > 55: return 3
            if r["n_gtvn"] == 1 and r["laterality"] in ("unilateral", "midline") and r["max_node_mm"] <= 35: return 1
            return 2
        meta["N_ordinal"] = meta.apply(rn, axis=1)
        meta = meta.drop(columns=["laterality"])

    rad = pd.read_csv(args.radiomics)
    rad_cols = [c for c in rad.columns if c.split("_")[0] in ("ct", "pt") and not c.startswith("err_")]
    rad = rad[["PatientID"] + rad_cols]

    df = df.merge(meta, on="PatientID", how="left").merge(rad, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    clin_feats = [k for k in CLIN if k in df.columns]
    base = [c for c in GEOM if c in df.columns] + clin_feats + (["N_ordinal"] if "N_ordinal" in df.columns else [])
    return df, base, rad_cols


def prep(Xtr, Xva):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return (sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva)))


def eval_model(df, feats, splits, kind, use_loco=False):
    """kind: 'cox' | 'icare' | 'bagged'. Returns mean C-index over folds/centers."""
    groups = ([{"val": list(df[df.center == c]["PatientID"])} for c in sorted(df.center.unique())]
              if use_loco else splits)
    cis = []
    for g in groups:
        va_ids = set(g["val"])
        tr = df[~df.PatientID.isin(va_ids)]; va = df[df.PatientID.isin(va_ids)]
        if len(va) < 8 or len(tr) < 30 or va["event"].sum() < 3:
            continue
        Xtr, Xva = prep(tr[feats], va[feats])
        ytr = make_y(tr["RFS"], tr["event"])
        yva = make_y(va["RFS"], va["event"])
        try:
            if kind == "cox":
                from sksurv.linear_model import CoxPHSurvivalAnalysis
                mdl = CoxPHSurvivalAnalysis(alpha=1.0)
                mdl.fit(Xtr, ytr); risk = mdl.predict(Xva)
            elif kind == "icare":
                mdl = IcareSurvival(); mdl.fit(Xtr, ytr); risk = np.asarray(mdl.predict(Xva))
            elif kind == "bagged":
                mdl = BaggedIcareSurvival(n_estimators=40, aggregation_method="median",
                                          random_state=0, n_jobs=4)
                mdl.fit(Xtr, ytr); risk = np.asarray(mdl.predict(Xva))
            cis.append(float(harrell_cindex(yva, risk)))
        except Exception as e:
            continue
    return float(np.mean(cis)) if cis else float("nan"), len(cis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()

    df, base, rad_cols = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict): splits = splits.get("folds", splits.get("splits", []))
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    print(f"[data] {len(df)} patients w/ RFS+event ({int(df['event'].sum())} events)", flush=True)
    print(f"[data] base feats: {len(base)} | radiomics: {len(rad_cols)}", flush=True)
    print(f"[data] splits: {len(splits)} folds", flush=True)
    print()

    configs = [
        ("Cox   base (geom+clin+N)",         "cox",    base),
        ("ICARE base (geom+clin+N)",         "icare",  base),
        ("ICARE base + radiomics",           "icare",  base + rad_cols),
        ("BaggedICARE base + radiomics",     "bagged", base + rad_cols),
        ("BaggedICARE radiomics only",       "bagged", rad_cols),
    ]
    print(f"{'config':<34}{'CV_Cindex':>12}{'(folds)':>9}{'LOCO':>10}", flush=True)
    print("-" * 65, flush=True)
    rows = []
    for name, kind, feats in configs:
        cv, nf = eval_model(df, feats, splits, kind, use_loco=False)
        loco, _ = eval_model(df, feats, splits, kind, use_loco=True)
        rows.append((name, cv, loco))
        print(f"{name:<34}{cv:>12.3f}{nf:>9}{loco:>10.3f}", flush=True)

    print()
    cox_cv = rows[0][1]
    best = max(rows, key=lambda r: (r[1] if not np.isnan(r[1]) else -1))
    print(f"[Cox baseline]  CV C-index: {cox_cv:.3f}", flush=True)
    print(f"[best]          {best[0]}: CV={best[1]:.3f}, LOCO={best[2]:.3f}", flush=True)
    d = best[1] - cox_cv
    print(f"[verdict]       best changes CV C-index by {d:+.3f} vs our Cox", flush=True)
    if d > 0.01 and best[2] >= rows[0][2] - 0.02:
        print("                -> ICARE HELPS and generalizes. Worth building the full 1b bank.", flush=True)
    elif d > 0.01:
        print("                -> better CV but LOCO weaker: center-overfit risk. Cautious.", flush=True)
    else:
        print("                -> ICARE does NOT beat Cox on existing features. Reconsider 1b.", flush=True)


if __name__ == "__main__":
    main()

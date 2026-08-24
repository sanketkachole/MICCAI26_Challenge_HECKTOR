#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 03: TN staging (radiomics-free version)
N-stage : RULES from mask geometry (node count, biggest node mm, side).
T-stage : small HistGradientBoosting model on tumor size + node burden + clinical.
Trains on case_metadata.csv (step 00) + clinical CSV. No radiomics.
Saves models/staging_Tmodel.joblib and models/staging_config.json for inference.
"""
import argparse, json, os, re
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

COLUMN_PATTERNS = {
    "PatientID": [r"^patient.?id$", r"^id$"],
    "Age": [r"^age$"], "Gender": [r"^gender$", r"^sex$"],
    "Tobacco": [r"tobacco", r"^smok"], "Alcohol": [r"alcohol"],
    "Performance": [r"performance", r"^ecog$"], "HPV": [r"hpv"],
    "Treatment": [r"treatment"], "Tstage": [r"^t.?stage$"], "Nstage": [r"^n.?stage$"],
}
CLINICAL_FEATURES = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]
GEOM_FEATURES = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]

def detect_columns(headers):
    m, used = {}, set()
    for canon, pats in COLUMN_PATTERNS.items():
        for h in headers:
            if h in used: continue
            if any(re.search(p, str(h).strip().lower()) for p in pats):
                m[canon] = h; used.add(h); break
    return m

def get_meta_col(meta, *names):
    for n in names:
        if n in meta.columns: return n
    return None

def rule_N(n_nodes, max_mm, laterality, n3_mm, n1_mm):
    if n_nodes == 0: return "N0"
    if max_mm > n3_mm: return "N3"
    if n_nodes == 1 and laterality in ("unilateral", "midline") and max_mm <= n1_mm: return "N1"
    return "N2"

def fit_encoders(df, cols):
    """Learn a deterministic category->code map for text columns (saved for inference)."""
    enc = {}
    for c in cols:
        if c in df.columns and df[c].dtype.kind not in "biufc":
            cats = list(pd.Categorical(df[c].astype(str).replace({"nan": np.nan})).categories)
            enc[c] = cats
    return enc


def apply_encoders(df, cols, enc):
    """Build a float matrix using saved encoders; numeric cols pass through; NaN kept."""
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            X[c] = np.nan
            continue
        s = df[c]
        if c in enc:  # text column with a saved category list
            cat = pd.Categorical(s.astype(str), categories=enc[c])
            codes = pd.Series(cat.codes, index=df.index).astype(float)
            codes[codes < 0] = np.nan  # unseen category -> NaN (HistGB handles)
            X[c] = codes
        else:
            X[c] = pd.to_numeric(s, errors="coerce")
    return X.values

def cv_balanced_acc_T(X, y, groups_by_fold):
    preds = np.empty(len(y), dtype=object)
    for fold, (tr_idx, va_idx) in groups_by_fold:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
              l2_regularization=1.0, random_state=0, class_weight="balanced")
        clf.fit(X[tr_idx], y[tr_idx]); preds[va_idx] = clf.predict(X[va_idx])
    return balanced_accuracy_score(y, preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="./models")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clin = pd.read_csv(args.clinical); meta = pd.read_csv(args.meta)
    cmap = detect_columns(list(clin.columns)); pid_c = cmap["PatientID"]
    clin[pid_c] = clin[pid_c].astype(str).str.strip()

    mcol = {
        "n_gtvn": get_meta_col(meta, "n_gtvn", "geom_n_nodes"),
        "max_node_mm": get_meta_col(meta, "max_node_mm", "geom_max_node_mm"),
        "laterality": get_meta_col(meta, "laterality", "geom_laterality"),
        "gtvp_ml": get_meta_col(meta, "gtvp_ml", "geom_gtvp_ml"),
        "gtvn_total_ml": get_meta_col(meta, "gtvn_total_ml", "geom_total_vol_ml"),
        "PatientID": get_meta_col(meta, "PatientID"),
    }
    meta = meta.rename(columns={v: k for k, v in mcol.items() if v})
    meta["PatientID"] = meta["PatientID"].astype(str).str.strip()

    df = clin.merge(meta, left_on=pid_c, right_on="PatientID", how="inner")
    df["center"] = df["PatientID"].str.split("-").str[0]
    print(f"[data] merged {len(df)} patients")

    splits = json.load(open(args.splits))
    pid_to_fold = {str(p): f["fold"] for f in splits["folds"] for p in f["val"]}
    df["fold"] = df["PatientID"].map(pid_to_fold)

    # N-stage rules + threshold tuning
    Ntrue = df[cmap["Nstage"]].astype(str).str.strip().str.upper().str[:2]
    valid = Ntrue.isin(["N0","N1","N2","N3"]) & df["n_gtvn"].notna()
    dN = df[valid].copy(); yN = Ntrue[valid].values
    lat = dN["laterality"].fillna("none").values
    nn = dN["n_gtvn"].fillna(0).values; mm = dN["max_node_mm"].fillna(0.0).values
    best = (-1, 60.0, 30.0)
    for n3 in [50,55,58,60,62,65]:
        for n1 in [25,28,30,32,35]:
            pred = [rule_N(nn[i], mm[i], lat[i], n3, n1) for i in range(len(dN))]
            ba = balanced_accuracy_score(yN, pred)
            if ba > best[0]: best = (ba, float(n3), float(n1))
    baN_all, N3_MM, N1_MM = best
    predN = [rule_N(nn[i], mm[i], lat[i], N3_MM, N1_MM) for i in range(len(dN))]
    print(f"\n[N-stage RULE] tuned: N3>{N3_MM}mm, N1<={N1_MM}mm")
    print(f"[N-stage RULE] balanced accuracy (GT masks): {baN_all:.3f}")
    for c in sorted(dN["center"].unique()):
        idx = dN["center"].values == c
        if idx.sum() >= 5:
            print(f"    center {c:5s} n={idx.sum():3d} BA={balanced_accuracy_score(np.array(yN)[idx], np.array(predN)[idx]):.3f}")

    # T-stage model
    Ttrue = df[cmap["Tstage"]].astype(str).str.strip().str.upper().str[:2].replace({"T0":"T1"})
    okT = Ttrue.isin(["T1","T2","T3","T4"])
    dT = df[okT].copy(); yT = Ttrue[okT].values
    feats = GEOM_FEATURES + CLINICAL_FEATURES
    dT_ren = dT.rename(columns={cmap[k]: k for k in CLINICAL_FEATURES if k in cmap})
    encoders = fit_encoders(dT_ren, feats)          # deterministic, saved for inference
    XT = apply_encoders(dT_ren, feats, encoders)
    fold_arr = dT["fold"].values; idx_all = np.arange(len(dT)); groups = []
    for fo in sorted(pd.Series(fold_arr).dropna().unique()):
        va = idx_all[fold_arr == fo]; tr = idx_all[fold_arr != fo]
        if len(va) and len(tr): groups.append((fo, (tr, va)))
    baT = cv_balanced_acc_T(XT, yT, groups)
    print(f"\n[T-stage MODEL] features: {feats}")
    print(f"[T-stage MODEL] CV balanced accuracy: {baT:.3f}")

    Tclf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
           l2_regularization=1.0, random_state=0, class_weight="balanced")
    Tclf.fit(XT, yT)
    joblib.dump(Tclf, os.path.join(args.out, "staging_Tmodel.joblib"))
    cfg = {"N_rule": {"N3_MM": N3_MM, "N1_MAX_MM": N1_MM, "MIDLINE_DEADZONE_MM": 6.0},
           "T_features": feats,
           "T_encoders": encoders,
           "T_clinical_source": {k: cmap.get(k) for k in CLINICAL_FEATURES},
           "T_classes": ["T1","T2","T3","T4"],
           "cv_scores": {"N_balanced_acc_GTmask": round(baN_all,4), "T_balanced_acc_cv": round(baT,4)}}
    json.dump(cfg, open(os.path.join(args.out, "staging_config.json"), "w"), indent=2)
    print(f"\n[save] {args.out}/staging_Tmodel.joblib")
    print(f"[save] {args.out}/staging_config.json")
    print(f"\nEXPECTED staging (mean of N,T BA on GT masks) ~= {(baN_all+baT)/2:.3f} (drops a bit on predicted masks)")

if __name__ == "__main__":
    main()
    
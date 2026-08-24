#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 27 (Phase 1b): T-stage label cleanup.

Phase-0 found junk T labels ('0': 3 patients, 'N': 4 patients) polluting the T-model.
Balanced accuracy weights each class equally, so tiny junk classes drag it down a lot.

This script:
  1. Inspects the raw T values (what are '0' and 'N' really?)
  2. Compares T-model balanced accuracy under handling strategies:
       A. current (keep all, including junk)  -> reproduces the low ~0.34
       B. drop junk rows (T in T1-T4 only)     -> the honest T accuracy
       C. drop junk + size-focused features
  3. Reports per-class recall so we see if cleaning helps the real T1-T4 classes.

Uses vectorized ops + flushed prints (lessons learned). REPORT ONLY.
USAGE (HECKTOR env, as a batch job):
  python 27_t_cleanup.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --splits outputs/eda/splits_2026.json
"""
import argparse, json, re, warnings
print("[boot] importing...", flush=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
print("[boot] imports done", flush=True)

GEOM_ALL = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
GEOM_SIZE = ["gtvp_ml", "max_node_mm", "gtvn_total_ml"]  # gtvp size matters most for T
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]
VALID_T = ["1", "2", "3", "4"]


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
    return None


def load(args):
    print("[load] reading CSVs...", flush=True)
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    tcol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("tstage", "t")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    cmap = {det(clin.columns, k): k for k in CLIN if det(clin.columns, k)}
    keep = {pidc: "PatientID", tcol: "T_raw", **cmap}
    clin = clin[list(keep)].rename(columns=keep)
    # normalize: strip "T", uppercase
    clin["T"] = clin["T_raw"].astype(str).str.upper().str.replace("T", "", regex=False).str.strip().str[:1]
    meta = pd.read_csv(args.meta)
    df = clin.merge(meta, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    print(f"[load] done ({len(df)} patients)", flush=True)
    return df


def eval_T(df, feats, splits, model_ctor):
    """5-fold CV balanced accuracy on whatever rows df contains."""
    y = df["T"].values
    X = df[feats]
    preds = pd.Series(index=df.index, dtype=object)
    for f in splits:
        va_ids = set(f["val"])
        tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
        if va.sum() == 0 or tr.sum() < 20: continue
        m = model_ctor(); m.fit(X[tr], y[tr])
        preds[va] = m.predict(X[va])
    valid = preds.notna()
    return (balanced_accuracy_score(y[valid.values], preds[valid].values),
            y[valid.values], preds[valid].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()

    df = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict): splits = splits.get("folds", splits.get("splits", []))
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    # ---- 1. inspect raw T values ----
    print("\n" + "=" * 60, flush=True)
    print("RAW T-STAGE VALUES", flush=True)
    print("=" * 60, flush=True)
    print("normalized T distribution:", df["T"].value_counts(dropna=False).sort_index().to_dict(), flush=True)
    junk = df[~df["T"].isin(VALID_T)]
    print(f"\njunk-label rows ({len(junk)} total):", flush=True)
    for _, r in junk.iterrows():
        print(f"    {r['PatientID']:<12} T_raw='{r['T_raw']}' -> normalized '{r['T']}'", flush=True)

    clin_feats = [c for c in CLIN if c in df.columns]

    def histgb(): return Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", StandardScaler()),
                                   ("clf", HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                            learning_rate=0.05, random_state=0))])
    def logreg(): return Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", StandardScaler()),
                                   ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])

    # ---- 2. compare strategies ----
    print("\n" + "=" * 60, flush=True)
    print("T-MODEL BALANCED ACCURACY UNDER CLEANUP STRATEGIES", flush=True)
    print("=" * 60, flush=True)

    fA = GEOM_ALL + clin_feats
    # A. keep all (junk included) - reproduces the polluted number
    baA, ytA, ypA = eval_T(df, fA, splits, histgb)
    print(f"\nA. keep junk, HistGB, all-geom:          {baA:.3f}", flush=True)

    # B. drop junk rows
    dfc = df[df["T"].isin(VALID_T)].copy()
    print(f"   (dropped {len(df)-len(dfc)} junk rows -> {len(dfc)} clean patients)", flush=True)
    baB, ytB, ypB = eval_T(dfc, fA, splits, histgb)
    print(f"B. drop junk, HistGB, all-geom:          {baB:.3f}", flush=True)

    # C. drop junk + LogReg (class-balanced)
    baC, ytC, ypC = eval_T(dfc, fA, splits, logreg)
    print(f"C. drop junk, LogReg balanced, all-geom: {baC:.3f}", flush=True)

    # D. drop junk + LogReg + size-focused
    fS = GEOM_SIZE + clin_feats
    baD, ytD, ypD = eval_T(dfc, fS, splits, logreg)
    print(f"D. drop junk, LogReg, size-focused:      {baD:.3f}", flush=True)

    # ---- 3. per-class recall for the best clean strategy ----
    best = max([("B",baB,ytB,ypB),("C",baC,ytC,ypC),("D",baD,ytD,ypD)], key=lambda x: x[1])
    print(f"\nper-class recall (best clean strategy = {best[0]}, BA={best[1]:.3f}):", flush=True)
    cm = confusion_matrix(best[2], best[3], labels=VALID_T)
    for i, t in enumerate(VALID_T):
        rec = cm[i].sum() and cm[i][i]/cm[i].sum()
        print(f"    T{t}: recall={rec:.2f}  (support={cm[i].sum()})", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"[verdict] current (polluted) T BA = {baA:.3f}", flush=True)
    print(f"          cleaned best T BA        = {best[1]:.3f}  ({best[1]-baA:+.3f})", flush=True)
    if best[1] - baA > 0.02:
        print("          -> cleaning junk labels RECOVERS T accuracy. Apply in container.", flush=True)
    else:
        print("          -> cleaning helps little; junk labels were not the main issue.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

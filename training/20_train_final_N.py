#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 20: train the FINAL N-stage classifier on all patients and save it
for the container. Replaces the hand rule (offline CV 0.646) with a learned HistGB model
(offline CV 0.717, LOCO-verified).

Features (must match exactly what the container computes from the predicted mask):
  n_gtvn, max_node_mm, gtvn_total_ml, gtvp_ml, lat_unilateral, lat_bilateral,
  lat_midline, lat_none

Saves n_classifier.joblib:
  { "model": <fitted sklearn Pipeline>,
    "features": [...ordered feature names...],
    "classes": [...],
    "laterality_levels": ["unilateral","bilateral","midline","none"] }

The Pipeline includes median-impute + standardize + HistGB, so the container just builds
the feature row and calls predict.

USAGE:
  python 20_train_final_N.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --splits outputs/eda/splits_2026.json \
    --out outputs/models_pred2/n_classifier.joblib
"""
import argparse, json, re, warnings, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
import joblib

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
LAT_LEVELS = ["unilateral", "bilateral", "midline", "none"]
FEATURES = GEOM + [f"lat_{lv}" for lv in LAT_LEVELS]


def rule_N(n, maxmm, lat, n3=55.0, n1=35.0):
    if n == 0: return "N0"
    if maxmm > n3: return "N3"
    if n == 1 and lat in ("unilateral", "midline") and maxmm <= n1: return "N1"
    return "N2"


def load(args):
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    ncol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("nstage", "n")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    clin = clin[[pidc, ncol]].rename(columns={pidc: "PatientID", ncol: "N"})
    clin["N"] = clin["N"].astype(str).str.upper().str[:2]
    clin = clin[clin["N"].isin(["N0", "N1", "N2", "N3"])]

    meta = pd.read_csv(args.meta)
    df = clin.merge(meta, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    for lv in LAT_LEVELS:
        df[f"lat_{lv}"] = (df.get("laterality", "none") == lv).astype(int)
    return df


def make_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                                               learning_rate=0.05,
                                               class_weight="balanced", random_state=0)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="outputs/models_pred2/n_classifier.joblib")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict):
        splits = splits.get("folds", splits.get("splits", []))
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    print(f"[data] {len(df)} patients", flush=True)
    print(f"[data] N dist: {df['N'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"[features] {FEATURES}", flush=True)

    X = df[FEATURES]; y = df["N"].values

    # --- sanity: reproduce the honest CV before we save the all-data model ---
    cv_learned, cv_rule = [], []
    for f in splits:
        va_ids = set(f["val"])
        tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
        if va.sum() == 0: continue
        p = make_pipe().fit(X[tr], y[tr])
        cv_learned.append(balanced_accuracy_score(y[va], p.predict(X[va])))
        rule_pred = df[va].apply(lambda r: rule_N(r["n_gtvn"], r["max_node_mm"],
                                                  r.get("laterality", "none")), axis=1)
        cv_rule.append(balanced_accuracy_score(y[va], rule_pred))
    print(f"[check] CV balanced acc  learned={np.mean(cv_learned):.3f}  rule={np.mean(cv_rule):.3f}", flush=True)

    # --- fit final model on ALL data ---
    final = make_pipe().fit(X, y)
    payload = {
        "model": final,
        "features": FEATURES,
        "classes": list(final.classes_),
        "laterality_levels": LAT_LEVELS,
    }
    joblib.dump(payload, args.out)
    print(f"[save] {args.out}", flush=True)
    print(f"[save] classes: {list(final.classes_)}", flush=True)

    # quick self-test: predict on a couple of rows
    demo = X.iloc[:3]
    print(f"[demo] sample predictions: {list(final.predict(demo))}", flush=True)


if __name__ == "__main__":
    main()

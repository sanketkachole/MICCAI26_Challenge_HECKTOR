#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 18: learned N-stage classifier vs the hand-built rule (0.657).

Trains on PREDICTED-mask features (so it matches test conditions):
  geometry: n_gtvn, max_node_mm, gtvn_total_ml, gtvp_ml, laterality(one-hot)
  + node PET/CT radiomics stats (SUV mean/max/energy, node shape) if available
Compares against the deterministic rule on the SAME 5-fold splits + LOCO.

Guards: feature selection is light (few, meaningful features), models are shallow,
LOCO confirms cross-center generalization. REPORT ONLY.

USAGE:
  python 18_train_N_classifier.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --radiomics outputs/eda/radiomics_pred.csv \
    --splits outputs/eda/splits_2026.json
"""
import argparse, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]


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
    keep = ["PatientID"] + [c for c in GEOM if c in meta.columns] + \
           (["laterality"] if "laterality" in meta.columns else [])
    meta = meta[keep]

    # radiomics: keep only NODE (gtvn) features - relevant to N-staging
    rad = pd.read_csv(args.radiomics)
    node_rad = [c for c in rad.columns
                if c.startswith(("ct_gtvn_", "pt_gtvn_")) and not c.startswith("err_")]
    rad = rad[["PatientID"] + node_rad]

    df = clin.merge(meta, on="PatientID", how="left").merge(rad, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]

    # one-hot laterality
    if "laterality" in df.columns:
        for lv in ["unilateral", "bilateral", "midline", "none"]:
            df[f"lat_{lv}"] = (df["laterality"] == lv).astype(int)
    return df, node_rad


def rule_baseline(df, splits):
    """Rule N balanced accuracy on the CV val folds and overall."""
    def pred_row(r):
        return rule_N(r["n_gtvn"], r["max_node_mm"], r.get("laterality", "none"))
    df = df.copy()
    df["rule"] = df.apply(pred_row, axis=1)
    cv = []
    for f in splits:
        va = df[df.PatientID.isin(set(f["val"]))]
        if len(va) == 0: continue
        cv.append(balanced_accuracy_score(va["N"], va["rule"]))
    loco = []
    for c in sorted(df.center.unique()):
        va = df[df.center == c]
        if len(va) < 5: continue
        loco.append(balanced_accuracy_score(va["N"], va["rule"]))
    overall = balanced_accuracy_score(df["N"], df["rule"])
    return float(np.mean(cv)), float(np.mean(loco)), overall, df["rule"].values


def build_pipe(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", model),
    ])


def cv_eval(df, feats, splits, model_fn, use_loco=False):
    groups = ([{"val": list(df[df.center == c]["PatientID"])} for c in sorted(df.center.unique())]
              if use_loco else splits)
    y = df["N"].values; X = df[feats]
    accs = []
    for g in groups:
        va_ids = set(g["val"])
        tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
        if va.sum() < 5 or tr.sum() < 20: continue
        pipe = build_pipe(model_fn())
        pipe.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[va], pipe.predict(X[va])))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()

    df, node_rad = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict):
        splits = splits.get("folds", splits.get("splits", []))
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    print(f"[data] {len(df)} patients with N-stage", flush=True)
    print(f"[data] N distribution: {df['N'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"[data] node radiomics cols: {len(node_rad)}", flush=True)
    print()

    # ---- baseline: the deterministic rule ----
    r_cv, r_loco, r_overall, _ = rule_baseline(df, splits)
    print(f"[RULE baseline]  CV={r_cv:.3f}  LOCO={r_loco:.3f}  overall={r_overall:.3f}", flush=True)
    print()

    lat_oh = [c for c in df.columns if c.startswith("lat_")]
    geom_feats = [c for c in GEOM if c in df.columns] + lat_oh

    feature_sets = {
        "geom_only":            geom_feats,
        "geom+node_radiomics":  geom_feats + node_rad,
    }
    models = {
        "HistGB": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                                                         learning_rate=0.05,
                                                         class_weight="balanced", random_state=0),
        "RF":     lambda: RandomForestClassifier(n_estimators=400, max_depth=6,
                                                 class_weight="balanced", random_state=0, n_jobs=2),
        "LogReg": lambda: LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
    }

    print(f"{'feature_set':<24}{'model':<8}{'CV_bal_acc':>12}{'LOCO':>10}", flush=True)
    print("-" * 54, flush=True)
    rows = []
    for fs_name, fcols in feature_sets.items():
        for m_name, m_fn in models.items():
            cv = cv_eval(df, fcols, splits, m_fn, use_loco=False)
            loco = cv_eval(df, fcols, splits, m_fn, use_loco=True)
            rows.append((fs_name, m_name, cv, loco))
            print(f"{fs_name:<24}{m_name:<8}{cv:>12.3f}{loco:>10.3f}", flush=True)

    print()
    best = max(rows, key=lambda r: (r[2] if not np.isnan(r[2]) else -1))
    print(f"[best learned]  {best[0]} / {best[1]}: CV={best[2]:.3f}  LOCO={best[3]:.3f}", flush=True)
    print(f"[rule]          CV={r_cv:.3f}  LOCO={r_loco:.3f}", flush=True)
    d = best[2] - r_cv
    print(f"[verdict]       learned model changes N CV by {d:+.3f} vs rule", flush=True)
    if d > 0.02 and best[3] >= r_loco - 0.02:
        print("                -> learned N-classifier BEATS the rule and generalizes. Ship it.", flush=True)
    elif d > 0.02:
        print("                -> beats rule in CV but LOCO weaker: center-overfit risk.", flush=True)
    else:
        print("                -> learned model does NOT beat the rule. Keep the rule.", flush=True)


if __name__ == "__main__":
    main()

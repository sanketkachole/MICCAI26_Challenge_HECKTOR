#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 15: train a T-stage model using radiomics + clinical + geometry,
with STRICT overfitting guards, and compare honestly against the current geometry-only
T-model (baseline CV balanced accuracy ~0.518).

Guards:
  1. Feature selection: rank features by univariate ANOVA F on TRAIN only (inside each
     CV fold - no leakage), keep top-K.
  2. Honest evaluation: the SAME 5-fold splits used everywhere else (splits_2026.json).
  3. Leave-one-center-out (LOCO): confirms it generalizes across scanners, which is what
     the hidden test set actually measures.
  4. Compares several K values and model types; reports the geometry-only baseline too,
     so we can see if radiomics ACTUALLY helps or just inflates in-sample numbers.

Nothing is saved unless it beats baseline. This script only REPORTS - deciding what to
ship is a separate step once we see the numbers.

USAGE:
  python 15_train_T_radiomics.py \
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
CLIN_CANON = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]


def detect_clis(cols):
    cmap = {}
    for canon in CLIN_CANON:
        for c in cols:
            cl = re.sub(r"[^a-z]", "", c.lower())
            if canon.lower() == "age" and cl == "age": cmap[canon] = c
            elif canon.lower() == "gender" and cl in ("gender", "sex"): cmap[canon] = c
            elif canon.lower() == "tobacco" and "tobacco" in cl: cmap[canon] = c
            elif canon.lower() == "alcohol" and "alcohol" in cl: cmap[canon] = c
            elif canon.lower() == "performance" and "performance" in cl: cmap[canon] = c
            elif canon.lower() == "hpv" and "hpv" in cl: cmap[canon] = c
            elif canon.lower() == "treatment" and "treatment" in cl and "id" not in cl: cmap[canon] = c
    return cmap


def load(args):
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    tcol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("tstage", "t")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    cmap = detect_clis(clin.columns)
    clin_small = clin[[pidc, tcol] + list(cmap.values())].rename(
        columns={pidc: "PatientID", tcol: "T"})
    clin_small = clin_small.rename(columns={v: k for k, v in cmap.items()})
    clin_small["T"] = clin_small["T"].astype(str).str.upper().str.replace("T", "", regex=False)
    clin_small = clin_small[clin_small["T"].isin(["1", "2", "3", "4"])]

    meta = pd.read_csv(args.meta)[["PatientID"] + GEOM]
    rad = pd.read_csv(args.radiomics)
    rad_cols = [c for c in rad.columns
                if c.split("_")[0] in ("ct", "pt") and not c.startswith(("err_",))]
    rad = rad[["PatientID"] + rad_cols]

    df = clin_small.merge(meta, on="PatientID", how="left").merge(rad, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    return df, rad_cols


def build_pipe(model, k):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", model),
    ])


def cv_score(df, feat_cols, splits, model_fn, k):
    y = df["T"].values
    X = df[feat_cols]
    accs = []
    for fold in splits:
        va_ids = set(fold["val"])
        tr = ~df["PatientID"].isin(va_ids)
        va = df["PatientID"].isin(va_ids)
        if va.sum() == 0 or tr.sum() == 0:
            continue
        kk = min(k, X.shape[1])
        pipe = build_pipe(model_fn(), kk)
        pipe.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[va], pipe.predict(X[va])))
    return float(np.mean(accs)), float(np.std(accs))


def loco_score(df, feat_cols, model_fn, k):
    y = df["T"].values
    X = df[feat_cols]
    accs = []
    for ctr in sorted(df["center"].unique()):
        tr = df["center"] != ctr
        va = df["center"] == ctr
        if va.sum() < 5 or tr.sum() < 20:
            continue
        kk = min(k, X.shape[1])
        pipe = build_pipe(model_fn(), kk)
        pipe.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[va], pipe.predict(X[va])))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()

    df, rad_cols = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict):
        splits = splits.get("folds", splits.get("splits", []))
    # normalize split format -> list of {"val":[...]}
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    print(f"[data] {len(df)} patients with T-stage", flush=True)
    print(f"[data] radiomics feature cols: {len(rad_cols)}", flush=True)
    print(f"[data] T distribution: {df['T'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"[data] splits: {len(splits)} folds", flush=True)
    print()

    clin_feats = [c for c in CLIN_CANON if c in df.columns]
    geom_feats = [c for c in GEOM if c in df.columns]

    feature_sets = {
        "geometry_only (baseline)": geom_feats,
        "geometry+clinical":        geom_feats + clin_feats,
        "radiomics_only":           rad_cols,
        "radiomics+geometry+clin":  rad_cols + geom_feats + clin_feats,
    }
    models = {
        "HistGB": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                         learning_rate=0.05, random_state=0),
        "RF":     lambda: RandomForestClassifier(n_estimators=400, max_depth=6,
                                                 random_state=0, n_jobs=-1),
        "LogReg": lambda: LogisticRegression(max_iter=2000, C=0.5),
    }
    Ks = [10, 20, 40]

    print(f"{'feature_set':<28}{'model':<8}{'K':>4}{'CV_bal_acc':>14}{'LOCO':>10}", flush=True)
    print("-" * 66, flush=True)
    results = []
    for fs_name, fcols in feature_sets.items():
        for m_name, m_fn in models.items():
            ks = Ks if len(fcols) > 40 else [min(len(fcols), 40)]
            for k in ks:
                cv, sd = cv_score(df, fcols, splits, m_fn, k)
                loco = loco_score(df, fcols, m_fn, k)
                results.append((fs_name, m_name, k, cv, sd, loco))
                print(f"{fs_name:<28}{m_name:<8}{k:>4}{cv:>10.3f}±{sd:.2f}{loco:>10.3f}", flush=True)

    print()
    base = max(r[3] for r in results if r[0].startswith("geometry_only"))
    best = max(results, key=lambda r: r[3])
    print(f"[baseline] geometry-only best CV balanced accuracy: {base:.3f}", flush=True)
    print(f"[best]     {best[0]} / {best[1]} / K={best[2]}: CV={best[3]:.3f} (LOCO={best[5]:.3f})", flush=True)
    delta = best[3] - base
    print(f"[verdict]  radiomics changes T CV by {delta:+.3f}", flush=True)
    if delta > 0.02 and best[5] >= base - 0.02:
        print("           -> radiomics HELPS and generalizes across centers. Worth shipping.", flush=True)
    elif delta > 0.02:
        print("           -> better in CV but LOCO weak: likely center-overfit. Be cautious.", flush=True)
    else:
        print("           -> radiomics does NOT meaningfully help T. Keep geometry model; pivot to N.", flush=True)


if __name__ == "__main__":
    main()

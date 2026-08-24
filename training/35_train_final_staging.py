#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 35: train the FINAL staging models for the Slot 2 container.

Two validated changes:
  N-model: size-focused LogReg (CV 0.683 / LOCO 0.662) replaces the rule (0.656/0.628)
           features: max_node_mm, gtvn_total_ml, gtvp_ml + laterality one-hot
  T-model: LogReg on CLEAN labels (junk T0/nan dropped) (offline 0.539 vs 0.338 polluted)
           features: gtvp_ml, max_node_mm, gtvn_total_ml + clinical

Both saved as self-contained joblib payloads the container can load and call with a
plain feature dict. Verifies CV scores BEFORE saving (this check caught a wrong-model
bug earlier - keep it).

USAGE (HECKTOR env, batch):
  python 35_train_final_staging.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --splits outputs/eda/splits_2026.json \
    --outdir outputs/models_slot2
"""
import argparse, json, re, os, warnings
print("[boot] importing...", flush=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import joblib
print("[boot] imports done", flush=True)

# ---- EXACT feature sets validated in steps 25 and 27 ----
N_GEOM = ["max_node_mm", "gtvn_total_ml", "gtvp_ml"]
T_GEOM = ["gtvp_ml", "max_node_mm", "gtvn_total_ml"]
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]
LAT_LEVELS = ["unilateral", "bilateral", "midline", "none"]
N_FEATURES = N_GEOM + [f"lat_{lv}" for lv in LAT_LEVELS]
VALID_T = ["1", "2", "3", "4"]
NCLASSES = ["N0", "N1", "N2", "N3"]

# the container reads ehr.json with LONG key names; keep the mapping with the model
CLIN_LONG = {
    "Age": "Age",
    "Gender": "Gender",
    "Tobacco": "Tobacco Consumption",
    "Alcohol": "Alcohol Consumption",
    "Performance": "Performance Status",
    "HPV": "HPV Status",
    "Treatment": "Treatment",
}


def plain_rule(n, maxmm, lat, n3=55.0, n1=35.0):
    if n == 0: return "N0"
    if maxmm > n3: return "N3"
    if n == 1 and lat in ("unilateral", "midline") and maxmm <= n1: return "N1"
    return "N2"


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
    ncol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("nstage", "n")][0]
    tcol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("tstage", "t")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    cmap = {det(clin.columns, k): k for k in CLIN if det(clin.columns, k)}
    keep = {pidc: "PatientID", ncol: "N", tcol: "T_raw", **cmap}
    clin = clin[list(keep)].rename(columns=keep)
    clin["N"] = clin["N"].astype(str).str.upper().str[:2]
    clin["T"] = (clin["T_raw"].astype(str).str.upper()
                 .str.replace("T", "", regex=False).str.strip().str[:1])

    meta = pd.read_csv(args.meta)
    df = clin.merge(meta, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    for lv in LAT_LEVELS:
        df[f"lat_{lv}"] = (df.get("laterality", "none") == lv).astype(int)
    print(f"[load] done ({len(df)} patients)", flush=True)
    return df


def make_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ])


def cv_check(df, feats, label_col, splits):
    """Reproduce the offline CV score before saving (guards against shipping a wrong model)."""
    accs = []
    for f in splits:
        va_ids = set(f["val"])
        tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
        if va.sum() < 5 or tr.sum() < 20: continue
        p = make_pipe().fit(df[feats][tr], df[label_col].values[tr])
        accs.append(balanced_accuracy_score(df[label_col].values[va], p.predict(df[feats][va])))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--outdir", default="outputs/models_slot2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict): splits = splits.get("folds", splits.get("splits", []))
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    # =================================================== N MODEL
    print("\n" + "=" * 60, flush=True)
    print("N-STAGE MODEL (size-focused LogReg)", flush=True)
    print("=" * 60, flush=True)
    dn = df[df["N"].isin(NCLASSES)].copy()
    print(f"[N] {len(dn)} patients | dist {dn['N'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"[N] features: {N_FEATURES}", flush=True)

    n_cv = cv_check(dn, N_FEATURES, "N", splits)
    # rule baseline for comparison
    rule_pred = dn.apply(lambda r: plain_rule(r["n_gtvn"], r["max_node_mm"],
                                              r.get("laterality", "none")), axis=1)
    rule_accs = []
    for f in splits:
        va = dn[dn.PatientID.isin(set(f["val"]))]
        if len(va) < 5: continue
        rule_accs.append(balanced_accuracy_score(va["N"], rule_pred[va.index]))
    n_rule = float(np.mean(rule_accs))
    print(f"[N] CV balanced accuracy: LogReg={n_cv:.3f}  rule={n_rule:.3f}  (expect ~0.683 vs 0.656)", flush=True)
    if n_cv < n_rule:
        print("[N] WARNING: LogReg does NOT beat the rule here. Do not ship without review.", flush=True)

    n_model = make_pipe().fit(dn[N_FEATURES], dn["N"].values)
    n_payload = {
        "model": n_model,
        "features": N_FEATURES,
        "classes": list(n_model.classes_),
        "laterality_levels": LAT_LEVELS,
        "cv_balanced_accuracy": n_cv,
        "rule_baseline": n_rule,
        "note": "size-focused: node COUNT deliberately excluded (noisy on predicted masks)",
    }
    n_path = os.path.join(args.outdir, "n_classifier.joblib")
    joblib.dump(n_payload, n_path)
    print(f"[N] saved -> {n_path}", flush=True)
    print(f"[N] classes: {list(n_model.classes_)}", flush=True)

    # =================================================== T MODEL
    print("\n" + "=" * 60, flush=True)
    print("T-STAGE MODEL (clean labels + LogReg)", flush=True)
    print("=" * 60, flush=True)
    clin_feats = [c for c in CLIN if c in df.columns]
    T_FEATURES = T_GEOM + clin_feats
    dt_all = df.copy()
    dt = dt_all[dt_all["T"].isin(VALID_T)].copy()
    dropped = len(dt_all) - len(dt)
    print(f"[T] dropped {dropped} junk-label rows (T0 / missing) -> {len(dt)} clean patients", flush=True)
    print(f"[T] dist {dt['T'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"[T] features: {T_FEATURES}", flush=True)

    t_cv_clean = cv_check(dt, T_FEATURES, "T", splits)
    t_cv_dirty = cv_check(dt_all, T_FEATURES, "T", splits)
    print(f"[T] CV balanced accuracy: clean={t_cv_clean:.3f}  polluted={t_cv_dirty:.3f}  "
          f"(expect ~0.539 vs ~0.338)", flush=True)

    t_model = make_pipe().fit(dt[T_FEATURES], dt["T"].values)
    t_payload = {
        "model": t_model,
        "features": T_FEATURES,
        "geom_features": T_GEOM,
        "clinical_features": clin_feats,
        "clinical_long_names": {k: CLIN_LONG[k] for k in clin_feats if k in CLIN_LONG},
        "classes": list(t_model.classes_),
        "cv_balanced_accuracy": t_cv_clean,
        "note": "trained on CLEAN labels only (T1-T4); T0/missing rows excluded",
    }
    t_path = os.path.join(args.outdir, "t_classifier.joblib")
    joblib.dump(t_payload, t_path)
    print(f"[T] saved -> {t_path}", flush=True)
    print(f"[T] classes: {list(t_model.classes_)}", flush=True)

    # =================================================== container self-test
    print("\n" + "=" * 60, flush=True)
    print("CONTAINER-STYLE SELF-TEST (predict from a geometry dict)", flush=True)
    print("=" * 60, flush=True)
    demo_geoms = [
        dict(gtvp_ml=12.0, n_gtvn=0, max_node_mm=0.0,  gtvn_total_ml=0.0,  laterality="none"),
        dict(gtvp_ml=8.0,  n_gtvn=1, max_node_mm=24.0, gtvn_total_ml=2.4,  laterality="unilateral"),
        dict(gtvp_ml=20.0, n_gtvn=2, max_node_mm=42.0, gtvn_total_ml=12.5, laterality="bilateral"),
        dict(gtvp_ml=30.0, n_gtvn=1, max_node_mm=66.0, gtvn_total_ml=33.0, laterality="unilateral"),
    ]
    demo_clin = {k: 1.0 for k in clin_feats}
    for g in demo_geoms:
        row = {"gtvp_ml": g["gtvp_ml"], "max_node_mm": g["max_node_mm"],
               "gtvn_total_ml": g["gtvn_total_ml"]}
        for lv in LAT_LEVELS:
            row[f"lat_{lv}"] = 1 if g["laterality"] == lv else 0
        Xn = pd.DataFrame([[row[f] for f in N_FEATURES]], columns=N_FEATURES)
        npred = str(n_model.predict(Xn)[0])
        trow = dict(row); trow.update(demo_clin)
        Xt = pd.DataFrame([[trow.get(f, np.nan) for f in T_FEATURES]], columns=T_FEATURES)
        tpred = str(t_model.predict(Xt)[0])
        print(f"  nodes={g['n_gtvn']} max={g['max_node_mm']:>5.1f}mm lat={g['laterality']:<11}"
              f"-> N={npred}  T=T{tpred}", flush=True)

    print("\n[done] both models saved and callable from a plain feature dict.", flush=True)


if __name__ == "__main__":
    main()

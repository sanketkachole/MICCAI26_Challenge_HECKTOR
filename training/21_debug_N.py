#!/usr/bin/env python3
r"""Diagnostic: why did learned-N score differ between step 18 (0.717) and step 20 (0.576)?
Computes rule and learned model on IDENTICAL rows + splits, prints per-fold, checks
the obvious suspects (feature NaNs, laterality distribution, split membership)."""
import argparse, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
LAT_LEVELS = ["unilateral", "bilateral", "midline", "none"]
FEATURES = GEOM + [f"lat_{lv}" for lv in LAT_LEVELS]

def rule_N(n, maxmm, lat, n3=55.0, n1=35.0):
    if n == 0: return "N0"
    if maxmm > n3: return "N3"
    if n == 1 and lat in ("unilateral", "midline") and maxmm <= n1: return "N1"
    return "N2"

ap = argparse.ArgumentParser()
ap.add_argument("--clinical", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--splits", required=True)
a = ap.parse_args()

clin = pd.read_csv(a.clinical)
pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
ncol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("nstage", "n")][0]
clin[pidc] = clin[pidc].astype(str).str.strip()
clin = clin[[pidc, ncol]].rename(columns={pidc: "PatientID", ncol: "N"})
clin["N"] = clin["N"].astype(str).str.upper().str[:2]
clin = clin[clin["N"].isin(["N0","N1","N2","N3"])]

meta = pd.read_csv(a.meta)
print("meta columns:", list(meta.columns))
print("meta laterality values:", meta["laterality"].value_counts().to_dict() if "laterality" in meta else "NO laterality col")

df = clin.merge(meta, on="PatientID", how="left")
for lv in LAT_LEVELS:
    df[f"lat_{lv}"] = (df.get("laterality","none") == lv).astype(int)

# check for NaNs in features (a merge miss would create NaN -> impute -> garbage)
print("\nNaN count per feature after merge:")
print(df[FEATURES].isna().sum().to_dict())
print("rows with ANY feature NaN:", int(df[FEATURES].isna().any(axis=1).sum()), "/", len(df))

splits = json.load(open(a.splits))
if isinstance(splits, dict): splits = splits.get("folds", splits.get("splits", []))
norm=[]
for f in splits:
    if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
    elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
splits=norm
print(f"\nsplits: {len(splits)} folds, val sizes: {[len(f['val']) for f in splits]}")

# check split coverage: do the val ids actually match our PatientIDs?
all_val = set()
for f in splits: all_val |= set(f["val"])
our_ids = set(df["PatientID"])
print("val ids that match our patients:", len(all_val & our_ids), "/", len(all_val), "val ids total")
print("our patients NOT in any val fold:", len(our_ids - all_val))

def make_pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                              learning_rate=0.05, class_weight="balanced", random_state=0))])

X = df[FEATURES]; y = df["N"].values
print("\n--- per-fold detail ---")
learned, rule = [], []
for i, f in enumerate(splits):
    va_ids = set(f["val"])
    tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
    if va.sum() == 0:
        print(f"fold {i}: EMPTY val (no matching patients!)"); continue
    p = make_pipe().fit(X[tr], y[tr])
    lp = p.predict(X[va]); la = balanced_accuracy_score(y[va], lp)
    rp = df[va].apply(lambda r: rule_N(r["n_gtvn"], r["max_node_mm"], r.get("laterality","none")), axis=1)
    ra = balanced_accuracy_score(y[va], rp)
    learned.append(la); rule.append(ra)
    print(f"fold {i}: n_tr={tr.sum()} n_va={va.sum()} | learned={la:.3f} rule={ra:.3f} | "
          f"val N dist={pd.Series(y[va]).value_counts().sort_index().to_dict()}")
print(f"\nMEAN: learned={np.mean(learned):.3f}  rule={np.mean(rule):.3f}")

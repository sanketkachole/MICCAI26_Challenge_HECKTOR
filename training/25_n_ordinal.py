#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 25 (Phase 1): improved N-staging using the Phase-0 insight that
NODE SIZE (not count) separates N-stages, and errors are adjacent-stage boundary
mistakes -> use ordinal + cost-sensitive modeling.

Approaches compared (same 5-fold + LOCO honesty guards):
  0. rule (baseline, 0.656)
  1. LogReg all-geom (our prior best, 0.680)
  2. LogReg size-focused (drop noisy n_gtvn; emphasize size/volume)
  3. Ordinal LogReg (3 binary "N>=k" models, size-focused) - respects N0<N1<N2<N3
  4. Ordinal LogReg + tuned decision thresholds on max_node_mm
  5. Ordinal + clinical (add HPV etc.)

Also does a quick GRID SEARCH on the rule's size thresholds (n1, n3) to see how much
of the gap is just badly-placed cutoffs.

REPORT ONLY.
USAGE (HECKTOR env):
  python 25_n_ordinal.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --splits outputs/eda/splits_2026.json
"""
import argparse, json, re, warnings
print("[boot] importing numpy/pandas...", flush=True)
import numpy as np, pandas as pd
print("[boot] importing sklearn...", flush=True)
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
print("[boot] imports done", flush=True)

GEOM_ALL = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
GEOM_SIZE = ["max_node_mm", "gtvn_total_ml", "gtvp_ml"]   # size-focused: drop noisy n_gtvn
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]
LAT_LEVELS = ["unilateral", "bilateral", "midline", "none"]
NCLASSES = ["N0", "N1", "N2", "N3"]
NIDX = {c: i for i, c in enumerate(NCLASSES)}


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
    print("[load] reading clinical CSV...", flush=True)
    clin = pd.read_csv(args.clinical)
    print(f"[load] clinical OK ({len(clin)} rows), reading meta CSV...", flush=True)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    ncol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("nstage", "n")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    cmap = {det(clin.columns, k): k for k in CLIN if det(clin.columns, k)}
    keep = {pidc: "PatientID", ncol: "N", **cmap}
    clin = clin[list(keep)].rename(columns=keep)
    clin["N"] = clin["N"].astype(str).str.upper().str[:2]
    clin = clin[clin["N"].isin(NCLASSES)]

    meta = pd.read_csv(args.meta)
    print(f"[load] meta OK ({len(meta)} rows), merging...", flush=True)
    df = clin.merge(meta, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    for lv in LAT_LEVELS:
        df[f"lat_{lv}"] = (df.get("laterality", "none") == lv).astype(int)
    print(f"[load] done ({len(df)} patients)", flush=True)
    return df


# ---------- Ordinal classifier: 3 binary "N >= k" models ----------
class OrdinalLogReg:
    """Frank & Hall ordinal: for k in {1,2,3} fit P(N>=k). Predict by summing."""
    def __init__(self, C=0.5, class_weight="balanced"):
        self.C = C; self.class_weight = class_weight; self.models = {}
    def fit(self, X, y_idx):
        self.models = {}
        for k in (1, 2, 3):
            yk = (y_idx >= k).astype(int)
            if len(np.unique(yk)) < 2:
                self.models[k] = None
            else:
                m = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000, C=self.C,
                                       class_weight=self.class_weight))])
                m.fit(X, yk); self.models[k] = m
        return self
    def predict_idx(self, X):
        p_ge = np.zeros((X.shape[0], 4))
        p_ge[:, 0] = 1.0
        for k in (1, 2, 3):
            if self.models[k] is not None:
                p_ge[:, k] = self.models[k].predict_proba(X)[:, 1]
            else:
                p_ge[:, k] = 0.0
        # P(N=k) = P(>=k) - P(>=k+1); class = argmax
        p = np.zeros((X.shape[0], 4))
        for k in range(4):
            hi = p_ge[:, k+1] if k < 3 else 0.0
            p[:, k] = np.clip(p_ge[:, k] - hi, 0, None)
        return p.argmax(1)


def rule_predict_vec(df, n1=35.0, n3=55.0):
    """Vectorized plain-rule prediction over the whole dataframe (fast, no df.apply).
    Priority order matches the original exactly: n==0 -> N0 wins over all; else
    mm>n3 -> N3; else single unilateral/midline small -> N1; else N2."""
    n = df["n_gtvn"].values
    mm = df["max_node_mm"].values
    lat = df.get("laterality", pd.Series(["none"]*len(df))).values
    out = np.empty(len(df), dtype=object)
    out[:] = "N2"                                  # default (else branch)
    is_n1 = (n == 1) & np.isin(lat, ["unilateral", "midline"]) & (mm <= n1)
    out[is_n1] = "N1"                              # 3rd priority
    out[mm > n3] = "N3"                            # 2nd priority (overrides N1/N2)
    out[n == 0] = "N0"                             # 1st priority (overrides everything)
    return out


def eval_rule_vec(df, splits, n1=35.0, n3=55.0, use_loco=False):
    pred = rule_predict_vec(df, n1, n3)
    dfp = df.copy(); dfp["_p"] = pred
    groups = ([{"val": list(dfp[dfp.center == c]["PatientID"])} for c in sorted(dfp.center.unique())]
              if use_loco else splits)
    accs = []
    for g in groups:
        va = dfp[dfp.PatientID.isin(set(g["val"]))]
        if len(va) < 5: continue
        accs.append(balanced_accuracy_score(va["N"], va["_p"]))
    return float(np.mean(accs)) if accs else float("nan")


def eval_sklearn(df, feats, splits, model_ctor, use_loco=False, ordinal=False):
    groups = ([{"val": list(df[df.center == c]["PatientID"])} for c in sorted(df.center.unique())]
              if use_loco else splits)
    y_idx = df["N"].map(NIDX).values
    X = df[feats]
    accs = []
    for g in groups:
        va_ids = set(g["val"])
        tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
        if va.sum() < 5 or tr.sum() < 20: continue
        if ordinal:
            m = model_ctor(); m.fit(X[tr], y_idx[tr])
            pred = m.predict_idx(X[va])
        else:
            m = model_ctor(); m.fit(X[tr], df["N"].values[tr])
            pred_lbl = m.predict(X[va]); pred = pd.Series(pred_lbl).map(NIDX).values
        accs.append(balanced_accuracy_score(y_idx[va], pred))
    return float(np.mean(accs)) if accs else float("nan")


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

    print(f"[data] {len(df)} patients | N dist: {df['N'].value_counts().sort_index().to_dict()}", flush=True)
    print(flush=True)

    lat_oh = [c for c in df.columns if c.startswith("lat_")]
    clin_feats = [c for c in CLIN if c in df.columns]

    def logreg(): return Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("sc", StandardScaler()),
                                   ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])

    print(f"{'approach':<34}{'CV':>10}{'LOCO':>10}", flush=True)
    print("-" * 54, flush=True)

    # 0. rule (vectorized)
    r_cv = eval_rule_vec(df, splits)
    r_lo = eval_rule_vec(df, splits, use_loco=True)
    print(f"{'0. rule (baseline)':<34}{r_cv:>10.3f}{r_lo:>10.3f}", flush=True)

    # 1. LogReg all geom + laterality
    f1 = GEOM_ALL + lat_oh
    print(f"{'1. LogReg all-geom':<34}{eval_sklearn(df,f1,splits,logreg):>10.3f}{eval_sklearn(df,f1,splits,logreg,True):>10.3f}", flush=True)

    # 2. LogReg size-focused (drop n_gtvn) + laterality
    f2 = GEOM_SIZE + lat_oh
    print(f"{'2. LogReg size-focused':<34}{eval_sklearn(df,f2,splits,logreg):>10.3f}{eval_sklearn(df,f2,splits,logreg,True):>10.3f}", flush=True)

    # 3. Ordinal LogReg size-focused
    f3 = GEOM_SIZE + lat_oh
    print(f"{'3. Ordinal LogReg size':<34}{eval_sklearn(df,f3,splits,OrdinalLogReg,ordinal=True):>10.3f}"
          f"{eval_sklearn(df,f3,splits,OrdinalLogReg,True,ordinal=True):>10.3f}", flush=True)

    # 4. Ordinal LogReg size + clinical
    f4 = GEOM_SIZE + lat_oh + clin_feats
    print(f"{'4. Ordinal size + clinical':<34}{eval_sklearn(df,f4,splits,OrdinalLogReg,ordinal=True):>10.3f}"
          f"{eval_sklearn(df,f4,splits,OrdinalLogReg,True,ordinal=True):>10.3f}", flush=True)

    # 5. Ordinal all-geom + clinical
    f5 = GEOM_ALL + lat_oh + clin_feats
    print(f"{'5. Ordinal all-geom + clin':<34}{eval_sklearn(df,f5,splits,OrdinalLogReg,ordinal=True):>10.3f}"
          f"{eval_sklearn(df,f5,splits,OrdinalLogReg,True,ordinal=True):>10.3f}", flush=True)

    # ---- grid search on rule thresholds (vectorized, fast) ----
    print("\n[grid] tuning rule size thresholds (n1, n3) on CV:", flush=True)
    best = (r_cv, 35, 55)
    for n1 in range(20, 45, 5):
        for n3 in range(45, 75, 5):
            cv = eval_rule_vec(df, splits, n1=n1, n3=n3)
            if cv > best[0]: best = (cv, n1, n3)
    print(f"    best rule: n1={best[1]} n3={best[2]} -> CV={best[0]:.3f} (was 0.656 at 35/55)", flush=True)

    print("\n[verdict] compare all rows above to rule 0.656 and prior LogReg 0.680.", flush=True)
    print("          ship the best that also holds on LOCO.", flush=True)


if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 04: Prognosis / RFS risk (radiomics-free version)
Target : RFS (time, days) + Relapse (event). Rows with missing outcome are dropped.
Features: mask geometry (tumor + node burden) + clinical + nodal stage (rule_N ordinal).
Model  : Cox proportional hazards (lifelines), ridge-penalized, missing handled by
         median impute + missing-indicator. Outputs a RISK score (higher = worse).
Eval   : out-of-fold Harrell C-index (our 5 folds) + leave-one-center-out.
Saves  : outputs/models/prognosis_model.joblib (+ small json report) for inference.

USAGE
  python 04_prognosis.py --clinical CLIN.csv --meta case_metadata.csv \
      --splits splits_2026.json --out outputs/models
"""
import argparse, json, os, re
import numpy as np, pandas as pd, joblib

COLUMN_PATTERNS = {
    "PatientID": [r"^patient.?id$", r"^id$"],
    "Age": [r"^age$"], "Gender": [r"^gender$", r"^sex$"],
    "Tobacco": [r"tobacco", r"^smok"], "Alcohol": [r"alcohol"],
    "Performance": [r"performance", r"^ecog$"], "HPV": [r"hpv"],
    "Treatment": [r"treatment"], "Nstage": [r"^n.?stage$"],
    "Relapse": [r"relapse", r"recurrence", r"^event$"],
    "RFS": [r"^rfs$", r"survival", r"time.?to", r"^time$"],
}
NUM_CLIN = ["Age"]
CAT_CLIN = ["Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]
GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
N_ORDINAL = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}

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

def build_raw_features(df, cmap):
    """Fixed-column raw feature frame used at BOTH train and inference time."""
    X = pd.DataFrame(index=df.index)
    for g in GEOM:
        X[g] = pd.to_numeric(df[g], errors="coerce") if g in df else np.nan
    # nodal stage ordinal from rule_N (mask-derived; available at inference)
    rn = df["rule_N"].astype(str).str.upper().str[:2] if "rule_N" in df else pd.Series(index=df.index, dtype=object)
    X["N_ordinal"] = rn.map(N_ORDINAL)
    X["Age"] = pd.to_numeric(df[cmap["Age"]], errors="coerce") if cmap.get("Age") in df else np.nan
    for c in CAT_CLIN:
        src = cmap.get(c)
        X[c] = df[src] if (src and src in df) else np.nan
    return X

def concordance(time, risk, event):
    """Harrell's C for right-censored data. Higher risk should mean shorter time."""
    time = np.asarray(time, float); risk = np.asarray(risk, float); event = np.asarray(event, int)
    n = len(time); num = den = 0.0
    for i in range(n):
        if event[i] != 1:  # i must have an event and be the earlier time
            continue
        for j in range(n):
            if time[j] > time[i]:            # comparable: j outlived i
                den += 1
                if risk[i] > risk[j]: num += 1
                elif risk[i] == risk[j]: num += 0.5
    return num / den if den > 0 else float("nan")

def apply_prep(state, X):
    """Rebuild the model input matrix from a saved state dict (used at INFERENCE)."""
    Z = pd.DataFrame(index=X.index)
    base_cols = [c for c in state["feat_order"] if not c.endswith("_missing")]
    for c in base_cols:
        s = X[c] if c in X else pd.Series(np.nan, index=X.index)
        if c in state["cat_maps"]:
            cat = pd.Categorical(s.astype(str), categories=state["cat_maps"][c])
            s = pd.Series(cat.codes, index=X.index).replace({-1: np.nan})
        else:
            s = pd.to_numeric(s, errors="coerce")
        if c + "_missing" in state["feat_order"]:
            Z[c + "_missing"] = s.isna().astype(float)
        s = s.fillna(state["medians"][c])
        Z[c] = (s - state["mu"][c]) / state["sd"][c]
    return Z[state["feat_order"]]


def predict_risk_from_state(state, X):
    """Inference entry point: higher = higher risk. No custom class needed."""
    return state["cph"].predict_partial_hazard(apply_prep(state, X)).values


class PrognosisModel:
    """Training-time convenience wrapper. Saved as plain state (see .state())."""
    def __init__(self, penalizer=0.1):
        self.penalizer = penalizer
    def fit(self, X, time, event):
        from lifelines import CoxPHFitter
        cat_maps, medians, mu, sd = {}, {}, {}, {}
        Z = pd.DataFrame(index=X.index)
        for c in X.columns:
            s = X[c]
            if s.dtype.kind not in "biufc":
                cats = pd.Categorical(s.astype(str).replace({"nan": np.nan}))
                cat_maps[c] = list(cats.categories)
                s = pd.Series(cats.codes, index=X.index).replace({-1: np.nan})
            if s.isna().any():
                Z[c + "_missing"] = s.isna().astype(float)
            med = float(s.median()) if s.notna().any() else 0.0
            medians[c] = med; s = s.fillna(med)
            m, d = float(s.mean()), float(s.std() or 1.0)
            mu[c], sd[c] = m, d
            Z[c] = (s - m) / d
        keep = [c for c in Z.columns if Z[c].std() > 1e-9]  # Cox dislikes constants
        Z = Z[keep]
        dd = Z.copy(); dd["_t"] = np.asarray(time, float); dd["_e"] = np.asarray(event, int)
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(dd, duration_col="_t", event_col="_e")
        self._state = {"cat_maps": cat_maps, "medians": medians, "mu": mu, "sd": sd,
                       "feat_order": keep, "cph": cph}
        return self
    def state(self):
        return self._state
    def predict_risk(self, X):
        return predict_risk_from_state(self._state, X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="outputs/models")
    ap.add_argument("--penalizer", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    clin = pd.read_csv(args.clinical); meta = pd.read_csv(args.meta)
    cmap = detect_columns(list(clin.columns)); pid_c = cmap["PatientID"]
    clin[pid_c] = clin[pid_c].astype(str).str.strip()
    ren = {get_meta_col(meta, "n_gtvn", "geom_n_nodes"): "n_gtvn",
           get_meta_col(meta, "max_node_mm", "geom_max_node_mm"): "max_node_mm",
           get_meta_col(meta, "gtvp_ml", "geom_gtvp_ml"): "gtvp_ml",
           get_meta_col(meta, "gtvn_total_ml", "geom_total_vol_ml"): "gtvn_total_ml",
           get_meta_col(meta, "rule_N"): "rule_N", get_meta_col(meta, "PatientID"): "PatientID"}
    meta = meta.rename(columns={k: v for k, v in ren.items() if k})
    meta["PatientID"] = meta["PatientID"].astype(str).str.strip()
    df = clin.merge(meta, left_on=pid_c, right_on="PatientID", how="inner")
    df["center"] = df["PatientID"].str.split("-").str[0]

    time = pd.to_numeric(df[cmap["RFS"]], errors="coerce")
    event = pd.to_numeric(df[cmap["Relapse"]], errors="coerce")
    ok = time.notna() & event.notna() & (time > 0)
    df, time, event = df[ok].copy(), time[ok].values, event[ok].values.astype(int)
    print(f"[data] {len(df)} patients with outcome (dropped {(~ok).sum()} missing/invalid); "
          f"event rate {event.mean():.3f}")

    X = build_raw_features(df, cmap)
    splits = json.load(open(args.splits))
    pid_to_fold = {str(p): f["fold"] for f in splits["folds"] for p in f["val"]}
    fold = df["PatientID"].map(pid_to_fold).values

    # out-of-fold risk
    oof = np.full(len(df), np.nan)
    for fo in sorted(pd.Series(fold).dropna().unique()):
        tr = np.where(fold != fo)[0]; va = np.where(fold == fo)[0]
        if len(tr) < 20 or len(va) == 0: continue
        m = PrognosisModel(args.penalizer).fit(X.iloc[tr], time[tr], event[tr])
        oof[va] = m.predict_risk(X.iloc[va])
    m_ok = ~np.isnan(oof)
    c_oof = concordance(time[m_ok], oof[m_ok], event[m_ok])
    print(f"\n[C-index] out-of-fold (5-fold): {c_oof:.3f}")

    # leave-one-center-out
    print("[C-index] leave-one-center-out:")
    for c in sorted(df["center"].unique()):
        te = df["center"].values == c; trm = ~te
        if te.sum() < 10 or event[trm].sum() < 5: continue
        m = PrognosisModel(args.penalizer).fit(X.iloc[np.where(trm)[0]], time[trm], event[trm])
        r = m.predict_risk(X.iloc[np.where(te)[0]])
        print(f"    hold-out {c:5s} n={te.sum():3d}  C={concordance(time[te], r, event[te]):.3f}")

    # final model on all data -> save PLAIN STATE (no custom class to import at inference)
    final = PrognosisModel(args.penalizer).fit(X, time, event)
    joblib.dump({"state": final.state(), "clinical_colmap": cmap,
                 "feature_spec": {"GEOM": GEOM, "CAT_CLIN": CAT_CLIN, "N_ORDINAL": N_ORDINAL}},
                os.path.join(args.out, "prognosis_model.joblib"))
    json.dump({"c_index_oof": round(float(c_oof), 4), "n": int(len(df)),
               "event_rate": round(float(event.mean()), 4)},
              open(os.path.join(args.out, "prognosis_report.json"), "w"), indent=2)
    print(f"\n[save] {args.out}/prognosis_model.joblib")
    print(f"[save] {args.out}/prognosis_report.json")

if __name__ == "__main__":
    main()
    
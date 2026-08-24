#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 16: test whether radiomics improves prognosis (RFS C-index),
with strict overfitting guards, compared to the current geometry+clinical Cox model
(leaderboard C-index ~0.789).

Guards (Cox overfits easily with many features -> essential):
  1. Feature preselection by univariate Cox concordance on TRAIN ONLY (inside each fold).
  2. Honest 5-fold CV (splits_2026.json) + leave-one-center-out (LOCO).
  3. Small K only (Cox with 428 features would memorize). Compare K in {0(baseline),5,10,15}.
  4. Baseline = geometry + clinical + N-ordinal (what we ship now).

REPORT ONLY. Decide what to ship after seeing numbers.

USAGE:
  python 16_prognosis_radiomics.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --radiomics outputs/eda/radiomics_pred.csv \
    --splits outputs/eda/splits_2026.json
"""
import argparse, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
CLIN_CANON = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]
N_ORD = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}


def detect(cols, canon):
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


def load(args):
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    rfs = detect(clin.columns, "RFS")
    rel = detect(clin.columns, "Relapse")
    keep = {pidc: "PatientID", rfs: "RFS", rel: "event"}
    cmap = {k: detect(clin.columns, k) for k in CLIN_CANON}
    for k, v in cmap.items():
        if v: keep[v] = k
    df = clin[list(keep)].rename(columns=keep)
    df = df.dropna(subset=["RFS", "event"])
    df = df[df["RFS"] > 0]

    meta = pd.read_csv(args.meta)[["PatientID"] + GEOM]
    # N-ordinal from predicted-mask rule (recompute simple rule to match container)
    m2 = pd.read_csv(args.meta)
    if "laterality" in m2:
        def rule_n(r):
            if r["n_gtvn"] == 0: return 0
            if r["max_node_mm"] > 55: return 3
            if r["n_gtvn"] == 1 and r.get("laterality") in ("unilateral", "midline") and r["max_node_mm"] <= 35: return 1
            return 2
        meta = meta.merge(m2[["PatientID", "laterality"]], on="PatientID", how="left")
        meta["N_ordinal"] = meta.apply(rule_n, axis=1)
        meta = meta.drop(columns=["laterality"])
    else:
        meta["N_ordinal"] = 0

    rad = pd.read_csv(args.radiomics)
    rad_cols = [c for c in rad.columns if c.split("_")[0] in ("ct", "pt") and not c.startswith("err_")]
    rad = rad[["PatientID"] + rad_cols]

    df = df.merge(meta, on="PatientID", how="left").merge(rad, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    return df, rad_cols, [k for k in CLIN_CANON if k in df.columns]


def prep(Xtr, Xva):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return (pd.DataFrame(sc.transform(imp.transform(Xtr)), columns=Xtr.columns, index=Xtr.index),
            pd.DataFrame(sc.transform(imp.transform(Xva)), columns=Xva.columns, index=Xva.index))


def preselect_cox(Xtr, dur, evt, k):
    """Rank features by univariate concordance with survival on TRAIN, fast.
    Uses concordance_index of each single feature vs survival time (equivalent
    ranking to univariate Cox but ~100x faster: no iterative Cox fit per feature)."""
    dur_v = dur.values; evt_v = evt.values
    scores = []
    for c in Xtr.columns:
        x = Xtr[c].values
        if np.all(x == x[0]) or np.isnan(x).all():
            scores.append((0.0, c)); continue
        try:
            # higher feature value could mean higher OR lower risk; take distance from 0.5
            ci = concordance_index(dur_v, x, evt_v)
            scores.append((abs(ci - 0.5), c))
        except Exception:
            scores.append((0.0, c))
    scores.sort(reverse=True)
    return [c for _, c in scores[:k]]


def eval_config(df, base_feats, rad_cols, splits, k_rad, use_loco=False):
    """Return mean C-index over folds (or centers if use_loco)."""
    groups = ([{"val": list(df[df.center == c]["PatientID"])} for c in sorted(df.center.unique())]
              if use_loco else splits)
    cis = []
    for g in groups:
        va_ids = set(g["val"])
        tr = df[~df.PatientID.isin(va_ids)]
        va = df[df.PatientID.isin(va_ids)]
        if len(va) < 8 or len(tr) < 30 or va["event"].sum() < 3:
            continue
        feats = list(base_feats)
        if k_rad > 0:
            sel = preselect_cox(tr[rad_cols].fillna(tr[rad_cols].median()),
                                tr["RFS"], tr["event"], k_rad)
            feats = feats + sel
        Xtr, Xva = prep(tr[feats], va[feats])
        dtr = Xtr.copy(); dtr["T"] = tr["RFS"].values; dtr["E"] = tr["event"].values
        try:
            cph = CoxPHFitter(penalizer=0.5)
            cph.fit(dtr, "T", "E")
            risk = cph.predict_partial_hazard(Xva).values
            ci = concordance_index(va["RFS"].values, -risk, va["event"].values)
            cis.append(ci)
        except Exception:
            continue
    return float(np.mean(cis)) if cis else float("nan"), len(cis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()

    df, rad_cols, clin_feats = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict):
        splits = splits.get("folds", splits.get("splits", []))
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    base = [c for c in GEOM if c in df.columns] + clin_feats + ["N_ordinal"]
    print(f"[data] {len(df)} patients with RFS+event ({int(df['event'].sum())} events)", flush=True)
    print(f"[data] radiomics cols: {len(rad_cols)} | baseline feats: {len(base)}", flush=True)
    print(f"[data] splits: {len(splits)} folds", flush=True)
    print()
    print(f"{'config':<34}{'CV_Cindex':>12}{'(folds)':>9}{'LOCO':>10}", flush=True)
    print("-" * 65, flush=True)

    configs = [("baseline (geom+clin+N)", 0),
               ("baseline + 5 radiomics", 5),
               ("baseline + 10 radiomics", 10),
               ("baseline + 15 radiomics", 15)]
    rows = []
    for name, k in configs:
        cv, nf = eval_config(df, base, rad_cols, splits, k, use_loco=False)
        loco, _ = eval_config(df, base, rad_cols, splits, k, use_loco=True)
        rows.append((name, k, cv, loco))
        print(f"{name:<34}{cv:>12.3f}{nf:>9}{loco:>10.3f}", flush=True)

    print()
    base_cv = rows[0][2]
    best = max(rows, key=lambda r: (r[2] if not np.isnan(r[2]) else -1))
    print(f"[baseline]  CV C-index: {base_cv:.3f}", flush=True)
    print(f"[best]      {best[0]}: CV={best[2]:.3f}, LOCO={best[3]:.3f}", flush=True)
    d = best[2] - base_cv
    print(f"[verdict]   radiomics changes prognosis CV C-index by {d:+.3f}", flush=True)
    if d > 0.01 and best[3] >= rows[0][3] - 0.01:
        print("            -> radiomics HELPS prognosis and generalizes. Worth shipping.", flush=True)
    elif d > 0.01:
        print("            -> better CV but LOCO weaker: center-overfit risk. Cautious.", flush=True)
    else:
        print("            -> radiomics does NOT help prognosis. Keep current Cox model.", flush=True)


if __name__ == "__main__":
    main()
    
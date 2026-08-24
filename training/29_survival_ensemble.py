#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 29 (Phase 2): survival ensemble via rank-averaging.

Tests whether rank-averaging DECORRELATED survival models beats single Cox:
  models: Cox (sksurv), Random Survival Forest, Gradient-Boosted Survival, BaggedICARE
  ensemble: average the RANKS of each model's risk -> combined risk

Also tests better features (research-backed, NOT texture which overfit):
  - HPV-missing as its own binary signal (24% missing, informative)
  - PET intensity: SUVmean/SUVmax for GTVp & GTVn, and TLG = volume x SUVmean
    (metabolic features are prognostic and less scanner-sensitive than texture)

Same 5-fold + LOCO honesty guards. REPORT ONLY.

Runs in the `icare` env (sksurv 0.22.2 + icare + sklearn 1.3.2).
USAGE (batch job):
  python 29_survival_ensemble.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --radiomics outputs/eda/radiomics_pred.csv \
    --splits outputs/eda/splits_2026.json
"""
import argparse, json, re, warnings
print("[boot] importing...", flush=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from icare.survival import BaggedIcareSurvival, harrell_cindex
print("[boot] imports done", flush=True)

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]


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
        if canon == "RFS" and cl in ("rfs", "survival", "time"): return c
        if canon == "Relapse" and cl in ("relapse", "event", "recurrence"): return c
    return None


def make_y(dur, evt):
    return np.array(list(zip(evt.astype(bool), dur.astype(float))),
                    dtype=[('event', '?'), ('time', '<f8')])


def find_rad(cols, modality, region, stat):
    """Find a radiomics column like pt_gtvp_firstorder_Mean (robust to naming)."""
    for c in cols:
        cl = c.lower()
        if cl.startswith(f"{modality}_{region}_") and "firstorder" in cl and stat.lower() in cl:
            return c
    return None


def load(args):
    print("[load] reading clinical...", flush=True)
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    rfs = det(clin.columns, "RFS"); rel = det(clin.columns, "Relapse")
    keep = {pidc: "PatientID", rfs: "RFS", rel: "event"}
    for k in CLIN:
        v = det(clin.columns, k)
        if v: keep[v] = k
    df = clin[list(keep)].rename(columns=keep)
    df = df.dropna(subset=["RFS", "event"]); df = df[df["RFS"] > 0]
    # HPV-missing as its own signal
    if "HPV" in df.columns:
        df["HPV_missing"] = df["HPV"].isna().astype(int)

    print("[load] reading meta...", flush=True)
    meta = pd.read_csv(args.meta)
    geomcols = [c for c in GEOM if c in meta.columns]
    mkeep = ["PatientID"] + geomcols + (["laterality"] if "laterality" in meta.columns else [])
    meta = meta[mkeep]
    if "laterality" in meta.columns:
        def rn(r):
            if r["n_gtvn"] == 0: return 0
            if r["max_node_mm"] > 55: return 3
            if r["n_gtvn"] == 1 and r["laterality"] in ("unilateral", "midline") and r["max_node_mm"] <= 35: return 1
            return 2
        meta["N_ordinal"] = meta.apply(rn, axis=1)
        meta = meta.drop(columns=["laterality"])

    print("[load] reading radiomics (PET intensity only)...", flush=True)
    rad = pd.read_csv(args.radiomics)
    # pull PET SUVmean/max/energy for gtvp and gtvn
    pet_feats = {}
    for region in ("gtvp", "gtvn"):
        for stat in ("Mean", "Maximum", "Energy"):
            col = find_rad(rad.columns, "pt", region, stat)
            if col: pet_feats[f"SUV_{region}_{stat}"] = col
    rkeep = ["PatientID"] + list(pet_feats.values())
    rad_small = rad[rkeep].rename(columns={v: k for k, v in pet_feats.items()})

    df = df.merge(meta, on="PatientID", how="left").merge(rad_small, on="PatientID", how="left")
    # TLG proxies = volume x SUVmean
    if "SUV_gtvp_Mean" in df.columns and "gtvp_ml" in df.columns:
        df["TLG_gtvp"] = df["gtvp_ml"] * df["SUV_gtvp_Mean"]
    if "SUV_gtvn_Mean" in df.columns and "gtvn_total_ml" in df.columns:
        df["TLG_gtvn"] = df["gtvn_total_ml"] * df["SUV_gtvn_Mean"]

    df["center"] = df["PatientID"].str.split("-").str[0]
    print(f"[load] done ({len(df)} patients, {int(df['event'].sum())} events)", flush=True)
    return df, list(pet_feats.keys())


def prep(Xtr, Xva):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva))


def model_ctors():
    return {
        "Cox":  lambda: CoxPHSurvivalAnalysis(alpha=1.0),
        "RSF":  lambda: RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                             max_features="sqrt", random_state=0, n_jobs=4),
        "GBSA": lambda: GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                                         max_depth=3, random_state=0),
        "bICARE": lambda: BaggedIcareSurvival(n_estimators=40, aggregation_method="median",
                                              random_state=0, n_jobs=4),
    }


def eval_all(df, feats, splits, use_loco=False):
    """Fit each model per fold; collect per-model C-index and the rank-ensemble C-index."""
    groups = ([{"val": list(df[df.center == c]["PatientID"])} for c in sorted(df.center.unique())]
              if use_loco else splits)
    names = list(model_ctors().keys())
    per_model = {n: [] for n in names}
    ens_scores = []
    for g in groups:
        va_ids = set(g["val"])
        tr = df[~df.PatientID.isin(va_ids)]; va = df[df.PatientID.isin(va_ids)]
        if len(va) < 8 or len(tr) < 30 or va["event"].sum() < 3:
            continue
        Xtr, Xva = prep(tr[feats], va[feats])
        ytr = make_y(tr["RFS"], tr["event"]); yva = make_y(va["RFS"], va["event"])
        risks = {}
        for n, ctor in model_ctors().items():
            try:
                m = ctor(); m.fit(Xtr, ytr)
                r = np.asarray(m.predict(Xva)).ravel()
                risks[n] = r
                per_model[n].append(float(harrell_cindex(yva, r)))
            except Exception as e:
                print(f"    [warn] {n} failed on a fold: {str(e)[:60]}", flush=True)
        # rank-ensemble: average ranks of available models
        if len(risks) >= 2:
            ranks = np.vstack([rankdata(risks[n]) for n in risks])
            ens = ranks.mean(0)
            ens_scores.append(float(harrell_cindex(yva, ens)))
    out = {n: (float(np.mean(v)) if v else float("nan")) for n, v in per_model.items()}
    out["ENSEMBLE"] = float(np.mean(ens_scores)) if ens_scores else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()

    df, pet_feats = load(args)
    splits = json.load(open(args.splits))
    if isinstance(splits, dict): splits = splits.get("folds", splits.get("splits", []))
    norm = []
    for f in splits:
        if isinstance(f, dict) and "val" in f: norm.append({"val": f["val"]})
        elif isinstance(f, dict) and "validation" in f: norm.append({"val": f["validation"]})
    splits = norm

    base = [c for c in GEOM if c in df.columns] + [k for k in CLIN if k in df.columns]
    if "N_ordinal" in df.columns: base += ["N_ordinal"]
    if "HPV_missing" in df.columns: base += ["HPV_missing"]
    pet_avail = [c for c in (pet_feats + ["TLG_gtvp", "TLG_gtvn"]) if c in df.columns]

    print(f"\n[data] base feats ({len(base)}): {base}", flush=True)
    print(f"[data] PET intensity feats ({len(pet_avail)}): {pet_avail}", flush=True)
    print(f"[data] splits: {len(splits)} folds", flush=True)

    feature_sets = {
        "base (geom+clin+N)":      base,
        "base + PET intensity":    base + pet_avail,
    }

    for fs_name, feats in feature_sets.items():
        print("\n" + "=" * 60, flush=True)
        print(f"FEATURE SET: {fs_name}", flush=True)
        print("=" * 60, flush=True)
        cv = eval_all(df, feats, splits, use_loco=False)
        lo = eval_all(df, feats, splits, use_loco=True)
        print(f"{'model':<12}{'CV_Cindex':>12}{'LOCO':>10}", flush=True)
        print("-" * 34, flush=True)
        for n in ["Cox", "RSF", "GBSA", "bICARE", "ENSEMBLE"]:
            print(f"{n:<12}{cv.get(n, float('nan')):>12.3f}{lo.get(n, float('nan')):>10.3f}", flush=True)

    print("\n[verdict] Cox is our current model. Ship the ENSEMBLE only if it beats", flush=True)
    print("          Cox on BOTH CV and LOCO by a clear margin. Otherwise keep Cox.", flush=True)


if __name__ == "__main__":
    main()

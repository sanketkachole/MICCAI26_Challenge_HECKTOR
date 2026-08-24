#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 23: compare N-staging approaches (Priority 2 & 3).

Three approaches, same 5-fold + LOCO honesty guards:
  1. PLAIN RULE (current container): node count/size/laterality thresholds.
  2. HPV-CONDITIONED RULE: AJCC 8th-ed. For HPV+ oropharyngeal, node COUNT does not
     matter (N1=ipsilateral<=6cm, N2=bilateral/contralateral<=6cm, N3=any>6cm).
     For HPV- or unknown HPV, fall back to the plain rule.
  3. LEARNED LogReg classifier (validated earlier at CV 0.680).

Also reports a 4th: HPV-rule but pick, per patient, whichever of rule/HPV-rule the data
supports -> not usable at test time, shown only as an upper-bound sanity check.

Decision output: which approach to ship for Slot 2.

Runs in HECKTOR env (needs sklearn; no icare/lifelines).
USAGE:
  python 23_hpv_nrule.py \
    --clinical ".../HECKTOR_2026_training_data.csv" \
    --meta outputs/eda/case_metadata_pred.csv \
    --splits outputs/eda/splits_2026.json
"""
import argparse, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
LAT_LEVELS = ["unilateral", "bilateral", "midline", "none"]


def plain_rule(n, maxmm, lat, n3=55.0, n1=35.0):
    """Current container rule."""
    if n == 0: return "N0"
    if maxmm > n3: return "N3"
    if n == 1 and lat in ("unilateral", "midline") and maxmm <= n1: return "N1"
    return "N2"


def hpv_rule(n, maxmm, lat, hpv, n3=60.0, n1=35.0):
    """AJCC 8th-ed HPV-conditioned N.
    HPV+ (hpv==1): count-independent.
        N0 = no nodes; N3 = any node >6cm (60mm);
        N1 = unilateral/midline nodes <=6cm; N2 = bilateral/contralateral <=6cm.
    HPV- or unknown: plain rule."""
    if hpv == 1:
        if n == 0: return "N0"
        if maxmm > n3: return "N3"
        if lat == "bilateral": return "N2"
        return "N1"      # unilateral or midline, <=6cm, any count
    else:
        return plain_rule(n, maxmm, lat)


def det(cols, canon):
    for c in cols:
        cl = re.sub(r"[^a-z]", "", c.lower())
        if canon == "HPV" and "hpv" in cl: return c
    return None


def load(args):
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    ncol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("nstage", "n")][0]
    hcol = det(clin.columns, "HPV")
    clin[pidc] = clin[pidc].astype(str).str.strip()
    clin = clin[[pidc, ncol, hcol]].rename(columns={pidc: "PatientID", ncol: "N", hcol: "HPV"})
    clin["N"] = clin["N"].astype(str).str.upper().str[:2]
    clin = clin[clin["N"].isin(["N0", "N1", "N2", "N3"])]

    meta = pd.read_csv(args.meta)
    df = clin.merge(meta, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    for lv in LAT_LEVELS:
        df[f"lat_{lv}"] = (df.get("laterality", "none") == lv).astype(int)
    return df


def bal_acc_overall(df, pred_col):
    return balanced_accuracy_score(df["N"], df[pred_col])


def cv_loco_for_rule(df, pred_col, splits):
    cv = []
    for f in splits:
        va = df[df.PatientID.isin(set(f["val"]))]
        if len(va) == 0: continue
        cv.append(balanced_accuracy_score(va["N"], va[pred_col]))
    loco = []
    for c in sorted(df.center.unique()):
        va = df[df.center == c]
        if len(va) < 5: continue
        loco.append(balanced_accuracy_score(va["N"], va[pred_col]))
    return float(np.mean(cv)), float(np.mean(loco))


def cv_loco_for_logreg(df, feats, splits):
    def one(groups):
        accs = []
        for g in groups:
            va_ids = set(g["val"])
            tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
            if va.sum() < 5 or tr.sum() < 20: continue
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])
            pipe.fit(df[feats][tr], df["N"].values[tr])
            accs.append(balanced_accuracy_score(df["N"].values[va], pipe.predict(df[feats][va])))
        return float(np.mean(accs)) if accs else float("nan")
    cv = one(splits)
    loco = one([{"val": list(df[df.center == c]["PatientID"])} for c in sorted(df.center.unique())])
    return cv, loco


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

    print(f"[data] {len(df)} patients", flush=True)
    print(f"[data] N dist: {df['N'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"[data] HPV dist: {df['HPV'].value_counts(dropna=False).to_dict()}", flush=True)
    print()

    # compute rule predictions
    df["plain"] = df.apply(lambda r: plain_rule(r["n_gtvn"], r["max_node_mm"],
                                                r.get("laterality", "none")), axis=1)
    df["hpvr"] = df.apply(lambda r: hpv_rule(r["n_gtvn"], r["max_node_mm"],
                                             r.get("laterality", "none"), r["HPV"]), axis=1)

    lat_oh = [c for c in df.columns if c.startswith("lat_")]
    logreg_feats = [c for c in GEOM if c in df.columns] + lat_oh

    print(f"{'approach':<28}{'CV_bal_acc':>12}{'LOCO':>10}{'overall':>10}", flush=True)
    print("-" * 60, flush=True)
    p_cv, p_loco = cv_loco_for_rule(df, "plain", splits)
    print(f"{'1. plain rule (current)':<28}{p_cv:>12.3f}{p_loco:>10.3f}{bal_acc_overall(df,'plain'):>10.3f}", flush=True)
    h_cv, h_loco = cv_loco_for_rule(df, "hpvr", splits)
    print(f"{'2. HPV-conditioned rule':<28}{h_cv:>12.3f}{h_loco:>10.3f}{bal_acc_overall(df,'hpvr'):>10.3f}", flush=True)
    l_cv, l_loco = cv_loco_for_logreg(df, logreg_feats, splits)
    print(f"{'3. LogReg classifier':<28}{l_cv:>12.3f}{l_loco:>10.3f}{'-':>10}", flush=True)

    print()
    approaches = [("plain rule", p_cv, p_loco),
                  ("HPV rule", h_cv, h_loco),
                  ("LogReg", l_cv, l_loco)]
    best = max(approaches, key=lambda a: (a[1] if not np.isnan(a[1]) else -1))
    print(f"[baseline]  plain rule CV={p_cv:.3f} LOCO={p_loco:.3f}", flush=True)
    print(f"[best]      {best[0]}: CV={best[1]:.3f} LOCO={best[2]:.3f}", flush=True)
    d = best[1] - p_cv
    print(f"[verdict]   best beats plain rule by {d:+.3f} CV", flush=True)
    if best[0] == "plain rule":
        print("            -> plain rule is best. Keep current N approach.", flush=True)
    elif d > 0.01 and best[2] >= p_loco - 0.02:
        print(f"            -> {best[0]} wins and generalizes. Ship it for Slot 2.", flush=True)
    else:
        print(f"            -> {best[0]} best on CV but check LOCO before shipping.", flush=True)

    # breakdown: how many patients does HPV-rule actually CHANGE vs plain?
    changed = (df["plain"] != df["hpvr"]).sum()
    print(f"\n[info] HPV-rule changes prediction for {changed}/{len(df)} patients "
          f"({100*changed/len(df):.0f}%) vs plain rule", flush=True)


if __name__ == "__main__":
    main()

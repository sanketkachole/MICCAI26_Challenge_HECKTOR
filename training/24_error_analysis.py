#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 24 (Phase 0): error analysis for N and T staging.

Shows WHERE we fail so Phase 1 can target real errors:
  - N-stage confusion matrix (rule predictions vs ground truth)
  - T-stage confusion matrix (current T-model, via simple CV predictions)
  - Per-center balanced accuracy for N and T
  - Which specific (true -> predicted) mistakes are most common
  - Node-count / node-size distribution by true N-stage (is geometry separable?)

REPORT ONLY. No models saved.

USAGE (HECKTOR env):
  python 24_error_analysis.py \
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]


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
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    ncol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("nstage", "n")][0]
    tcol = [c for c in clin.columns if re.sub(r"[^a-z]", "", c.lower()) in ("tstage", "t")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    cmap = {det(clin.columns, k): k for k in CLIN if det(clin.columns, k)}
    keep = {pidc: "PatientID", ncol: "N", tcol: "T", **cmap}
    clin = clin[list(keep)].rename(columns=keep)
    clin["N"] = clin["N"].astype(str).str.upper().str[:2]
    clin["T"] = clin["T"].astype(str).str.upper().str.replace("T", "", regex=False).str[:1]
    clin = clin[clin["N"].isin(["N0", "N1", "N2", "N3"])]

    meta = pd.read_csv(args.meta)
    df = clin.merge(meta, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    return df


def print_confmat(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n{title}")
    print("            " + "".join(f"{f'pred {l}':>10}" for l in labels) + f"{'recall':>10}")
    for i, l in enumerate(labels):
        row = cm[i]
        rec = row[i] / row.sum() if row.sum() else 0
        print(f"  true {l:<5}" + "".join(f"{v:>10}" for v in row) + f"{rec:>10.2f}")
    print(f"  {'support':<9}" + "".join(f"{cm[:,j].sum():>10}" for j in range(len(labels))))


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

    print("=" * 68)
    print("PHASE 0 ERROR ANALYSIS")
    print("=" * 68)
    print(f"[data] {len(df)} patients | centers: {sorted(df['center'].unique())}")
    print(f"[data] N dist: {df['N'].value_counts().sort_index().to_dict()}")
    print(f"[data] T dist: {df['T'].value_counts().sort_index().to_dict()}")

    # ---------- N-STAGE (rule) ----------
    df["N_pred"] = df.apply(lambda r: plain_rule(r["n_gtvn"], r["max_node_mm"],
                                                 r.get("laterality", "none")), axis=1)
    nlabels = ["N0", "N1", "N2", "N3"]
    print("\n" + "-" * 68)
    print("N-STAGE (current rule)")
    print("-" * 68)
    print(f"overall balanced accuracy: {balanced_accuracy_score(df['N'], df['N_pred']):.3f}")
    print_confmat(df["N"], df["N_pred"], nlabels, "N confusion (rows=truth, cols=prediction):")

    # most common N mistakes
    mis = df[df["N"] != df["N_pred"]]
    print(f"\ntop N mistakes (true -> predicted), {len(mis)} total errors:")
    combo = (mis["N"] + " -> " + mis["N_pred"]).value_counts().head(6)
    for k, v in combo.items():
        print(f"    {k}: {v}")

    # per-center N
    print("\nper-center N balanced accuracy:")
    for c in sorted(df["center"].unique()):
        sub = df[df["center"] == c]
        if len(sub) >= 5:
            print(f"    {c:<6} n={len(sub):<4} BA={balanced_accuracy_score(sub['N'], sub['N_pred']):.3f}")

    # geometry separability: node count & size by true N
    print("\nnode geometry by TRUE N-stage (is it separable?):")
    print(f"    {'N':<5}{'n_patients':>11}{'med_nodes':>11}{'med_maxmm':>11}{'med_totml':>11}")
    for n in nlabels:
        sub = df[df["N"] == n]
        if len(sub):
            print(f"    {n:<5}{len(sub):>11}{sub['n_gtvn'].median():>11.1f}"
                  f"{sub['max_node_mm'].median():>11.1f}{sub['gtvn_total_ml'].median():>11.2f}")

    # ---------- T-STAGE (current model, simple CV) ----------
    print("\n" + "-" * 68)
    print("T-STAGE (current HistGB model, 5-fold CV predictions)")
    print("-" * 68)
    clin_feats = [c for c in CLIN if c in df.columns]
    tfeats = [c for c in GEOM if c in df.columns] + clin_feats
    df["T_pred"] = None
    for f in splits:
        va_ids = set(f["val"])
        tr = ~df.PatientID.isin(va_ids); va = df.PatientID.isin(va_ids)
        if va.sum() == 0: continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                  learning_rate=0.05, random_state=0))])
        pipe.fit(df[tfeats][tr], df["T"].values[tr])
        df.loc[va, "T_pred"] = pipe.predict(df[tfeats][va])
    tl = sorted(df["T"].dropna().unique())
    valid = df["T_pred"].notna()
    print(f"overall balanced accuracy: {balanced_accuracy_score(df.loc[valid,'T'], df.loc[valid,'T_pred']):.3f}")
    print_confmat(df.loc[valid, "T"], df.loc[valid, "T_pred"], tl, "T confusion (rows=truth, cols=prediction):")
    print("\nper-center T balanced accuracy:")
    for c in sorted(df["center"].unique()):
        sub = df[(df["center"] == c) & valid]
        if len(sub) >= 5:
            print(f"    {c:<6} n={len(sub):<4} BA={balanced_accuracy_score(sub['T'], sub['T_pred']):.3f}")

    print("\n" + "=" * 68)
    print("KEY QUESTIONS TO ANSWER FROM ABOVE:")
    print("  1. Which N class is most confused? (look at N confusion recall row)")
    print("  2. Is node geometry separable by N? (do medians differ across N0-N3?)")
    print("  3. Which centers drag N/T down? (per-center BA)")
    print("=" * 68)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 41: do DISSEMINATION features improve the prognosis C-index?

Tests the Phase B hypothesis: our survival signal is saturated on "how much tumour"
(volume/TLG/Energy are collinear). Dissemination features measure "how spread out",
which is a genuinely independent axis.

Guards baked in from earlier mistakes:
  * LOCO reported BOTH unweighted and PATIENT-WEIGHTED. The unweighted mean over 8
    centres previously created a fake +0.018 advantage driven by tiny centres
    (USZ n=11). Patient-weighted is the honest number.
  * Repeated 5-fold CV over 10 seeds with mean +/- std, so we can see whether any
    difference exceeds noise.
  * Collinearity check: correlation of each new feature with gtvp_ml. Anything
    above ~0.8 is a volume proxy and will not add signal.
  * Univariate C-index of each new feature on its own.

STOP/GO: ship only if patient-weighted LOCO improves by >= 0.01 AND repeated-CV
does not get worse. Otherwise keep the current Cox model and ship Slot 2 as built.

Run in HECKTOR env (lifelines + sklearn), as a batch job.
"""
import argparse, json, re, warnings
print("[boot] importing...", flush=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
print("[boot] imports done", flush=True)

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]

# candidate dissemination features (only those that are not volume proxies)
DIS_CANDIDATES = [
    "Dmax_mm", "Dmax_norm", "spread_mean_mm", "n_lesions",
    "centroid_spread_mm", "centroid_max_mm",
    "dist_p_to_n_max_mm", "dist_p_to_n_mean_mm",
    "frac_largest", "gtvn_to_gtvp_ratio",
    "suv_hetero", "suv_max_all", "suv_std_all",
]


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


def load(args):
    print("[load] clinical...", flush=True)
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    rfs = det(clin.columns, "RFS"); rel = det(clin.columns, "Relapse")
    keep = {pidc: "PatientID", rfs: "RFS", rel: "event"}
    for k in CLIN:
        v = det(clin.columns, k)
        if v: keep[v] = k
    df = clin[list(keep)].rename(columns=keep)
    df = df.dropna(subset=["RFS", "event"])
    df = df[df["RFS"] > 0]
    if "HPV" in df.columns:
        df["HPV_missing"] = df["HPV"].isna().astype(int)

    print("[load] meta...", flush=True)
    meta = pd.read_csv(args.meta)
    gc = [c for c in GEOM if c in meta.columns]
    mk = ["PatientID"] + gc + (["laterality"] if "laterality" in meta.columns else [])
    meta = meta[mk]
    if "laterality" in meta.columns:
        def rn(r):
            if r["n_gtvn"] == 0: return 0
            if r["max_node_mm"] > 55: return 3
            if r["n_gtvn"] == 1 and r["laterality"] in ("unilateral", "midline") and r["max_node_mm"] <= 35: return 1
            return 2
        meta["N_ordinal"] = meta.apply(rn, axis=1)
        meta = meta.drop(columns=["laterality"])

    print("[load] dissemination...", flush=True)
    dis = pd.read_csv(args.dissemination)
    dis_cols = [c for c in DIS_CANDIDATES if c in dis.columns]
    dis = dis[["PatientID"] + dis_cols]

    df = df.merge(meta, on="PatientID", how="left").merge(dis, on="PatientID", how="left")
    df["center"] = df["PatientID"].str.split("-").str[0]
    print("[load] done (%d patients, %d events)" % (len(df), int(df["event"].sum())), flush=True)
    return df, dis_cols


def prep(Xtr, Xva):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva))


def fit_predict(Xtr, ttr, etr, Xva):
    d = pd.DataFrame(Xtr, columns=["f%d" % i for i in range(Xtr.shape[1])])
    d["T"] = np.asarray(ttr); d["E"] = np.asarray(etr)
    cph = CoxPHFitter(penalizer=0.5).fit(d, "T", "E")
    dv = pd.DataFrame(Xva, columns=["f%d" % i for i in range(Xva.shape[1])])
    return cph.predict_partial_hazard(dv).values


def loco_detail(df, feats):
    """Per-centre C-index plus unweighted and patient-weighted means."""
    rows = []
    for c in sorted(df.center.unique()):
        tr = df[df.center != c]; va = df[df.center == c]
        if len(va) < 8 or va["event"].sum() < 3 or len(tr) < 30:
            continue
        Xtr, Xva = prep(tr[feats], va[feats])
        try:
            risk = fit_predict(Xtr, tr["RFS"].values, tr["event"].values, Xva)
            ci = float(concordance_index(va["RFS"].values, -risk, va["event"].values))
            rows.append((c, len(va), int(va["event"].sum()), ci))
        except Exception as e:
            print("    [warn] %s failed: %s" % (c, str(e)[:40]), flush=True)
    if not rows:
        return rows, np.nan, np.nan
    ci = np.array([r[3] for r in rows])
    n = np.array([r[1] for r in rows], dtype=float)
    return rows, float(ci.mean()), float((ci * n).sum() / n.sum())


def repeated_cv(df, feats, seeds=10):
    strat = df["event"].astype(int).astype(str) + "_" + df["center"].astype(str)
    out = []
    for s in range(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        cis = []
        for tr_i, va_i in skf.split(df, strat):
            tr = df.iloc[tr_i]; va = df.iloc[va_i]
            if va["event"].sum() < 3:
                continue
            Xtr, Xva = prep(tr[feats], va[feats])
            try:
                risk = fit_predict(Xtr, tr["RFS"].values, tr["event"].values, Xva)
                cis.append(float(concordance_index(va["RFS"].values, -risk, va["event"].values)))
            except Exception:
                pass
        if cis:
            out.append(float(np.mean(cis)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--dissemination", required=True)
    args = ap.parse_args()

    df, dis_cols = load(args)
    base = [c for c in GEOM if c in df.columns] + [k for k in CLIN if k in df.columns]
    if "N_ordinal" in df.columns: base += ["N_ordinal"]
    if "HPV_missing" in df.columns: base += ["HPV_missing"]

    print("\n[feats] base Cox: %d features" % len(base), flush=True)
    print("[feats] dissemination available: %s" % dis_cols, flush=True)

    # ---------- 1. are the new features actually independent of volume? ----------
    print("\n" + "=" * 70, flush=True)
    print("COLLINEARITY WITH TUMOUR VOLUME (>0.8 means it is a volume proxy)", flush=True)
    print("=" * 70, flush=True)
    print("  %-24s%12s%14s%12s" % ("feature", "corr(gtvp)", "univar C-idx", "verdict"), flush=True)
    keep_dis = []
    for c in dis_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() < 100 or v.nunique() < 3:
            print("  %-24s%12s%14s%12s" % (c, "--", "--", "too sparse"), flush=True)
            continue
        vv = v.fillna(v.median())
        r = float(np.corrcoef(vv, df["gtvp_ml"].fillna(df["gtvp_ml"].median()))[0, 1])
        try:
            ci = float(concordance_index(df["RFS"], vv, df["event"]))
        except Exception:
            ci = np.nan
        proxy = abs(r) > 0.8
        if not proxy:
            keep_dis.append(c)
        print("  %-24s%12.3f%14.3f%12s" % (c, r, ci, "VOLUME PROXY" if proxy else "independent"), flush=True)
    print("\n  -> keeping %d non-proxy features: %s" % (len(keep_dis), keep_dis), flush=True)

    # ---------- 2. LOCO comparison ----------
    configs = [
        ("base Cox (current)", base),
        ("base + dissemination", base + keep_dis),
        ("base + Dmax only", base + [c for c in ["Dmax_mm", "Dmax_norm"] if c in keep_dis]),
    ]
    print("\n" + "=" * 70, flush=True)
    print("LEAVE-ONE-CENTRE-OUT (unweighted mean can mislead - see weighted)", flush=True)
    print("=" * 70, flush=True)
    results = {}
    for name, feats in configs:
        if not feats:
            continue
        rows, unw, wtd = loco_detail(df, feats)
        results[name] = (unw, wtd, rows)
        print("\n%s  (%d features)" % (name, len(feats)), flush=True)
        print("  %-8s%8s%8s%10s" % ("centre", "n", "events", "C-index"), flush=True)
        for c, n, e, ci in rows:
            print("  %-8s%8d%8d%10.3f" % (c, n, e, ci), flush=True)
        print("  %-8s%8s%8s%10.3f   <- unweighted" % ("MEAN", "", "", unw), flush=True)
        print("  %-8s%8s%8s%10.3f   <- PATIENT-WEIGHTED (trust this)" % ("", "", "", wtd), flush=True)

    # ---------- 3. repeated CV stability ----------
    print("\n" + "=" * 70, flush=True)
    print("REPEATED 5-FOLD CV (10 seeds) - is any difference above noise?", flush=True)
    print("=" * 70, flush=True)
    cv_res = {}
    for name, feats in configs:
        if not feats:
            continue
        arr = repeated_cv(df, feats)
        cv_res[name] = arr
        print("  %-24s %.3f +/- %.3f" % (name, arr.mean(), arr.std()), flush=True)
    if "base Cox (current)" in cv_res and "base + dissemination" in cv_res:
        a = cv_res["base + dissemination"]; b = cv_res["base Cox (current)"]
        n = min(len(a), len(b))
        d = a[:n] - b[:n]
        print("\n  difference (dissemination - base): %+.3f +/- %.3f  (better in %d/%d seeds)"
              % (d.mean(), d.std(), int((d > 0).sum()), n), flush=True)

    # ---------- 4. verdict ----------
    print("\n" + "=" * 70, flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)
    if "base Cox (current)" in results and "base + dissemination" in results:
        b_unw, b_wtd, _ = results["base Cox (current)"]
        d_unw, d_wtd, _ = results["base + dissemination"]
        print("  LOCO patient-weighted:  base %.3f  ->  +dissemination %.3f  (%+.3f)"
              % (b_wtd, d_wtd, d_wtd - b_wtd), flush=True)
        print("  LOCO unweighted:        base %.3f  ->  +dissemination %.3f  (%+.3f)"
              % (b_unw, d_unw, d_unw - b_unw), flush=True)
        cv_ok = True
        if "base + dissemination" in cv_res and "base Cox (current)" in cv_res:
            cv_ok = cv_res["base + dissemination"].mean() >= cv_res["base Cox (current)"].mean() - 0.005
        if (d_wtd - b_wtd) >= 0.01 and cv_ok:
            print("\n  -> SHIP IT: patient-weighted LOCO gain >= 0.01 and CV holds.", flush=True)
        elif (d_wtd - b_wtd) >= 0.01:
            print("\n  -> CAUTION: LOCO gain but repeated-CV got worse. Probably noise.", flush=True)
        else:
            print("\n  -> DO NOT SHIP: no real gain. Keep current Cox, ship Slot 2 as built.", flush=True)


if __name__ == "__main__":
    main()

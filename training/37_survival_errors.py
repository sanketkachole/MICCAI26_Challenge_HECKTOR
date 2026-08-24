#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 37 (Phase A): WHERE does our survival model fail?

Produces out-of-fold (LOCO) risk predictions from the current Cox model, then asks:
  1. Which SUBGROUPS have low concordance? (center, HPV, N-stage, T-stage,
     tumour burden tertile, event vs censored)
  2. Which INDIVIDUAL patients are most badly misranked, and what do they have
     in common? (high predicted risk but long survival, or low risk but early event)
  3. Does predicted risk actually separate early-relapse from late/no-relapse?
  4. Which single features carry the most univariate prognostic signal, and which
     are we NOT currently using?

REPORT ONLY. This aims the feature engineering in Phase B.
Run in HECKTOR env (lifelines + sklearn), as a batch job.
"""
import argparse, json, re, warnings
print("[boot] importing...", flush=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
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
        if canon == "Nstage" and cl in ("nstage", "n"): return c
        if canon == "Tstage" and cl in ("tstage", "t"): return c
    return None


def find_rad(cols, modality, region, stat):
    for c in cols:
        cl = c.lower()
        if cl.startswith("%s_%s_" % (modality, region)) and "firstorder" in cl and stat.lower() in cl:
            return c
    return None


def load(args):
    print("[load] clinical...", flush=True)
    clin = pd.read_csv(args.clinical)
    pidc = [c for c in clin.columns if c.lower() in ("patientid", "id")][0]
    clin[pidc] = clin[pidc].astype(str).str.strip()
    rfs = det(clin.columns, "RFS"); rel = det(clin.columns, "Relapse")
    ncol = det(clin.columns, "Nstage"); tcol = det(clin.columns, "Tstage")
    keep = {pidc: "PatientID", rfs: "RFS", rel: "event"}
    if ncol: keep[ncol] = "N_true"
    if tcol: keep[tcol] = "T_true"
    for k in CLIN:
        v = det(clin.columns, k)
        if v: keep[v] = k
    df = clin[list(keep)].rename(columns=keep)
    df = df.dropna(subset=["RFS", "event"]); df = df[df["RFS"] > 0]
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

    print("[load] radiomics (PET intensity)...", flush=True)
    rad = pd.read_csv(args.radiomics)
    pet = {}
    for region in ("gtvp", "gtvn"):
        for stat in ("Mean", "Maximum", "Energy"):
            col = find_rad(rad.columns, "pt", region, stat)
            if col: pet["SUV_%s_%s" % (region, stat)] = col
    rad_small = rad[["PatientID"] + list(pet.values())].rename(
        columns={v: k for k, v in pet.items()})

    df = df.merge(meta, on="PatientID", how="left").merge(rad_small, on="PatientID", how="left")
    if "SUV_gtvp_Mean" in df and "gtvp_ml" in df:
        df["TLG_gtvp"] = df["gtvp_ml"] * df["SUV_gtvp_Mean"]
    if "SUV_gtvn_Mean" in df and "gtvn_total_ml" in df:
        df["TLG_gtvn"] = df["gtvn_total_ml"] * df["SUV_gtvn_Mean"]
    df["center"] = df["PatientID"].str.split("-").str[0]
    print("[load] done (%d patients, %d events)" % (len(df), int(df["event"].sum())), flush=True)
    return df, list(pet.keys())


def prep(Xtr, Xva):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva))


def oof_risk(df, feats):
    """Leave-one-center-out risk for every patient (matches the unseen-center setting)."""
    risk = pd.Series(index=df.index, dtype=float)
    for c in sorted(df.center.unique()):
        tr = df[df.center != c]; va = df[df.center == c]
        if len(tr) < 30 or tr["event"].sum() < 5:
            continue
        Xtr, Xva = prep(tr[feats], va[feats])
        d = pd.DataFrame(Xtr, columns=["f%d" % i for i in range(Xtr.shape[1])])
        d["T"] = tr["RFS"].values; d["E"] = tr["event"].values
        try:
            cph = CoxPHFitter(penalizer=0.5).fit(d, "T", "E")
            dv = pd.DataFrame(Xva, columns=["f%d" % i for i in range(Xva.shape[1])])
            risk[va.index] = cph.predict_partial_hazard(dv).values
        except Exception as e:
            print("  [warn] center %s failed: %s" % (c, str(e)[:50]), flush=True)
    return risk


def ci_of(sub):
    if len(sub) < 10 or sub["event"].sum() < 3 or sub["risk"].isna().all():
        return np.nan, len(sub), int(sub["event"].sum())
    s = sub.dropna(subset=["risk"])
    try:
        return (float(concordance_index(s["RFS"], -s["risk"], s["event"])),
                len(s), int(s["event"].sum()))
    except Exception:
        return np.nan, len(sub), int(sub["event"].sum())


def report_group(df, col, title, bins=None):
    print("\n%s" % title, flush=True)
    print("  %-22s%8s%8s%10s" % ("group", "n", "events", "C-index"), flush=True)
    if bins is not None:
        g = pd.cut(df[col], bins=bins, duplicates="drop")
        keys = g.cat.categories
        for k in keys:
            sub = df[g == k]
            ci, n, e = ci_of(sub)
            print("  %-22s%8d%8d%10s" % (str(k), n, e, "--" if np.isnan(ci) else "%.3f" % ci), flush=True)
    else:
        for k in sorted(df[col].dropna().unique()):
            sub = df[df[col] == k]
            ci, n, e = ci_of(sub)
            print("  %-22s%8d%8d%10s" % (str(k), n, e, "--" if np.isnan(ci) else "%.3f" % ci), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    args = ap.parse_args()

    df, pet_feats = load(args)
    base = [c for c in GEOM if c in df.columns] + [k for k in CLIN if k in df.columns]
    if "N_ordinal" in df.columns: base += ["N_ordinal"]
    if "HPV_missing" in df.columns: base += ["HPV_missing"]
    print("[feats] current Cox uses %d features" % len(base), flush=True)

    df = df.reset_index(drop=True)
    df["risk"] = oof_risk(df, base)
    overall, n, e = ci_of(df)
    print("\n" + "=" * 64, flush=True)
    print("OVERALL out-of-fold (LOCO) C-index: %.3f   (n=%d, events=%d)" % (overall, n, e), flush=True)
    print("=" * 64, flush=True)

    # ---------- 1. subgroup concordance ----------
    report_group(df, "center", "BY CENTRE:")
    if "HPV" in df.columns:
        df["HPV_grp"] = np.where(df["HPV"].isna(), "missing",
                          np.where(df["HPV"] == 1, "HPV+", "HPV-"))
        report_group(df, "HPV_grp", "BY HPV STATUS:")
    if "N_true" in df.columns:
        df["N_grp"] = df["N_true"].astype(str).str.upper().str[:2]
        report_group(df, "N_grp", "BY TRUE N-STAGE:")
    if "T_true" in df.columns:
        df["T_grp"] = ("T" + df["T_true"].astype(str).str.upper()
                       .str.replace("T", "", regex=False).str.strip().str[:1])
        report_group(df, "T_grp", "BY TRUE T-STAGE:")
    if "gtvp_ml" in df.columns:
        q = df["gtvp_ml"].quantile([0, .33, .66, 1.0]).values
        report_group(df, "gtvp_ml", "BY PRIMARY TUMOUR VOLUME (tertiles):", bins=q)
    if "n_gtvn" in df.columns:
        df["node_grp"] = np.where(df["n_gtvn"] == 0, "0 nodes",
                          np.where(df["n_gtvn"] == 1, "1 node", "2+ nodes"))
        report_group(df, "node_grp", "BY PREDICTED NODE COUNT:")

    # ---------- 2. worst-misranked patients ----------
    print("\n" + "=" * 64, flush=True)
    print("WORST MISRANKED PATIENTS", flush=True)
    print("=" * 64, flush=True)
    d = df.dropna(subset=["risk"]).copy()
    d["risk_pct"] = d["risk"].rank(pct=True)
    d["time_pct"] = d["RFS"].rank(pct=True)
    ev = d[d["event"] == 1].copy()
    ev["miss"] = ev["time_pct"] - (1 - ev["risk_pct"])   # relapsed but we called them low-risk
    worst = ev.nlargest(10, "miss")
    print("\nRELAPSED but predicted LOW risk (we missed these):", flush=True)
    print("  %-12s%8s%8s%8s%10s%8s" % ("patient", "RFS", "gtvp_ml", "nodes", "maxnode", "HPV"), flush=True)
    for _, r in worst.iterrows():
        print("  %-12s%8.0f%8.1f%8.0f%10.1f%8s" % (
            r["PatientID"], r["RFS"], r.get("gtvp_ml", np.nan), r.get("n_gtvn", np.nan),
            r.get("max_node_mm", np.nan), str(r.get("HPV", "?"))), flush=True)

    cen = d[d["event"] == 0].copy()
    cen["miss"] = cen["risk_pct"] - cen["time_pct"]      # no relapse but we called them high-risk
    worst2 = cen.nlargest(10, "miss")
    print("\nNO relapse (long follow-up) but predicted HIGH risk (false alarms):", flush=True)
    print("  %-12s%8s%8s%8s%10s%8s" % ("patient", "RFS", "gtvp_ml", "nodes", "maxnode", "HPV"), flush=True)
    for _, r in worst2.iterrows():
        print("  %-12s%8.0f%8.1f%8.0f%10.1f%8s" % (
            r["PatientID"], r["RFS"], r.get("gtvp_ml", np.nan), r.get("n_gtvn", np.nan),
            r.get("max_node_mm", np.nan), str(r.get("HPV", "?"))), flush=True)

    # ---------- 3. does risk separate early vs late relapse? ----------
    print("\n" + "=" * 64, flush=True)
    print("RISK SEPARATION CHECK (events only, split by median RFS)", flush=True)
    print("=" * 64, flush=True)
    if len(ev) > 10:
        med = ev["RFS"].median()
        early = ev[ev["RFS"] <= med]["risk_pct"].mean()
        late = ev[ev["RFS"] > med]["risk_pct"].mean()
        print("  mean predicted-risk percentile, EARLY relapse: %.3f" % early, flush=True)
        print("  mean predicted-risk percentile, LATE  relapse: %.3f" % late, flush=True)
        print("  separation (early - late, want clearly >0): %+.3f" % (early - late), flush=True)

    # ---------- 4. univariate signal of every feature we have ----------
    print("\n" + "=" * 64, flush=True)
    print("UNIVARIATE PROGNOSTIC SIGNAL (|C-index - 0.5|, higher = more useful)", flush=True)
    print("=" * 64, flush=True)
    cand = [c for c in df.columns if c not in
            ("PatientID", "center", "RFS", "event", "risk", "risk_pct", "time_pct",
             "N_true", "T_true", "N_grp", "T_grp", "HPV_grp", "node_grp", "laterality")]
    rows = []
    for c in cand:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() < 50 or v.nunique() < 3:
            continue
        vv = v.fillna(v.median())
        try:
            ci = float(concordance_index(df["RFS"], vv, df["event"]))
            rows.append((abs(ci - 0.5), ci, c, "IN MODEL" if c in base else "not used"))
        except Exception:
            pass
    rows.sort(reverse=True)
    print("  %-26s%10s%10s  %s" % ("feature", "C-index", "|C-.5|", "status"), flush=True)
    for s, ci, c, st in rows[:20]:
        print("  %-26s%10.3f%10.3f  %s" % (c, ci, s, st), flush=True)

    print("\n[next] Phase B will add lesion-dissemination features (Dmax, lesion count,", flush=True)
    print("       spread) which are NOT in the list above - the main untried signal.", flush=True)


if __name__ == "__main__":
    main()

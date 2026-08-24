#!/usr/bin/env python3
r"""
HECKTOR 2026 - Step 31: CONTAINER-FRIENDLY survival ensemble.

PROBLEM: the winning 4-model ensemble needs scikit-survival (RSF, GBSA) + the icare
package. scikit-survival 0.22.2 requires sklearn 1.3.2, but the container runs
sklearn 1.9.0 (nnU-Net + our saved models). Downgrading risks breaking the container.

SOLUTION: reimplement ICARE in pure numpy (it is a simple algorithm) and pair it with
the lifelines Cox we already ship. Then test whether Cox + numpyICARE captures the
LOCO gain (full 4-model ensemble was LOCO 0.662 vs Cox base 0.636).

This script:
  1. Implements NumpyICARE + BaggedNumpyICARE (no new dependencies).
  2. VERIFIES it against the real icare package on the same data.
  3. Compares container-friendly ensembles on 5-fold CV + LOCO.

Run in the `icare` env so we can verify against the real package.
USAGE (batch):
  python 31_container_ensemble.py --clinical ... --meta ... --radiomics ... --splits ...
"""
import argparse, json, re, warnings
print("[boot] importing...", flush=True)
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
print("[boot] imports done", flush=True)

GEOM = ["gtvp_ml", "n_gtvn", "max_node_mm", "gtvn_total_ml"]
CLIN = ["Age", "Gender", "Tobacco", "Alcohol", "Performance", "HPV", "Treatment"]


# ==================================================================== NumpyICARE
class NumpyICARE:
    """ICARE in pure numpy: learn only a SIGN per feature from univariate concordance,
    then risk = mean of signed z-scores. No sklearn/sksurv/icare dependency.

    Container-safe: uses only numpy + lifelines.concordance_index.
    """
    def __init__(self, cmin=0.0):
        self.cmin = cmin      # keep a feature only if |C-0.5| >= cmin
        self.signs_ = None
        self.keep_ = None
        self.mu_ = None
        self.sd_ = None

    def fit(self, X, time, event):
        X = np.asarray(X, dtype=float)
        self.mu_ = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), self.mu_, X)
        self.sd_ = Xi.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = (Xi - self.mu_) / self.sd_
        signs, keep = [], []
        for j in range(Z.shape[1]):
            try:
                c = concordance_index(time, -Z[:, j], event)   # higher z => higher risk?
            except Exception:
                c = 0.5
            signs.append(1.0 if c >= 0.5 else -1.0)
            keep.append(abs(c - 0.5) >= self.cmin)
        self.signs_ = np.array(signs)
        self.keep_ = np.array(keep)
        if not self.keep_.any():
            self.keep_ = np.ones_like(self.keep_, dtype=bool)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xi = np.where(np.isnan(X), self.mu_, X)
        Z = (Xi - self.mu_) / self.sd_
        Zs = Z * self.signs_
        return Zs[:, self.keep_].mean(axis=1)


class BaggedNumpyICARE:
    """Bootstrap samples + random feature subsets, median-aggregated."""
    def __init__(self, n_estimators=40, feat_frac=0.6, cmin=0.0, random_state=0):
        self.n = n_estimators; self.ff = feat_frac; self.cmin = cmin
        self.rs = random_state; self.models_ = []

    def fit(self, X, time, event):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.rs)
        n, p = X.shape
        k = max(2, int(round(self.ff * p)))
        self.models_ = []
        t = np.asarray(time); e = np.asarray(event)
        for _ in range(self.n):
            idx = rng.integers(0, n, n)                     # bootstrap rows
            cols = rng.choice(p, size=k, replace=False)     # random feature subset
            if e[idx].sum() < 3:                            # need some events
                continue
            m = NumpyICARE(cmin=self.cmin).fit(X[np.ix_(idx, cols)], t[idx], e[idx])
            self.models_.append((m, cols))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if not self.models_:
            return np.zeros(X.shape[0])
        preds = np.vstack([m.predict(X[:, cols]) for m, cols in self.models_])
        return np.median(preds, axis=0)


# ==================================================================== data
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


def find_rad(cols, modality, region, stat):
    for c in cols:
        cl = c.lower()
        if cl.startswith(f"{modality}_{region}_") and "firstorder" in cl and stat.lower() in cl:
            return c
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
    df = df.dropna(subset=["RFS", "event"]); df = df[df["RFS"] > 0]
    if "HPV" in df.columns:
        df["HPV_missing"] = df["HPV"].isna().astype(int)

    print("[load] meta...", flush=True)
    meta = pd.read_csv(args.meta)
    geomcols = [c for c in GEOM if c in meta.columns]
    mkeep = ["PatientID"] + geomcols + (["laterality"] if "laterality" in meta.columns else [])
    meta = meta[mkeep]
    if "laterality" in meta.columns:
        def rn(r):
            if r["n_gtvn"] == 0: return 0
            if r["max_node_mm"] > 55: return 3
            if r["n_gtvn"] == 1 and r["laterality"] in ("unilateral","midline") and r["max_node_mm"] <= 35: return 1
            return 2
        meta["N_ordinal"] = meta.apply(rn, axis=1)
        meta = meta.drop(columns=["laterality"])

    print("[load] radiomics (PET intensity)...", flush=True)
    rad = pd.read_csv(args.radiomics)
    pet = {}
    for region in ("gtvp", "gtvn"):
        for stat in ("Mean", "Maximum", "Energy"):
            col = find_rad(rad.columns, "pt", region, stat)
            if col: pet[f"SUV_{region}_{stat}"] = col
    rad_small = rad[["PatientID"] + list(pet.values())].rename(columns={v: k for k, v in pet.items()})

    df = df.merge(meta, on="PatientID", how="left").merge(rad_small, on="PatientID", how="left")
    if "SUV_gtvp_Mean" in df and "gtvp_ml" in df: df["TLG_gtvp"] = df["gtvp_ml"]*df["SUV_gtvp_Mean"]
    if "SUV_gtvn_Mean" in df and "gtvn_total_ml" in df: df["TLG_gtvn"] = df["gtvn_total_ml"]*df["SUV_gtvn_Mean"]
    df["center"] = df["PatientID"].str.split("-").str[0]
    print(f"[load] done ({len(df)} patients, {int(df['event'].sum())} events)", flush=True)
    return df, list(pet.keys())


def prep(Xtr, Xva):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xva))


def fit_cox_lifelines(Xtr, ttr, etr):
    d = pd.DataFrame(Xtr, columns=[f"f{i}" for i in range(Xtr.shape[1])])
    d["T"] = np.asarray(ttr); d["E"] = np.asarray(etr)
    cph = CoxPHFitter(penalizer=0.5).fit(d, "T", "E")
    return cph


def predict_cox(cph, Xva):
    d = pd.DataFrame(Xva, columns=[f"f{i}" for i in range(Xva.shape[1])])
    return cph.predict_partial_hazard(d).values


def eval_combos(df, feats, splits, use_loco=False):
    groups = ([{"val": list(df[df.center == c]["PatientID"])} for c in sorted(df.center.unique())]
              if use_loco else splits)
    res = {"Cox": [], "npICARE": [], "bagICARE": [],
           "Cox+npICARE": [], "Cox+bagICARE": [], "Cox+bag+np": []}
    for g in groups:
        va_ids = set(g["val"])
        tr = df[~df.PatientID.isin(va_ids)]; va = df[df.PatientID.isin(va_ids)]
        if len(va) < 8 or len(tr) < 30 or va["event"].sum() < 3: continue
        Xtr, Xva = prep(tr[feats], va[feats])
        ttr, etr = tr["RFS"].values, tr["event"].values
        tva, eva = va["RFS"].values, va["event"].values
        r = {}
        try:
            cph = fit_cox_lifelines(Xtr, ttr, etr); r["Cox"] = predict_cox(cph, Xva)
        except Exception as e:
            print(f"    [warn] Cox failed: {str(e)[:50]}", flush=True)
        try:
            r["npICARE"] = NumpyICARE().fit(Xtr, ttr, etr).predict(Xva)
        except Exception as e:
            print(f"    [warn] npICARE failed: {str(e)[:50]}", flush=True)
        try:
            r["bagICARE"] = BaggedNumpyICARE(n_estimators=40).fit(Xtr, ttr, etr).predict(Xva)
        except Exception as e:
            print(f"    [warn] bagICARE failed: {str(e)[:50]}", flush=True)

        def ci(risk): return float(concordance_index(tva, -risk, eva))
        for k in ("Cox", "npICARE", "bagICARE"):
            if k in r: res[k].append(ci(r[k]))
        def rank_ens(keys):
            avail = [r[k] for k in keys if k in r]
            if len(avail) < 2: return None
            return np.vstack([rankdata(a) for a in avail]).mean(0)
        for name, keys in [("Cox+npICARE", ("Cox","npICARE")),
                           ("Cox+bagICARE", ("Cox","bagICARE")),
                           ("Cox+bag+np", ("Cox","bagICARE","npICARE"))]:
            e = rank_ens(keys)
            if e is not None: res[name].append(ci(e))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--radiomics", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--verify-icare", action="store_true",
                    help="also compare NumpyICARE against the real icare package")
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
    pet_avail = [c for c in (pet_feats + ["TLG_gtvp","TLG_gtvn"]) if c in df.columns]
    feats = base + pet_avail
    print(f"\n[feats] {len(feats)} features (base {len(base)} + PET {len(pet_avail)})", flush=True)

    # ---- optional: verify NumpyICARE against the real package ----
    if args.verify_icare:
        try:
            from icare.survival import IcareSurvival, harrell_cindex
            print("\n[verify] comparing NumpyICARE vs real icare package on one split...", flush=True)
            f0 = splits[0]; va_ids = set(f0["val"])
            tr = df[~df.PatientID.isin(va_ids)]; va = df[df.PatientID.isin(va_ids)]
            Xtr, Xva = prep(tr[feats], va[feats])
            ttr, etr = tr["RFS"].values, tr["event"].values
            tva, eva = va["RFS"].values, va["event"].values
            ours = NumpyICARE().fit(Xtr, ttr, etr).predict(Xva)
            ci_ours = float(concordance_index(tva, -ours, eva))
            yy = np.array(list(zip(etr.astype(bool), ttr.astype(float))),
                          dtype=[('event','?'),('time','<f8')])
            yv = np.array(list(zip(eva.astype(bool), tva.astype(float))),
                          dtype=[('event','?'),('time','<f8')])
            real = np.asarray(IcareSurvival().fit(Xtr, yy).predict(Xva)).ravel()
            ci_real = float(harrell_cindex(yv, real))
            corr = float(np.corrcoef(rankdata(ours), rankdata(real))[0,1])
            print(f"[verify] NumpyICARE C-index={ci_ours:.3f} | real icare C-index={ci_real:.3f}", flush=True)
            print(f"[verify] rank correlation between them: {corr:.3f}", flush=True)
            if corr > 0.8 and abs(ci_ours - ci_real) < 0.05:
                print("[verify] -> NumpyICARE reproduces the package well ✅", flush=True)
            else:
                print("[verify] -> DIFFERS; inspect before shipping ⚠", flush=True)
        except Exception as e:
            print(f"[verify] skipped ({str(e)[:60]})", flush=True)

    print("\n" + "=" * 58, flush=True)
    print("CONTAINER-FRIENDLY ENSEMBLES (lifelines Cox + numpy ICARE)", flush=True)
    print("=" * 58, flush=True)
    cv = eval_combos(df, feats, splits, use_loco=False)
    lo = eval_combos(df, feats, splits, use_loco=True)
    print(f"{'model':<16}{'CV_Cindex':>12}{'LOCO':>10}", flush=True)
    print("-" * 38, flush=True)
    for k in ["Cox", "npICARE", "bagICARE", "Cox+npICARE", "Cox+bagICARE", "Cox+bag+np"]:
        print(f"{k:<16}{cv.get(k, float('nan')):>12.3f}{lo.get(k, float('nan')):>10.3f}", flush=True)

    print("\n[reference] full 4-model ensemble (needs sksurv) was LOCO 0.662", flush=True)
    print("[reference] Cox base (current container)            was LOCO 0.636", flush=True)
    print("[verdict]   ship the best container-friendly combo that beats 0.636 on LOCO.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
Patch inference.py for the Slot 2 container.

Changes (all verified before writing):
  1. Load the two NEW staging models (optional -> falls back if absent).
  2. N output   -> size-focused LogReg classifier (CV 0.683 vs rule 0.656).
     N for prognosis -> STILL the rule, because the Cox model was trained with the
     rule's N as an input feature. Changing what prognosis sees could shift the
     C-index (40% of the score), so we deliberately keep it identical.
  3. T output   -> clean-label LogReg classifier (offline 0.542 vs polluted).
     Falls back to the old HistGB model if the new one is missing or errors.

Run from the docker_task folder:
    python patch_inference.py
It writes inference.py.bak first, then patches in place.
"""
import re, shutil, sys, os

SRC = "inference.py"
BAK = "inference.py.bak"

# ---------------------------------------------------------------- replacements
OLD_LOAD = '''    staging_cfg = json.load(open(MODEL_PATH / "staging_config.json"))
    Tclf = joblib.load(MODEL_PATH / "staging_Tmodel.joblib")
    prog = joblib.load(MODEL_PATH / "prognosis_model.joblib")'''

NEW_LOAD = '''    staging_cfg = json.load(open(MODEL_PATH / "staging_config.json"))
    Tclf = joblib.load(MODEL_PATH / "staging_Tmodel.joblib")
    prog = joblib.load(MODEL_PATH / "prognosis_model.joblib")

    # --- Slot 2: learned staging models. OPTIONAL: if either file is missing the
    # container still runs using the rule (N) and the old HistGB model (T). ---
    Nclf = _try_load(MODEL_PATH / "n_classifier.joblib")
    Tclf2 = _try_load(MODEL_PATH / "t_classifier.joblib")'''

OLD_STAGING = '''    # ---- 4. N (rule) + T (model) ----
    nr = staging_cfg["N_rule"]
    n_stage = rule_N(g["n_nodes"], g["max_node_mm"], g["laterality"], nr["N3_MM"], nr["N1_MAX_MM"])
    # staging row uses SHORT clinical names; map ehr long keys -> short via T_clinical_source
    t_src = staging_cfg.get("T_clinical_source", {})  # {short: real_csv_name}
    staging_row = dict(geom)
    for short, real in t_src.items():
        staging_row[short] = ehr.get(real if real else short)
    Xt = apply_encoders(staging_row, staging_cfg["T_features"], staging_cfg.get("T_encoders", {}))
    t_stage = str(Tclf.predict(Xt)[0])'''

NEW_STAGING = '''    # ---- 4. N + T staging ----
    # The RULE N is always computed: the prognosis (Cox) model was trained with it
    # as an input feature, so we keep feeding prognosis exactly what it expects.
    nr = staging_cfg["N_rule"]
    n_stage_rule = rule_N(g["n_nodes"], g["max_node_mm"], g["laterality"],
                          nr["N3_MM"], nr["N1_MAX_MM"])

    # staging row uses SHORT clinical names; map ehr long keys -> short
    t_src = staging_cfg.get("T_clinical_source", {})  # {short: real_csv_name}
    staging_row = dict(geom)
    for short, real in t_src.items():
        staging_row[short] = ehr.get(real if real else short)

    # --- N OUTPUT: learned size-focused classifier (falls back to the rule) ---
    n_stage = n_stage_rule
    if Nclf is not None:
        try:
            row = {"max_node_mm": float(geom["max_node_mm"]),
                   "gtvn_total_ml": float(geom["gtvn_total_ml"]),
                   "gtvp_ml": float(geom["gtvp_ml"])}
            for lv in Nclf["laterality_levels"]:
                row["lat_" + lv] = 1.0 if geom["laterality"] == lv else 0.0
            Xn = np.array([[row.get(f, np.nan) for f in Nclf["features"]]], dtype=float)
            n_stage = str(Nclf["model"].predict(Xn)[0])
        except Exception as e:
            print("[warn] N classifier failed, using rule: %s" % e, flush=True)
            n_stage = n_stage_rule

    # --- T OUTPUT: clean-label classifier (falls back to the old HistGB model) ---
    t_stage = None
    if Tclf2 is not None:
        try:
            row = {}
            for gf in Tclf2["geom_features"]:
                row[gf] = float(geom[gf])
            for short in Tclf2["clinical_features"]:
                longname = Tclf2.get("clinical_long_names", {}).get(short, short)
                v = ehr.get(longname, staging_row.get(short))
                try:
                    row[short] = float(v)
                except (TypeError, ValueError):
                    row[short] = np.nan
            Xt2 = np.array([[row.get(f, np.nan) for f in Tclf2["features"]]], dtype=float)
            raw = str(Tclf2["model"].predict(Xt2)[0])
            t_stage = raw if raw.upper().startswith("T") else ("T" + raw)
        except Exception as e:
            print("[warn] T classifier failed, using old model: %s" % e, flush=True)
            t_stage = None
    if t_stage is None:
        Xt = apply_encoders(staging_row, staging_cfg["T_features"], staging_cfg.get("T_encoders", {}))
        t_stage = str(Tclf.predict(Xt)[0])'''

OLD_PROG = '''    prog_row = dict(geom); prog_row.update(ehr); prog_row["rule_N"] = n_stage'''
NEW_PROG = '''    # NOTE: feed the RULE's N (not the classifier's) - this is what Cox was trained on
    prog_row = dict(geom); prog_row.update(ehr); prog_row["rule_N"] = n_stage_rule'''

HELPER = '''

def _try_load(path):
    """Load an optional model file; return None if missing or unreadable."""
    try:
        if path.exists():
            return joblib.load(path)
        print("[info] optional model not found: %s" % path.name, flush=True)
    except Exception as e:
        print("[warn] could not load %s: %s" % (path.name, e), flush=True)
    return None

'''


def main():
    if not os.path.exists(SRC):
        sys.exit("ERROR: %s not found. Run this from the docker_task folder." % SRC)
    src = open(SRC, encoding="utf-8").read()

    # sanity: required anchors present?
    checks = [("model loading", OLD_LOAD), ("staging block", OLD_STAGING), ("prognosis row", OLD_PROG)]
    for name, anchor in checks:
        if anchor not in src:
            sys.exit("ERROR: could not find the %s section. inference.py differs from expected;\n"
                     "       paste me the file and I'll adjust the patch." % name)

    shutil.copy2(SRC, BAK)
    print("[backup] %s -> %s" % (SRC, BAK))

    out = src.replace(OLD_LOAD, NEW_LOAD)
    out = out.replace(OLD_STAGING, NEW_STAGING)
    out = out.replace(OLD_PROG, NEW_PROG)

    # insert the _try_load helper just before the pipeline function
    marker = "# =============================================================== main pipeline"
    if "_try_load" not in out.split("def _run_pipeline")[0]:
        out = out.replace(marker, HELPER.rstrip() + "\n\n" + marker, 1)

    # make sure numpy is imported (we use np in the new code)
    if not re.search(r"^import numpy as np", out, re.M):
        out = re.sub(r"^(import .*?)$", r"import numpy as np\n\1", out, count=1, flags=re.M)
        print("[patch] added 'import numpy as np'")

    open(SRC, "w", encoding="utf-8").write(out)

    # verify
    import ast
    try:
        ast.parse(out)
    except SyntaxError as e:
        shutil.copy2(BAK, SRC)
        sys.exit("ERROR: patched file has a syntax error (%s). Restored the backup." % e)

    ok = all(s in out for s in ["_try_load", "n_classifier.joblib", "t_classifier.joblib",
                                "n_stage_rule", 'prog_row["rule_N"] = n_stage_rule'])
    print("[verify] all patches present: %s" % ok)
    print("[verify] file parses cleanly: True")
    print("\nPatched. Next: copy the two .joblib files into model\\, then rebuild.")


if __name__ == "__main__":
    main()

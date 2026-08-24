#!/usr/bin/env python3
import argparse, os, re, glob, sys, json, time, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
STEP11 = os.path.join(HERE, "11_extract_radiomics.py")


def one_patient_via_subprocess(pid, pred_root, data_root, timeout_s):
    tmpd = tempfile.mkdtemp()
    tmp_csv = os.path.join(tmpd, f"{pid}.csv")
    tmp_jsonl = tmp_csv.replace(".csv", ".jsonl")
    cmd = [sys.executable, "-u", STEP11,
           "--pred-root", pred_root, "--data-root", data_root,
           "--out", tmp_csv, "--workers", "1", "--only", pid]
    try:
        subprocess.run(cmd, timeout=timeout_s, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"PatientID": pid, "center": pid.split("-")[0],
                "error": f"TIMEOUT after {timeout_s}s (skipped)"}
    if os.path.exists(tmp_jsonl):
        for line in open(tmp_jsonl):
            try:
                r = json.loads(line)
                if r.get("PatientID") == pid:
                    return r
            except Exception:
                pass
    return {"PatientID": pid, "center": pid.split("-")[0],
            "error": "no output produced (skipped)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="outputs/eda/radiomics_pred.csv")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    jsonl_path = args.out.replace(".csv", ".jsonl")
    done = set()
    if os.path.exists(jsonl_path):
        for line in open(jsonl_path):
            try: done.add(json.loads(line)["PatientID"])
            except Exception: pass
    print(f"[resume] {len(done)} already done", flush=True)

    ct_index = {}
    for root, _d, files in os.walk(args.data_root):
        p = os.path.basename(root)
        for b in files:
            if re.search(r"CT\.nii\.gz$", b, re.I):
                ct_index[p] = os.path.join(root, b)

    masks = sorted(glob.glob(os.path.join(args.pred_root, "fold_*/validation/*.nii.gz")))
    todo = [os.path.basename(m).replace(".nii.gz", "") for m in masks]
    todo = [p for p in todo if p not in done and p in ct_index]
    print(f"[todo] {len(todo)} patients, timeout={args.timeout}s each", flush=True)

    with open(jsonl_path, "a") as fout:
        for i, pid in enumerate(todo, 1):
            t0 = time.time()
            print(f"  [{i}/{len(todo)}] {pid} ... ", end="", flush=True)
            row = one_patient_via_subprocess(pid, args.pred_root, args.data_root, args.timeout)
            fout.write(json.dumps(row) + "\n"); fout.flush()
            tag = "ERROR/TIMEOUT" if row.get("error") else "ok"
            print(f"{tag} ({time.time()-t0:.0f}s)", flush=True)

    import pandas as pd
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset="PatientID").sort_values("PatientID")
    df.to_csv(args.out, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"[write] {args.out}  ({df.shape[0]} rows x {df.shape[1]} cols, {n_err} errors)", flush=True)


if __name__ == "__main__":
    main()

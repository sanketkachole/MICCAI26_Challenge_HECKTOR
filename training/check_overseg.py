import pandas as pd, numpy as np
pred = pd.read_csv("outputs/eda/case_metadata_pred.csv")
gt   = pd.read_csv("outputs/eda/case_metadata.csv")   # from step 00 (ground-truth masks)
# unify GT column names
ren = {}
for a,b in [("n_gtvn","n_gtvn"),("geom_n_nodes","n_gtvn"),("max_node_mm","max_node_mm"),
            ("geom_max_node_mm","max_node_mm"),("laterality","laterality"),("geom_laterality","laterality")]:
    if a in gt.columns: ren[a]=b
gt = gt.rename(columns=ren)
m = pred.merge(gt[["PatientID","n_gtvn","laterality"]], on="PatientID", suffixes=("_pred","_gt"))
print("=== node count: predicted vs ground truth ===")
print("pred n_gtvn: mean %.2f  median %.0f" % (m["n_gtvn_pred"].mean(), m["n_gtvn_pred"].median()))
print("gt   n_gtvn: mean %.2f  median %.0f" % (m["n_gtvn_gt"].mean(),   m["n_gtvn_gt"].median()))
print()
print("=== laterality: predicted vs ground truth (counts) ===")
print("PRED:", dict(m["laterality_pred"].value_counts()))
print("GT  :", dict(m["laterality_gt"].value_counts()))
print()
# how often pred says bilateral but GT is unilateral/none
cross = pd.crosstab(m["laterality_gt"], m["laterality_pred"])
print("rows=GT, cols=PRED")
print(cross.to_string())
print()
# среди cases where GT has 1 node, what does pred report?
one = m[m["n_gtvn_gt"]==1]
print(f"cases with exactly 1 GT node (n={len(one)}): pred node count mean %.2f median %.0f" %
      (one["n_gtvn_pred"].mean(), one["n_gtvn_pred"].median()))
print(f"  of these, pred laterality: {dict(one['laterality_pred'].value_counts())}")

"""Audit: is the storm model's 8-tree fit a real optimum or a short fuse?

The published storm booster stopped at best_iteration 8. Across five seeds the
same configuration stopped at 8, 275, 29, 28 and 33, and the seeds that ran
longer scored better. That pattern is what early stopping does when its
patience is shorter than the plateau the validation metric's own noise
produces: AUCPR on roughly 1,600 validation positives wanders, a transient
peak at round 8 is not beaten for 50 rounds, and the fit ends.

If that is the cause, raising patience should both lift the mean and shrink
the spread. If the 8 trees are a real optimum, more patience will change
nothing and the instability lives somewhere else.

Hurricane only. Flood was in here as a control and was costing roughly nine
tenths of the runtime for it: at patience 500 early stopping almost never
fires, so each flood fit runs the full 2,000 rounds over 1.87M rows instead
of stopping near 280. Flood's seed spread is already 0.4% of its mean, so the
control was answering a question nobody asked. CRISK_PATIENCE_HAZ=flood,storm
puts it back if the hurricane result ever needs one.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import v2_models as V

FEATS = list(V.FEATS)
PATIENCE = [int(x) for x in os.environ.get("CRISK_PATIENCE", "50,200,500").split(",")]
SEEDS = [int(s) for s in os.environ.get("CRISK_SEEDS", "0,1,2,3,4").split(",")]
HAZ = os.environ.get("CRISK_PATIENCE_HAZ", "storm").split(",")
YEARS = {"train": (2000, 2014), "val": (2015, 2019), "test": (2020, 2026)}
LABEL_END = {"storm": 2024}
BASE = dict(n_estimators=2000, learning_rate=0.05, max_depth=6,
            min_child_weight=50, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=5.0, n_jobs=-1, tree_method="hist", eval_metric="aucpr")

panel = pd.read_parquet("artifacts/daily_panel.parquet")
panel["year"] = panel["DATE"].dt.year
panel = panel[panel["year"].between(2000, 2026)]
panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
SPLIT = {k: panel[panel["year"].between(*v)] for k, v in YEARS.items()}
MONO = tuple(1 if f.startswith(("prcp", "ratio", "log")) else 0 for f in FEATS)


def S(name, haz):
    part = SPLIT[name]
    cap = LABEL_END.get(haz)
    return part if cap is None else part[part["year"] <= cap]


DEST = pathlib.Path("artifacts/audit_patience.json")
# Merge rather than overwrite: the sweep is run in pieces as the interesting
# range becomes clear, and a later invocation must not erase the levels an
# earlier one paid for. Results from a different run_id are dropped, not
# merged -- patience levels fitted against different boosters are not
# comparable.
out = {"run_id": V.RUN_ID, "patience": [], "seeds": SEEDS, "hazards": {}}
if DEST.exists():
    prev = json.loads(DEST.read_text())
    if prev.get("run_id") == V.RUN_ID:
        out["hazards"] = prev.get("hazards", {})
    else:
        print(f"ignoring existing {DEST.name}: it is from {prev.get('run_id')}")


def seal():
    """Record only the levels that actually finished.

    A sweep that is killed partway must not leave its unreached levels listed
    as though they had results -- a reader of the json would see a patience
    the fit never reached.
    """
    out["patience"] = sorted({int(p) for byp in out["hazards"].values()
                              for p in byp})
    DEST.write_text(json.dumps(out, indent=1))

for haz in HAZ:
    tr, va, te = S("train", haz), S("val", haz), S("test", haz)
    yc = f"y_{haz}"
    ytr, yva, yte = tr[yc].to_numpy(), va[yc].to_numpy(), te[yc].to_numpy()
    Xtr, Xva, Xte = tr[FEATS], va[FEATS], te[FEATS]
    # setdefault, not assignment: assigning would discard the levels the
    # merge above just loaded, which is the whole point of the merge.
    out["hazards"].setdefault(haz, {})
    print(f"\n=== {haz}  train_pos={ytr.sum():,}  val_pos={yva.sum():,}  "
          f"test_pos={yte.sum():,}")
    for pat in PATIENCE:
        aps, iters = [], []
        for sd in SEEDS:
            t0 = time.time()
            m = xgb.XGBClassifier(monotone_constraints=MONO, random_state=sd,
                                  early_stopping_rounds=pat, **BASE)
            m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
            a = float(average_precision_score(
                yte, m.predict_proba(Xte)[:, 1]))
            aps.append(a)
            iters.append(int(m.best_iteration))
            print(f"  patience={pat:<4d} seed={sd}  AP={a:.5f}  "
                  f"iter={m.best_iteration:>4d}  {time.time() - t0:.0f}s")
        a = np.array(aps)
        out["hazards"][haz][str(pat)] = {
            "ap": aps, "best_iteration": iters, "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)), "min": float(a.min()),
            "max": float(a.max())}
        print(f"  patience={pat:<4d} MEAN={a.mean():.5f}  sd={a.std(ddof=1):.5f}"
              f"  iters={iters}")
        # Written after every patience level, not once at the end. A long
        # sweep that gets interrupted should still leave the levels it
        # finished, and the first two usually answer the question.
        seal()

print("\n=== summary ===")
for haz, byp in out["hazards"].items():
    for pat, r in byp.items():
        print(f"{haz:6s} patience {pat:>4s}  AP {r['mean']:.5f} "
              f"+/- {r['sd']:.5f}  iters {r['best_iteration']}")

seal()
print(f"\nwrote {DEST}")

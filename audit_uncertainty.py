"""Audit: how much of the published AP is real and how much is 23 hurricanes?

Two independent sources of noise get a number here.

1. Episode-block bootstrap. The test set holds ~765k county-days, but they are
   not 765k observations: a single cyclone writes a positive row into every
   county it crosses on every day it is near. Resampling rows would therefore
   produce an absurdly tight interval. This script cuts the test period into
   contiguous date blocks, one per weather episode, and resamples whole blocks.
   The interval it returns is the interval the effective sample size supports.

2. Seed spread. Subsampling and column sampling make the fit stochastic, so the
   published AP is one draw. Refitting at several seeds says how much of the
   last digit is the seed.

The bootstrap scores the published boosters at their own `iteration_range`, so
its interval belongs to the number in the README. The seed sweep necessarily
refits, so its models are siblings of the published one rather than the
published one itself.

Splits here replicate the published run (no embargo), because the point is to
put an interval on the number that was published. Re-run after the next
training round to get the interval for the embargoed numbers.
"""
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import episode_ci as E
import v2_models as V

DRAWS = int(os.environ.get("CRISK_BOOT_DRAWS", 2000))
SEEDS = [int(s) for s in os.environ.get("CRISK_SEEDS", "0,1,2,3,4").split(",")]
# Re-exported so it reaches the output json, where a reader needs to know which
# gap defined an episode. Defined once in episode_ci.
GAP_DAYS = E.GAP_DAYS
YEARS = {"train": (2000, 2014), "val": (2015, 2019), "test": (2020, 2026)}
LABEL_END = {"storm": 2024}
XGB = dict(n_estimators=2000, learning_rate=0.05, max_depth=6,
           min_child_weight=50, subsample=0.9, colsample_bytree=0.9,
           reg_lambda=5.0, n_jobs=-1, tree_method="hist",
           eval_metric="aucpr", early_stopping_rounds=50)
BASELINE_FEATS = ("ratio_fwd3", "prcp_fwd3", "prcp_fwd3_max", "prcp_prev30",
                  "ratio_prev3", "log_fwd3", "pop_density")

panel = pd.read_parquet("artifacts/daily_panel.parquet")
panel["year"] = panel["DATE"].dt.year
panel = panel[panel["year"].between(2000, 2026)]
panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
SPLIT = {k: panel[panel["year"].between(*v)] for k, v in YEARS.items()}


def S(name, haz):
    part = SPLIT[name]
    cap = LABEL_END.get(haz)
    return part if cap is None else part[part["year"] <= cap]


scorers = V.load_scorers()
res = V.results(f"xgb_{V.TAG}.json")
out = {"run_id": V.RUN_ID, "draws": DRAWS, "gap_days": GAP_DAYS,
       "seeds": SEEDS, "hazards": {}}

for haz in V.HAZARDS:
    te = S("test", haz)
    y = te[f"y_{haz}"].to_numpy()
    p = scorers[haz](te)
    blk, nb = E.blocks(te, f"{haz}_onset")
    ap = float(average_precision_score(y, p))
    print(f"\n=== {haz} ===")
    print(f"  test rows {len(te):,}  positives {int(y.sum()):,}  "
          f"blocks {nb}  published AP {res[haz]['test']['ap']:.5f}  "
          f"rescored {ap:.5f}")

    rng = np.random.default_rng(0)
    model_b = E.boot(y, p, blk, nb, rng, DRAWS)

    # The headline claim is a ratio, so the ratio needs the interval. Drawing
    # the same blocks for model and baseline keeps them paired: the ratio's
    # spread is then the spread of the comparison, not two spreads added.
    best = max(BASELINE_FEATS, key=lambda f: average_precision_score(
        y, te[f].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()))
    bv = te[best].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    rng = np.random.default_rng(0)
    base_b = E.boot(y, bv, blk, nb, rng, DRAWS)
    ratio_b = model_b / base_b

    base_ap = float(average_precision_score(y, bv))
    # The point ratio, not the bootstrap mean of the ratio. A ratio of two
    # resampled quantities is biased away from the point estimate, so the
    # published figure has to be ap/baseline and the bootstrap supplies only
    # the interval around it.
    hz = {"test_rows": int(len(te)), "test_pos": int(y.sum()),
          "episodes": int(nb), "ap_published": float(res[haz]["test"]["ap"]),
          "ap_rescored": ap, "ap_boot": E.pct(model_b),
          "best_baseline": best, "baseline_ap": base_ap,
          "vs_baseline_point": ap / base_ap if base_ap else None,
          "baseline_boot": E.pct(base_b), "vs_baseline_boot": E.pct(ratio_b)}
    print(f"  AP        {ap:.5f}  95% [{hz['ap_boot']['lo95']:.5f}, "
          f"{hz['ap_boot']['hi95']:.5f}]")
    print(f"  baseline  {best}  {hz['baseline_ap']:.5f}  95% "
          f"[{hz['baseline_boot']['lo95']:.5f}, {hz['baseline_boot']['hi95']:.5f}]")
    print(f"  vs base   {hz['vs_baseline_point']:.3f}x  95% "
          f"[{hz['vs_baseline_boot']['lo95']:.2f}x, "
          f"{hz['vs_baseline_boot']['hi95']:.2f}x]"
          + ("   <-- interval includes 1.0, no demonstrated edge"
             if hz["vs_baseline_boot"]["lo95"] <= 1 else ""))
    out["hazards"][haz] = hz

FEATS = V.FEATS
MONO = tuple(1 if f.startswith(("prcp", "ratio", "log")) else 0 for f in FEATS)
for haz in V.HAZARDS:
    tr, va, te = S("train", haz), S("val", haz), S("test", haz)
    yc = f"y_{haz}"
    aps, iters = [], []
    print(f"\n=== {haz} seed sweep ===")
    for sd in SEEDS:
        t0 = time.time()
        m = xgb.XGBClassifier(monotone_constraints=MONO, random_state=sd, **XGB)
        m.fit(tr[FEATS], tr[yc], verbose=False,
              eval_set=[(tr[FEATS], tr[yc]), (va[FEATS], va[yc])])
        a = float(average_precision_score(te[yc], m.predict_proba(te[FEATS])[:, 1]))
        aps.append(a)
        iters.append(int(m.best_iteration))
        print(f"  seed {sd}  AP={a:.5f}  best_iter={m.best_iteration}  "
              f"{time.time() - t0:.0f}s")
    a = np.array(aps)
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    out["hazards"][haz]["seed_sweep"] = {
        "ap": aps, "best_iteration": iters, "mean": float(a.mean()), "sd": sd,
        "min": float(a.min()), "max": float(a.max()),
        "spread": float(a.max() - a.min())}
    print(f"  mean={a.mean():.5f}  sd={'n/a' if sd is None else f'{sd:.5f}'}  "
          f"range={a.min():.5f}-{a.max():.5f}")

dest = pathlib.Path("artifacts/audit_uncertainty.json")
dest.write_text(json.dumps(out, indent=1))
print(f"\nwrote {dest}")
for haz, hz in out["hazards"].items():
    s = hz["seed_sweep"]
    sd = "n/a" if s["sd"] is None else f"{s['sd']:.5f}"
    print(f"{haz:6s} AP {hz['ap_rescored']:.5f} "
          f"[{hz['ap_boot']['lo95']:.5f}, {hz['ap_boot']['hi95']:.5f}] "
          f"on {hz['episodes']} episodes   "
          f"vs {hz['best_baseline']} {hz['vs_baseline_point']:.3f}x "
          f"[{hz['vs_baseline_boot']['lo95']:.2f}, "
          f"{hz['vs_baseline_boot']['hi95']:.2f}]   "
          f"seed sd {sd}")

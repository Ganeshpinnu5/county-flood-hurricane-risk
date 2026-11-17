"""Audit: how much of the reported skill comes from forward OBSERVED rainfall?

Reads the sealed run's config out of its manifest, then refits twice: once on
the published feature list and once with the forward-rain features removed. The
gap is the part of the headline number that is not available at forecast time.

Both arms are refits, not the sealed boosters reloaded, so the "as published"
column is a reproduction rather than the published figure itself. It is printed
anyway: the two arms are only comparable to each other if they were fitted the
same way, and a reproduction landing on the headline is the check that this
script still replicates the run.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import v2_models as V

FEATS = list(V.FEATS)
FWD = ["prcp_fwd1", "prcp_fwd3", "prcp_fwd3_max", "ratio_fwd3", "log_fwd3"]
MONO_DOWN = {"cyc_dist_min", "cyc_dist1", "cyc_dist2", "cyc_dist3",
             "cyc_cone_margin"}

# Read out of the sealed manifest rather than copied here. A copy is what this
# file used to hold, and it silently stopped matching the moment the pipeline
# gained a split embargo: the ablation went on reporting untrimmed splits under
# a docstring promising the run's config exactly, which would have put the two
# columns of the published table on different test sets.
MAN = json.loads((V.run_dir() / "manifest.json").read_text())
P = MAN["params"]
XGB = dict(n_estimators=P["xgb"]["rounds"], learning_rate=P["xgb"]["lr"],
           max_depth=P["xgb"]["max_depth"],
           min_child_weight=P["xgb"]["min_child_weight"],
           subsample=P["xgb"]["subsample"],
           colsample_bytree=P["xgb"]["colsample_bytree"],
           reg_lambda=P["xgb"]["reg_lambda"], random_state=P["seed"],
           n_jobs=-1, tree_method="hist", eval_metric="aucpr",
           early_stopping_rounds=P["xgb"]["early"])
YEARS = {k: tuple(v) for k, v in P["years"].items()}
LABEL_END = {h: int(y) for h, y in (P.get("label_end") or {}).items()}
EMBARGO = (max([P["embargo_days"], *P["leads"]]) if P.get("embargo_days")
           else 0)

panel = pd.read_parquet("artifacts/daily_panel.parquet")
panel["year"] = panel["DATE"].dt.year
lo = min(s[0] for s in YEARS.values())
hi = max(s[1] for s in YEARS.values())
panel = panel[panel["year"].between(lo, hi)]
panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
if EMBARGO:
    panel = panel[panel["DATE"] <= panel["DATE"].max()
                  - pd.Timedelta(days=EMBARGO)]
_blocks = sorted(YEARS.items(), key=lambda kv: kv[1][0])
SPLIT = {}
for _i, (_name, _span) in enumerate(_blocks):
    _part = panel[panel["year"].between(*_span)]
    if EMBARGO and _i + 1 < len(_blocks):
        _part = _part[_part["DATE"] < pd.Timestamp(year=_blocks[_i + 1][1][0],
                                                   month=1, day=1)
                      - pd.Timedelta(days=EMBARGO)]
    SPLIT[_name] = _part


def S(name, haz):
    part = SPLIT[name]
    cap = LABEL_END.get(haz)
    return part if cap is None else part[part["year"] <= cap]


def fit(haz, feats, label):
    tr, va, te = S("train", haz), S("val", haz), S("test", haz)
    y = f"y_{haz}"
    mono = tuple(1 if f.startswith(("prcp", "ratio", "log"))
                 else -1 if f in MONO_DOWN else 0 for f in feats)
    m = xgb.XGBClassifier(monotone_constraints=mono, **XGB)
    m.fit(tr[feats], tr[y], eval_set=[(tr[feats], tr[y]), (va[feats], va[y])],
          verbose=False)
    p = m.predict_proba(te[feats])[:, 1]
    ap = average_precision_score(te[y], p)
    prev = te[y].mean()
    imp = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:6]
    return dict(hazard=haz, variant=label, n_feats=len(feats), test_ap=ap,
                prevalence=prev, lift=ap / prev, best_iter=int(m.best_iteration),
                top=[(f, round(float(v), 4)) for f, v in imp])


out = []
for haz in V.HAZARDS:
    te = S("test", haz)
    # single-feature baselines, forward and backward separately
    base = {}
    for f in ("ratio_fwd3", "prcp_fwd3", "log_fwd3", "ratio_prev3",
              "prcp_prev3", "prcp_prev30", "ratio_prev30"):
        v = te[f].fillna(-1).to_numpy()
        base[f] = float(average_precision_score(te[f"y_{haz}"], v))
    prev = float(te[f"y_{haz}"].mean())
    print(f"\n=== {haz}  prevalence {prev:.5f} ===")
    print("  single-feature baselines (test AP):")
    for f, v in sorted(base.items(), key=lambda x: -x[1]):
        print(f"    {f:14s} {v:.5f}  ({v/prev:5.1f}x)")

    for label, feats in (("ALL (as published)", FEATS),
                         ("NO forward rain", [f for f in FEATS if f not in FWD])):
        r = fit(haz, feats, label)
        out.append(r)
        print(f"  {label:20s} AP={r['test_ap']:.5f}  lift={r['lift']:5.1f}x  "
              f"rounds={r['best_iter']}")
        print(f"      top: {r['top']}")

print("\n=== summary ===")
for r in out:
    print(f"{r['hazard']:6s} {r['variant']:20s} AP={r['test_ap']:.5f} "
          f"lift={r['lift']:5.1f}x rounds={r['best_iter']}")
# Stamped with the run it replicates. Without this the file is just two numbers
# per hazard, and nothing stops a stale copy being read into the report of a
# later run -- which is the drift this audit exists to rule out.
json.dump({"run_id": MAN["run_id"], "feats": len(FEATS), "embargo_days": EMBARGO,
           "variants": out},
          open("artifacts/audit_fwd_ablation.json", "w"), indent=1)

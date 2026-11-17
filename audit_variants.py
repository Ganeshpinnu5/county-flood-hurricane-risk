"""Audit: what the embargo costs, what dropping SVI costs, and what the
cyclone forecast features buy.

Both changes are defended on principle rather than on skill, which is exactly
the case where a number is needed: a reviewer is entitled to know whether
"we removed the indefensible feature" also means "we lost half the result".

Fitting them as separate variants rather than folding both into one retrain is
the point. A single run that changes two things cannot say which one moved the
number.

    embargo      -- the 2014 tail no longer holds 2015 answers
    no_svi       -- RPL_THEME1/4/THEMES out. Whether a cyclone crosses a county
                    is not a function of its social vulnerability. SVI belongs
                    to the consequence stage, where it describes who is harmed
                    by a hazard that has already been predicted.
    no_socio     -- also drops pop_density. Conceptually the worse offender:
                    dense counties report more flood episodes partly because
                    there is someone there to report them, so the model can
                    score reporting intensity and call it hazard. Measured
                    separately because removing it is a bigger claim than the
                    review asked for.
    no_cyclone   -- the cyc_* block out, leaving the feature set the panel had
                    before the forecast track was added. Run against both
                    hazards, not just the hurricane one: whether a cyclone
                    forecast helps the flood model is the other half of the
                    question and nobody has measured it.
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import cyclone_features
import v2_models as V

FEATS = list(V.FEATS)
SVI = ["RPL_THEME1", "RPL_THEME4", "RPL_THEMES"]
CYC = list(cyclone_features.FEATURES)
MONO_DOWN = {"cyc_dist_min", "cyc_dist1", "cyc_dist2", "cyc_dist3",
             "cyc_cone_margin"}
YEARS = {"train": (2000, 2014), "val": (2015, 2019), "test": (2020, 2026)}
LABEL_END = {"storm": 2024}
LEADS = (1, 2, 3)
EMBARGO = 3
XGB = dict(n_estimators=2000, learning_rate=0.05, max_depth=6,
           min_child_weight=50, subsample=0.9, colsample_bytree=0.9,
           reg_lambda=5.0, random_state=0, n_jobs=-1, tree_method="hist",
           eval_metric="aucpr", early_stopping_rounds=50)
BASELINE_FEATS = ("ratio_fwd3", "prcp_fwd3", "prcp_fwd3_max", "prcp_prev30",
                  "ratio_prev3", "log_fwd3", "pop_density")

VARIANTS = {
    "published": dict(embargo=0, drop=[]),
    "embargo": dict(embargo=EMBARGO, drop=[]),
    "no_svi": dict(embargo=0, drop=SVI),
    "no_svi_embargo": dict(embargo=EMBARGO, drop=SVI),
    "no_socio_embargo": dict(embargo=EMBARGO, drop=SVI + ["pop_density"]),
    "no_cyclone": dict(embargo=0, drop=CYC),
}

# Naming variants on the command line runs only those. The full sweep is
# twelve fits on a 3.2M-row panel and a desktop with browsers open does not
# reliably have the memory for it; being able to resume the tail of the sweep
# without refitting the variants that already succeeded is the difference
# between a lost afternoon and a five-minute rerun.
WANT = [v for v in sys.argv[1:] if not v.startswith("-")] or list(VARIANTS)
unknown = [v for v in WANT if v not in VARIANTS]
if unknown:
    sys.exit(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")

# Read only what is actually fitted. The panel carries 70 columns and this
# needs about 50; the rest were being loaded and then ignored.
NEEDED = sorted({"county_fips", "DATE", *FEATS, *BASELINE_FEATS,
                 *(f"y_{h}" for h in V.HAZARDS)})
panel = pd.read_parquet("artifacts/daily_panel.parquet", columns=NEEDED)
panel["year"] = panel["DATE"].dt.year
panel = panel[panel["year"].between(2000, 2026)]
panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
# float64 buys nothing here and doubles the resident set.
f64 = panel.select_dtypes("float64").columns
panel[f64] = panel[f64].astype("float32")


def splits(embargo):
    """Replicates pipeline.py's split construction at a given embargo."""
    p = panel
    if embargo:
        p = p[p["DATE"] <= p["DATE"].max() - pd.Timedelta(days=embargo)]
    blocks = sorted(YEARS.items(), key=lambda kv: kv[1][0])
    out, yr = {}, p["year"]
    for i, (name, span) in enumerate(blocks):
        part = p[yr.between(*span)]
        if embargo and i + 1 < len(blocks):
            part = part[part["DATE"] < pd.Timestamp(
                year=blocks[i + 1][1][0], month=1, day=1)
                - pd.Timedelta(days=embargo)]
        out[name] = part
    return out


def fit(sp, haz, feats):
    def S(name):
        part = sp[name]
        cap = LABEL_END.get(haz)
        return part if cap is None else part[part["year"] <= cap]

    tr, va, te = S("train"), S("val"), S("test")
    y = f"y_{haz}"
    # Same directions pipeline.py trains with: more rain and a closer
    # forecast track both have to raise predicted risk, never lower it.
    mono = tuple(1 if f.startswith(("prcp", "ratio", "log"))
                 else -1 if f in MONO_DOWN else 0
                 for f in feats)
    m = xgb.XGBClassifier(monotone_constraints=mono, **XGB)
    m.fit(tr[feats], tr[y], eval_set=[(tr[feats], tr[y]), (va[feats], va[y])],
          verbose=False)
    p = m.predict_proba(te[feats])[:, 1]
    yt = te[y].to_numpy()
    ap = float(average_precision_score(yt, p))
    prev = float(yt.mean())
    base = {f: float(average_precision_score(
        yt, te[f].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()))
        for f in BASELINE_FEATS if f in te.columns}
    best = max(base, key=base.get)
    return dict(
        n_feats=len(feats), test_ap=ap, prevalence=prev, lift=ap / prev,
        best_iteration=int(m.best_iteration),
        best_baseline=best, best_baseline_ap=base[best],
        vs_best_baseline=ap / base[best] if base[best] else None,
        n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
        n_train_pos=int(tr[y].sum()), n_val_pos=int(va[y].sum()),
        n_test_pos=int(yt.sum()),
        top=[(f, round(float(v), 4)) for f, v in
             sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:6]])


dest = pathlib.Path("artifacts/audit_variants.json")
out = {"run_id": V.RUN_ID, "embargo_days": EMBARGO, "variants": {}}
if dest.exists():
    # Carry forward variants fitted on an earlier invocation so a resumed
    # sweep still ends with one complete file. A matching run id is not
    # sufficient: the panel gained the cyc_* block without the run id moving,
    # so entries recorded against the narrower feature set compared as though
    # they were cyclone-era fits and understated storm AP by a factor of three.
    # Each entry therefore carries the exact feature list it was fitted on,
    # and anything that disagrees with what this config would build now is
    # discarded rather than merged.
    prev = json.loads(dest.read_text())
    if prev.get("run_id") == V.RUN_ID:
        out["variants"] = {
            k: v for k, v in prev.get("variants", {}).items()
            if k not in WANT and k in VARIANTS
            and v.get("feats") == [f for f in FEATS
                                   if f not in VARIANTS[k]["drop"]]}

for name in WANT:
    cfg = VARIANTS[name]
    sp = splits(cfg["embargo"])
    feats = [f for f in FEATS if f not in cfg["drop"]]
    out["variants"][name] = {"embargo": cfg["embargo"], "dropped": cfg["drop"],
                             "feats": feats, "hazards": {}}
    print(f"\n=== {name}  embargo={cfg['embargo']}d  "
          f"feats={len(feats)}  dropped={cfg['drop'] or 'none'}")
    print(f"    train ends {sp['train']['DATE'].max().date()}  "
          f"val ends {sp['val']['DATE'].max().date()}  "
          f"test ends {sp['test']['DATE'].max().date()}")
    for haz in V.HAZARDS:
        t0 = time.time()
        r = fit(sp, haz, feats)
        out["variants"][name]["hazards"][haz] = r
        print(f"  {haz:6s} AP={r['test_ap']:.5f}  lift={r['lift']:5.1f}x  "
              f"vs {r['best_baseline']}={r['vs_best_baseline']:.2f}x  "
              f"iter={r['best_iteration']:>3d}  "
              f"pos={r['n_test_pos']:,}  {time.time() - t0:.0f}s")

print("\n=== change against the published variant ===")
ref = out["variants"].get("published", {}).get("hazards")
for name, v in (out["variants"].items() if ref else []):
    if name == "published":
        continue
    for haz, r in v["hazards"].items():
        d = r["test_ap"] - ref[haz]["test_ap"]
        print(f"{name:18s} {haz:6s} AP {r['test_ap']:.5f} "
              f"({d:+.5f}, {d / ref[haz]['test_ap'] * 100:+.2f}%)   "
              f"vs base {r['vs_best_baseline']:.2f}x "
              f"(was {ref[haz]['vs_best_baseline']:.2f}x)")

dest.write_text(json.dumps(out, indent=1))
print(f"\nwrote {dest}")

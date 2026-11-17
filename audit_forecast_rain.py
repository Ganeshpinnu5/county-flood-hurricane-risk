"""Audit B13: what does the published model score on forecast rain?

The panel's prcp_fwd* features come from the same GHCN record as the labels, so
they are the rain that fell, not the rain a forecaster expected. The published
boosters were fitted on them and are scored on them, which makes every headline
figure an upper bound rather than an estimate of deployed skill.

This scores the published boosters twice over the *same* county-days: once on
the observed forward window as published, once with the five forward features
rebuilt from archived forecasts that were actually available at the origin
date. No refitting -- the model is held fixed so the difference is attributable
to the input and nothing else. AP on observed rain is the ceiling; AP on
forecast rain is the number a deployed system could have earned.

Either of two archives can supply the forward window, selected by
CRISK_FORECAST_ARM, and neither covers the whole test period:

- `gefs`, the default: NOAA's operational GEFS ensemble mean, from the
  2020-09-23 start of v12. Leaves 60 of the 65 flood episodes and 19 of the 23
  storm ones, so it is the arm with enough events to carry a result.
- `open-meteo`: the previous-runs archive, from 2024-01-25. Leaves 23 flood
  episodes and only 6 storm ones. Too few for the storm claim, but it is an
  independently produced forecast over those days, which makes it worth having
  as a check on the GEFS arm rather than a substitute for it.

Whichever arm runs, the observed-rain arm is re-scored on exactly the days that
arm covers rather than quoted from the README. Quoting the published figure
would confound the change of input with the change of days, which is the one
thing this script exists to avoid.

The per-county climatology that divides ratio_fwd3 is held at its published
value, recovered from the panel rather than recomputed. It is a frozen scaling
constant the booster was fitted against; rebuilding it from forecast rain would
change the function being scored and stop this being a controlled comparison.
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import episode_ci as E
import v2_models as V

DRAWS = int(os.environ.get("CRISK_BOOT_DRAWS", 2000))
YEARS = {"test": (2020, 2026)}
LABEL_END = {"storm": 2024}
LEADS = (1, 2, 3)
FWD = ["prcp_fwd1", "prcp_fwd3", "prcp_fwd3_max", "ratio_fwd3", "log_fwd3"]
BASELINE_FEATS = ("ratio_fwd3", "prcp_fwd3", "prcp_fwd3_max", "prcp_prev30",
                  "ratio_prev3", "log_fwd3", "pop_density")
# Two archives answer this question -- Open-Meteo's previous-runs and the
# operational GEFS on S3 -- and they cover different spans, so which one is
# being scored is a parameter rather than a constant.
ARMS = {
    "open-meteo": ("artifacts/forecast_rain.parquet",
                   "open-meteo previous-runs, lead-resolved"),
    "gefs": ("artifacts/gefs_forecast_rain.parquet",
             "NOAA GEFS v12 ensemble mean, 0.5 deg, 00Z run"),
}
ARM = os.environ.get("CRISK_FORECAST_ARM", "gefs")
if ARM not in ARMS:
    raise SystemExit(f"CRISK_FORECAST_ARM must be one of {sorted(ARMS)}")
SRC = pathlib.Path(ARMS[ARM][0])
DEST = pathlib.Path(f"artifacts/audit_forecast_rain_{ARM}.json")

if not SRC.exists():
    raise SystemExit(f"{SRC} not found -- fetch the {ARM} archive first")

panel = pd.read_parquet("artifacts/daily_panel.parquet")
panel["year"] = panel["DATE"].dt.year
panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)

fc = pd.read_parquet(SRC).sort_values(["county_fips", "DATE"]).reset_index(drop=True)
print(f"forecast archive: {len(fc):,} county-days  "
      f"{fc['county_fips'].nunique()} counties  "
      f"{fc['DATE'].min().date()}..{fc['DATE'].max().date()}")

# The forecast for t+k issued at t sits on row t+k of the lead-k column, so
# shifting each lead back by its own horizon lands all three on origin t --
# which is the only arrangement where every value was knowable at t.
g = fc.groupby("county_fips", sort=False)
for k in LEADS:
    fc[f"f{k}"] = g[f"precipitation_previous_day{k}"].shift(-k)
fc["fc_fwd1"] = fc["f1"]
fc["fc_fwd3"] = fc[["f1", "f2", "f3"]].sum(axis=1, min_count=3)
fc["fc_fwd3_max"] = fc[["f1", "f2", "f3"]].max(axis=1)

# p95 of the observed forward window, per county, frozen at its published
# value. Recovered by inverting the ratio the panel already carries so there is
# no second definition of it in the tree.
den = panel.loc[panel["ratio_fwd3"].gt(0) & panel["prcp_fwd3"].gt(0),
                ["county_fips", "prcp_fwd3", "ratio_fwd3"]].copy()
den["p95_fwd3"] = den["prcp_fwd3"] / den["ratio_fwd3"]
p95 = den.groupby("county_fips")["p95_fwd3"].median()
spread = den.groupby("county_fips")["p95_fwd3"].agg(lambda s: s.max() - s.min())
assert spread.max() < 1e-6, f"p95_fwd3 is not constant per county: {spread.max()}"

fc = fc.merge(p95.rename("p95_fwd3"), on="county_fips", how="left")
fc["fc_ratio_fwd3"] = fc["fc_fwd3"] / fc["p95_fwd3"].replace(0, np.nan)
fc["fc_log_fwd3"] = np.log1p(fc["fc_fwd3"].clip(lower=0))

keep = ["county_fips", "DATE", "fc_fwd1", "fc_fwd3", "fc_fwd3_max",
        "fc_ratio_fwd3", "fc_log_fwd3"]
scorers = V.load_scorers()
out = {"run_id": V.RUN_ID, "draws": DRAWS, "gap_days": E.GAP_DAYS,
       "forecast_arm": ARM, "forecast_source": ARMS[ARM][1],
       "forecast_start": str(fc["DATE"].min().date()), "hazards": {}}


def ap(y, p):
    return float(average_precision_score(y, p))


def feat(frame, f):
    """A single feature as a scorable column, nulls and infinities flattened."""
    return frame[f].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()


for haz in V.HAZARDS:
    te = panel[panel["year"].between(*YEARS["test"])]
    cap = LABEL_END.get(haz)
    if cap is not None:
        te = te[te["year"] <= cap]
    # Inner join, then drop rows the archive could not complete: a forward
    # window missing one of its three days is not a forecast.
    j = te.merge(fc[keep], on=["county_fips", "DATE"], how="inner")
    j = j.dropna(subset=["fc_fwd1", "fc_fwd3", "fc_fwd3_max",
                         "fc_ratio_fwd3", "fc_log_fwd3"])
    if j.empty:
        print(f"\n=== {haz}: no overlap with the forecast archive")
        continue

    y = j[f"y_{haz}"].to_numpy()
    blk, nb = E.blocks(j, f"{haz}_onset")
    if nb == 0:
        print(f"\n=== {haz}: no onset dates in the overlap, nothing to score")
        continue

    # Same rows, same model, same frozen climatology -- only the forward window
    # changes, so the gap is the cost of not knowing the future.
    obs = j.copy()
    fcs = j.copy()
    for col, src in zip(FWD, ["fc_fwd1", "fc_fwd3", "fc_fwd3_max",
                              "fc_ratio_fwd3", "fc_log_fwd3"]):
        fcs[col] = j[src].to_numpy()

    p_obs = scorers[haz](obs)
    p_fc = scorers[haz](fcs)
    ap_obs, ap_fc = ap(y, p_obs), ap(y, p_fc)

    # Identical seeds mean identical block draws, so the difference is paired
    # and its interval is the interval of the comparison rather than two
    # intervals added together.
    b_obs = E.boot(y, p_obs, blk, nb, np.random.default_rng(0), DRAWS)
    b_fc = E.boot(y, p_fc, blk, nb, np.random.default_rng(0), DRAWS)
    d_boot = b_obs - b_fc
    r_boot = b_fc / b_obs

    # The baseline has to be handed the same forecast the model is handed.
    # Scoring a forecast-fed booster against an observation-fed single feature
    # would be the published mistake wearing a different hat: the baseline
    # would be the only one of the two allowed to see the rain that fell.
    # Selection runs on the forecast frame for the same reason. Three of the
    # candidates are past-rain or static and so identical in both frames --
    # which is the point, since those are the features a deployed baseline
    # really does know at the origin date.
    best = max(BASELINE_FEATS, key=lambda f: ap(y, feat(fcs, f)))
    bv = feat(fcs, best)
    b_base = E.boot(y, bv, blk, nb, np.random.default_rng(0), DRAWS)
    vs_base_boot = b_fc / b_base

    # The observation-fed baseline is kept beside it because that is the
    # quantity the published lift is measured against, so it is the bridge
    # between this table and the README's.
    best_obs = max(BASELINE_FEATS, key=lambda f: ap(y, feat(obs, f)))
    ap_base_obs = ap(y, feat(obs, best_obs))

    hz = {
        "rows": int(len(j)), "positives": int(y.sum()), "episodes": int(nb),
        "base_rate": float(y.mean()),
        "dates": [str(j["DATE"].min().date()), str(j["DATE"].max().date())],
        "ap_observed": ap_obs, "ap_forecast": ap_fc,
        "ap_observed_boot": E.pct(b_obs), "ap_forecast_boot": E.pct(b_fc),
        "drop_point": ap_obs - ap_fc, "drop_boot": E.pct(d_boot),
        "retained_point": ap_fc / ap_obs if ap_obs else None,
        "retained_boot": E.pct(r_boot),
        "best_baseline": best, "baseline_ap": ap(y, bv),
        "baseline_boot": E.pct(b_base),
        "forecast_vs_baseline": ap_fc / ap(y, bv) if ap(y, bv) else None,
        "vs_baseline_boot": E.pct(vs_base_boot),
        "best_baseline_observed": best_obs, "baseline_ap_observed": ap_base_obs,
        "forecast_vs_baseline_observed": (
            ap_fc / ap_base_obs if ap_base_obs else None),
        # The correlation between what was forecast and what fell is the
        # mechanism behind the gap, and it is the part that would improve with
        # a better forecast product rather than a better model.
        "fwd3_corr": float(np.corrcoef(j["prcp_fwd3"].fillna(0),
                                       j["fc_fwd3"].fillna(0))[0, 1]),
        "fwd1_corr": float(np.corrcoef(j["prcp_fwd1"].fillna(0),
                                       j["fc_fwd1"].fillna(0))[0, 1]),
        # Precision among the highest-ranked rows of each single rain feature.
        # This is where the two hazards part company, and an AP on its own
        # hides it: average precision is dominated by the top of the ranking,
        # so the question that matters is which feature puts real events there.
        "precision_at_k": {
            str(k): {"observed": float(y[np.argsort(
                         -feat(obs, "prcp_fwd3"))[:k]].mean()),
                     "forecast": float(y[np.argsort(
                         -feat(fcs, "prcp_fwd3"))[:k]].mean())}
            for k in (200, 1000, 5000)},
        # Section 3.2 showed 23 episodes cannot pin a hurricane AP tighter
        # than about +/-0.01, so anything at or below that is reported and not
        # claimed. The bar is on episodes rather than rows because a cyclone
        # writes one event into thousands of county-days.
        "underpowered": bool(nb < 25),
    }
    out["hazards"][haz] = hz

    print(f"\n=== {haz}  rows {len(j):,}  positives {int(y.sum()):,}  "
          f"episodes {nb}  {hz['dates'][0]}..{hz['dates'][1]}")
    print(f"  AP observed  {ap_obs:.5f}  95% "
          f"[{hz['ap_observed_boot']['lo95']:.5f}, "
          f"{hz['ap_observed_boot']['hi95']:.5f}]   <- as published")
    print(f"  AP forecast  {ap_fc:.5f}  95% "
          f"[{hz['ap_forecast_boot']['lo95']:.5f}, "
          f"{hz['ap_forecast_boot']['hi95']:.5f}]   <- deployable")
    # Observed rain is the ceiling only where it is the better input, and it
    # is not for every hazard. Calling the gap a "drop" and the share
    # "retained" regardless of sign is how a result like the storm arm's gets
    # printed as "235% retained", which reads as a mistake rather than a
    # finding.
    if hz["drop_point"] >= 0:
        print(f"  cost of not knowing the future  {hz['drop_point']:.5f} AP "
              f"95% [{hz['drop_boot']['lo95']:.5f}, "
              f"{hz['drop_boot']['hi95']:.5f}]")
        print(f"  retained     {hz['retained_point'] * 100:.1f}%  95% "
              f"[{hz['retained_boot']['lo95'] * 100:.1f}%, "
              f"{hz['retained_boot']['hi95'] * 100:.1f}%]")
    else:
        print(f"  forecast rain is the BETTER input here, by "
              f"{-hz['drop_point']:.5f} AP  95% "
              f"[{-hz['drop_boot']['hi95']:.5f}, "
              f"{-hz['drop_boot']['lo95']:.5f}]")
        print(f"  so observed forward rain is not this hazard's ceiling, and "
              f"the published input is costing it skill")
    print(f"  baseline     {best} (forecast-fed) {hz['baseline_ap']:.5f}  95% "
          f"[{hz['baseline_boot']['lo95']:.5f}, "
          f"{hz['baseline_boot']['hi95']:.5f}]")
    print(f"  model/base   {hz['forecast_vs_baseline']:.2f}x  95% "
          f"[{hz['vs_baseline_boot']['lo95']:.2f}x, "
          f"{hz['vs_baseline_boot']['hi95']:.2f}x]"
          + ("   <-- interval includes 1.0, no demonstrated edge"
             if hz["vs_baseline_boot"]["lo95"] <= 1 else ""))
    print(f"  for scale    observation-fed baseline {best_obs} "
          f"{ap_base_obs:.5f}, which the forecast model reaches "
          f"{hz['forecast_vs_baseline_observed']:.2f}x of")
    print(f"  fwd3 corr forecast-vs-observed {hz['fwd3_corr']:.3f}  "
          f"(lead 1: {hz['fwd1_corr']:.3f})")
    pk = hz["precision_at_k"]
    print("  precision at top-k of prcp_fwd3, observed vs forecast: " + "  ".join(
        f"k={k} {pk[k]['observed']:.1%}/{pk[k]['forecast']:.1%}" for k in pk))
    if hz["underpowered"]:
        print(f"  NOTE: {nb} episodes is too few to measure; reported, not claimed")

DEST.write_text(json.dumps(out, indent=1))
print(f"\nwrote {DEST}")

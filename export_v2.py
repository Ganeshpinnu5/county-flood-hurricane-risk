"""Export the dashboard payload for the V2 models.

Replaces export_proto.py, which served the V1 prototype: a FEMA-declaration
target, 2010-2019 training, and a one-feature logistic for flood. None of that
is what V2 trained, so the old exporter could only have published numbers the
paper does not contain.

Three things changed that matter here:

  label   NOAA StormEvents flood episodes and HURDAT2 cyclone passes, not FEMA
          declarations. A declaration is an administrative act that depends on
          damage estimates and a governor's request, so V1 scored counties for
          paperwork as much as for weather.
  models  XGBoost for both hazards, from a sealed hash-verified run, plus the
          exact-lead models the lead sweep produced.
  leads   A separate model per lead, so the dashboard can show the +1/+2/+3 day
          forecasts rather than implying the pooled 3-day number holds at day 3.

Shipping risk for every county-day would be ~3.2M rows. Instead we ship the
windows around the events the dashboard actually lists, plus the most recent
observed stretch, which is what a visitor can navigate to.
"""

import json
import pathlib

import numpy as np
import pandas as pd

import v2_models
from v2_models import FEATS, HAZARDS, LEADS

ARTIFACTS = pathlib.Path("artifacts")
DOCS = pathlib.Path("docs/data")

WINDOW = 5          # days either side of a listed event
RECENT_DAYS = 60
PER_COUNTY = {"flood": 6, "storm": 4}   # events listed per county, per hazard
BAND_SAMPLE = 150_000

# Onset dates further apart than this start a new weather episode. Seven days
# is past the lifetime of a rain system crossing the Gulf and past the time a
# single cyclone spends over land, so clusters below it are one event seen from
# several counties.
EPISODE_GAP_DAYS = 7


def fips5(s):
    """Both event tables key on county, one as int and one already padded."""
    return s.astype("int64").astype(str).str.zfill(5) if s.dtype.kind in "iu" \
        else s.astype(str).str.zfill(5)


def load_events():
    """One table of real hazard occurrences, both hazards, with a display title.

    These are the observational records the labels were built from, so an event
    listed here is by construction an event the model was scored against -- the
    dashboard cannot show an outcome the training never saw.
    """
    fl = pd.read_parquet(ARTIFACTS / "noaa" / "noaa_flood_events.parquet")
    fl["county_fips"] = fips5(fl["county_fips"])
    fl["hazard"] = "flood"
    fl["type"] = "Flood"
    fl["title"] = ("NOAA StormEvents flood episode "
                   + fl["EPISODE_ID"].astype("int64").astype(str))

    st = pd.read_parquet(ARTIFACTS / "noaa" / "hurdat_events.parquet")
    st["county_fips"] = fips5(st["county_fips"])
    st["hazard"] = "storm"
    st["type"] = "Tropical cyclone"
    st["title"] = ("HURDAT2 track " + st["storm_id"].astype(str)
                   + ", peak wind " + st["peak_wind_kt"].round().astype("int64")
                     .astype(str) + " kt over the county")

    cols = ["county_fips", "begin", "end", "hazard", "type", "title"]
    ev = pd.concat([fl[cols], st[cols]], ignore_index=True)
    ev["begin"] = pd.to_datetime(ev["begin"]).dt.normalize()
    ev["end"] = pd.to_datetime(ev["end"]).dt.normalize()
    ev["end"] = ev[["begin", "end"]].max(axis=1)

    # One NOAA episode spans several rows per county, one per sub-event inside
    # it, and HURDAT lists a county once per six-hourly track point. Left as-is
    # the same flood turns up three times in a county's history and crowds out
    # distinct events, so collapse to one row per county, hazard and start day,
    # keeping the longest duration seen.
    ev = (ev.sort_values("end", ascending=False)
            .drop_duplicates(["county_fips", "hazard", "begin"], keep="first"))
    return ev.sort_values("begin", ascending=False).reset_index(drop=True)


def listed(ev, covered):
    """The most recent events per county and hazard, capped.

    Capped per hazard rather than overall: a Gulf county logs far more flood
    episodes than cyclone passes, so a single recency cap would list floods only
    and hide the hurricane side of the model entirely.
    """
    ev = ev[ev["county_fips"].isin(covered)]
    keep = [g.head(PER_COUNTY[h]) for (_, h), g
            in ev.groupby(["county_fips", "hazard"], sort=False)]
    return pd.concat(keep, ignore_index=True)


def county_meta(panel, geo):
    """Every county the map draws, flagged for whether a model covers it.

    Names and centroids come from the map geometry so the whole lower 48 is
    present; modelled counties override the centroid with the mean position of
    their rainfall stations, which is the point the features were built from.
    """
    out = {f: [n[0], n[1], n[2], n[3], 0] for f, n in geo["names"].items()}
    stn = panel.groupby("county_fips")[["lat", "lon"]].mean()
    for fips, r in stn.iterrows():
        base = out.get(fips)
        name = base[0] if base else fips
        state = base[1] if base else ""
        out[fips] = [name, state, round(float(r.lat), 4),
                     round(float(r.lon), 4), 1]
    return out


def live_run_id():
    """The run that scored the live page, or None if it has not been built.

    meta.json and live.json are written by different programs, so nothing
    structural stops one from describing a model the other is not running. That
    is exactly what happened once: live.json was repointed at a corrected run
    while meta.json kept reporting the superseded one's accuracy, so the page
    published a model card for a model it had stopped using. Checked here, at
    the start, because the alternative is a visitor finding it.
    """
    p = DOCS / "live.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("run_id")


def live_n_counties():
    """How many counties the live page actually scores.

    The gap between this and the fitted count is the project's largest caveat,
    so the model card states both. Read from live.json rather than assumed,
    because it is the file that decides how many counties get a probability.
    """
    p = DOCS / "live.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("n_counties")


def main():
    prm = v2_models.results("params_run.json")
    xgb_res = v2_models.results("xgb_run.json")
    lead_res = v2_models.results("leads_run.json")

    # Optional: a run that trained no transformer publishes no transformer
    # numbers. Borrowing them from a run with a different label definition
    # would turn the architecture comparison into a label comparison.
    tft_res = v2_models.results_opt("tft_run.json")
    tft_lead_res = v2_models.results_opt("tft_leads_run.json")
    has_tft = tft_res is not None and tft_lead_res is not None
    if not has_tft:
        print("no transformer results in this run -- "
              "every TFT figure will be marked withdrawn")

    here, live = v2_models.provenance()["run_id"], live_run_id()
    if live is not None and live != here:
        raise SystemExit(
            f"run mismatch: exporting meta.json from {here}, but "
            f"docs/data/live.json was scored by {live}.\n"
            f"run `python fetch_live.py` first so both describe one model, "
            f"or set CRISK_RUN_DIR to the run live.json used."
        )

    pooled = v2_models.load_scorers()
    per_lead = v2_models.load_lead_scorers()

    panel = pd.read_parquet(ARTIFACTS / "daily_panel.parquet")
    panel["year"] = panel["DATE"].dt.year
    lo, hi = prm["years"]["train"][0], prm["years"]["test"][1]
    panel = panel[panel["year"].between(lo, hi)]
    panel = panel.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
    print(f"panel {len(panel):,} rows  {panel['year'].min()}-{panel['year'].max()}"
          f"  {panel['county_fips'].nunique()} counties")

    geo = json.loads((DOCS / "geo.json").read_text())
    covered = set(panel["county_fips"].unique())
    events = load_events()
    shown = listed(events, covered)
    print(f"events: {len(events):,} on record, {len(shown):,} listed "
          f"across {shown['county_fips'].nunique()} counties")

    # --- pick the rows worth scoring ---------------------------------------
    pos = pd.Series(np.arange(len(panel)),
                    index=pd.MultiIndex.from_arrays(
                        [panel["county_fips"], panel["DATE"]]))
    want = np.zeros(len(panel), dtype=bool)
    for r in shown.itertuples(index=False):
        days = pd.date_range(r.begin - pd.Timedelta(days=WINDOW),
                             r.end + pd.Timedelta(days=WINDOW), freq="D")
        idx = [i for i in (pos.get((r.county_fips, d), -1) for d in days) if i >= 0]
        want[idx] = True
    last = panel["DATE"].max()
    want |= (panel["DATE"] > last - pd.Timedelta(days=RECENT_DAYS)).to_numpy()

    sub = panel[want].copy()
    print(f"scoring {len(sub):,} of {len(panel):,} county-days")
    for haz in HAZARDS:
        sub[f"p_{haz}"] = pooled[haz](sub)
        for k in LEADS:
            sub[f"p_{haz}_l{k}"] = per_lead[(haz, k)](sub)

    # --- ground truth: was an event in progress on this day? ---------------
    # Taken from the event intervals rather than from a label column: y_{haz}
    # is forward-looking (onset within three days), so it is a target, not a
    # statement about what was happening on the day itself.
    for haz in HAZARDS:
        sub[f"{haz}_active"] = False
    key = pd.MultiIndex.from_arrays([sub["county_fips"], sub["DATE"]])
    at = pd.Series(np.arange(len(sub)), index=key)
    for r in events[events["county_fips"].isin(covered)].itertuples(index=False):
        days = pd.date_range(r.begin, r.end, freq="D")
        idx = [i for i in (at.get((r.county_fips, d), -1) for d in days) if i >= 0]
        if idx:
            sub.iloc[idx, sub.columns.get_loc(f"{r.hazard}_active")] = True

    # --- per-county event list, with what the models said beforehand -------
    split_of = {y: name for name, (a, b) in prm["years"].items()
                for y in range(a, b + 1)}
    scored = sub.set_index(["county_fips", "DATE"])

    def lead_up(fips, begin, haz):
        """Two readings of the same three days before onset.

        `pre` is the pooled model, which answers "onset within three days" and
        so should rise as onset approaches. `lead` is the exact-lead model that
        would actually have been run that morning -- the +3 day model three days
        out, the +1 day model the day before. The second is the operationally
        honest number and is usually the lower of the two.
        """
        pre, lead = [], []
        for k in (3, 2, 1):
            day = begin - pd.Timedelta(days=k)
            try:
                row = scored.loc[(fips, day)]
            except KeyError:
                pre.append(None)
                lead.append(None)
                continue
            pre.append(round(float(row[f"p_{haz}"]), 5))
            lead.append(round(float(row[f"p_{haz}_l{k}"]), 5))
        try:
            rain = scored.loc[(fips, begin - pd.Timedelta(days=1)), "prcp_fwd3"]
            rain = None if pd.isna(rain) else round(float(rain), 1)
        except KeyError:
            rain = None
        return pre, lead, rain

    ev_out = {}
    for r in shown.sort_values("begin", ascending=False).itertuples(index=False):
        pre, lead, rain = lead_up(r.county_fips, r.begin, r.hazard)
        e = {"begin": r.begin.strftime("%Y-%m-%d"),
             "end": r.end.strftime("%Y-%m-%d"),
             "hazard": r.hazard, "type": r.type, "title": r.title[:110],
             "split": split_of.get(r.begin.year)}
        if any(v is not None for v in pre):
            e["pre"] = {r.hazard: pre}
            e["lead"] = {r.hazard: lead}
            e["rain"] = rain
        ev_out.setdefault(r.county_fips, []).append(e)

    n_listed = sum(len(v) for v in ev_out.values())
    n_with = sum(1 for v in ev_out.values() for e in v if "pre" in e)
    print(f"events with a model lead-up: {n_with:,} of {n_listed:,}")

    # A county's panel only holds days its rainfall stations reported, and some
    # stations have multi-year outages -- one county has 3,702 of 9,645 days.
    # Those gaps are why some listed events carry no lead-up, so the number is
    # published rather than left for a visitor to discover as a blank card.
    days_per = panel.groupby("county_fips").size()
    span = int((panel["DATE"].max() - panel["DATE"].min()).days) + 1
    coverage = {
        "calendar_days": span,
        "median_days_per_county": int(days_per.median()),
        "min_days_per_county": int(days_per.min()),
        "counties_under_80pct": int((days_per < 0.8 * span).sum()),
        "events_listed_with_leadup": n_with,
        "events_listed": n_listed,
    }

    # Effective sample size, which is nothing like the row count. Rain and
    # tropical cyclones are spatially correlated, so every county a single
    # system crosses is close to one observation rather than many. AP computed
    # over 765,020 county-days therefore reads as a large-sample figure while
    # resting, on hurricanes, on about 23 storms. Published because a reader
    # otherwise has no way to see it, and because any confidence interval would
    # have to resample these blocks rather than rows.
    episodes = {}
    for haz in HAZARDS:
        t = panel[panel["year"].between(*prm["years"]["test"])]
        cap = prm.get("label_end", {}).get(haz)
        if cap is not None:
            t = t[t["year"] <= int(cap)]
        onset = t[t[f"{haz}_onset"] == 1]
        dates = np.sort(onset["DATE"].unique())
        gaps = np.diff(dates) / np.timedelta64(1, "D") if len(dates) else []
        episodes[haz] = {
            "onset_county_days": int(len(onset)),
            "distinct_dates": int(len(dates)),
            "episodes": int(1 + (np.asarray(gaps) > EPISODE_GAP_DAYS).sum())
                        if len(dates) else 0,
            "gap_days": EPISODE_GAP_DAYS,
        }
        print(f"{haz}: {episodes[haz]['onset_county_days']:,} onset county-days"
              f" = {episodes[haz]['distinct_dates']} dates"
              f" = {episodes[haz]['episodes']} episodes")

    # --- risk by date ------------------------------------------------------
    sub["d"] = sub["DATE"].dt.strftime("%Y-%m-%d")
    cols = (["county_fips", "p_flood", "p_storm", "prcp_fwd3", "ratio_fwd3",
             "flood_active", "storm_active"]
            + [f"p_flood_l{k}" for k in LEADS] + [f"p_storm_l{k}" for k in LEADS])
    by_day = {}
    for day, grp in sub[["d"] + cols].groupby("d", sort=True):
        by_day[day] = [
            [r[0], round(float(r[1]), 5), round(float(r[2]), 5),
             None if pd.isna(r[3]) else round(float(r[3]), 1),
             None if pd.isna(r[4]) else round(float(r[4]), 2),
             int(r[5]), int(r[6])]
            + [round(float(v), 5) for v in r[7:]]
            for r in grp[cols].itertuples(index=False, name=None)
        ]

    # Raw probabilities are tiny because the event itself is rare. Bands are
    # quantiles of an unbiased sample of all county-days, so "High" means high
    # relative to a normal day rather than high in absolute terms. Each lead has
    # its own quantiles: a +3 day model sees roughly a third of the pooled
    # prevalence, so reusing the pooled cut-offs would paint every day green.
    samp = panel.sample(n=min(BAND_SAMPLE, len(panel)), random_state=0)
    qs = (("p50", 0.50), ("p90", 0.90), ("p99", 0.99), ("p999", 0.999))
    bands = {}
    for haz in HAZARDS:
        s = pooled[haz](samp)
        bands[haz] = {q: float(np.quantile(s, v)) for q, v in qs}
        for k in LEADS:
            s = per_lead[(haz, k)](samp)
            bands[f"{haz}_l{k}"] = {q: float(np.quantile(s, v)) for q, v in qs}

    # --- metrics the dashboard reports -------------------------------------
    def deviation(haz):
        """The run's threshold-level scoring, keyed by metric name.

        Average precision ranks and says nothing about what a user at a
        threshold experiences, so the model card quotes Brier skill and
        precision from here. Read out of the run's own CSV because a figure
        typed into the page is a figure that will not be updated with the
        next run.
        """
        p = v2_models.run_dir() / "results" / f"deviation_{haz}_run.csv"
        d = pd.read_csv(p, index_col=0)
        return d["xgboost"].to_dict()

    def ablation():
        """Forward-rain ablation, if `audit_fwd_ablation.py` has been run.

        Optional because it refits two models and is not part of a training
        run. When absent the model card drops the claim rather than carrying a
        remembered number -- the same rule the transformer results follow.

        Run-id checked on the same grounds as `uncertainty`: this file used to
        be a bare list with no record of what it was measured against, so an
        ablation from an older feature set would have been printed beside this
        run's headline as though the two shared a test set.
        """
        p = ARTIFACTS / "audit_fwd_ablation.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        here = v2_models.provenance()["run_id"]
        if not isinstance(d, dict) or d.get("run_id") != here:
            print(f"  skipping audit_fwd_ablation.json: computed against "
                  f"{d.get('run_id') if isinstance(d, dict) else 'an unstamped run'}"
                  f", this export is {here}")
            return None
        rows = d["variants"]
        out = {}
        for haz in HAZARDS:
            full = next((r for r in rows if r["hazard"] == haz
                         and "NO" not in r["variant"]), None)
            less = next((r for r in rows if r["hazard"] == haz
                         and "NO" in r["variant"]), None)
            if not (full and less):
                return None
            out[haz] = {"ap_all": full["test_ap"], "ap_no_fwd": less["test_ap"],
                        "share_lost": 1 - less["test_ap"] / full["test_ap"]}
        return out

    def uncertainty():
        """Episode-block CIs and seed spread, if `audit_uncertainty.py` has run.

        Optional on the same terms as `ablation`. The run_id check matters more
        here than elsewhere: an interval computed against one run's boosters
        printed beside another run's point estimate would look like a tight
        result when it is really two unrelated numbers.
        """
        p = ARTIFACTS / "audit_uncertainty.json"
        if not p.exists():
            return None
        u = json.loads(p.read_text())
        here = v2_models.provenance()["run_id"]
        if u.get("run_id") != here:
            print(f"  skipping audit_uncertainty.json: computed against "
                  f"{u.get('run_id')}, this export is {here}")
            return None
        for hz in u["hazards"].values():
            # Audits written before the field existed stored only the
            # bootstrap mean of the ratio, which sits away from the point
            # estimate. Derive the published ratio from the two APs rather
            # than quoting a resampled mean as the result.
            if hz.get("vs_baseline_point") is None and hz["baseline_ap"]:
                hz["vs_baseline_point"] = hz["ap_rescored"] / hz["baseline_ap"]
        return u

    def patience():
        """Early-stopping sensitivity, if `audit_patience.py` has run.

        Carries the span of the whole sweep alongside the per-level means. The
        span is the reportable quantity: it says how far the test figure moves
        under a hyperparameter the validation set cannot pin down, which is a
        different and larger uncertainty than the seed spread at one level.
        """
        p = ARTIFACTS / "audit_patience.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        here = v2_models.provenance()["run_id"]
        if d.get("run_id") != here:
            print(f"  skipping audit_patience.json: computed against "
                  f"{d.get('run_id')}, this export is {here}")
            return None
        # The setting the published booster actually used, read from the run
        # that produced it. Hardcoding it here would let the report keep
        # naming 50 after a retrain moved it.
        man = json.loads((v2_models.RUN_DIR / "manifest.json").read_text())
        for hz in d["hazards"].values():
            if not hz:
                continue
            means = [r["mean"] for r in hz.values()]
            every = [a for r in hz.values() for a in r["ap"]]
            iters = [i for r in hz.values() for i in r["best_iteration"]]
            hz["_span"] = {
                "n_fits": len(every),
                "mean_lo": min(means), "mean_hi": max(means),
                "ap_lo": min(every), "ap_hi": max(every),
                "iter_lo": min(iters), "iter_hi": max(iters),
                "published": man["params"]["xgb"]["early"],
                "best_level": max(hz, key=lambda k: hz[k]["mean"])}
        return d

    def forecast_rain():
        """Observed-vs-forecast forward rain, if `audit_forecast_rain.py` ran.

        Optional and run_id-guarded like the audits above. Both arms are
        loaded when both exist, because they cover different spans and
        different forecast products: quoting whichever happened to run last
        would make the window the result depends on invisible.
        """
        out = {}
        for arm in ("gefs", "open-meteo"):
            p = ARTIFACTS / f"audit_forecast_rain_{arm}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            here = v2_models.provenance()["run_id"]
            if d.get("run_id") != here:
                print(f"  skipping {p.name}: computed against "
                      f"{d.get('run_id')}, this export is {here}")
                continue
            for hz in d["hazards"].values():
                # Which input won, stated once here rather than re-derived
                # from the sign of a difference at each point of use. For
                # hurricanes the forecast is the better input, so "retained"
                # and "drop" do not describe this hazard and the card must not
                # render them as though they did.
                hz["forecast_is_better"] = hz["ap_forecast"] > hz["ap_observed"]
            out[arm] = d
        return out or None

    def tft_block(haz):
        if not has_tft:
            return None
        t = tft_res[haz]
        return {"test_ap": t["test"]["ap"],
                "lift_over_prevalence": t["test"]["lift"],
                "vs_best_baseline": t["vs_best_baseline"],
                "epochs_run": t["epochs_run"],
                # Carried under the key the run recorded, not normalised to one
                # name. A run selected on cross-entropy and a run selected on
                # validation AP are not interchangeable, and the key is the only
                # thing in this block that says which rule chose the epoch.
                **{k: t[k] for k in ("best_val_ap", "best_val_loss")
                   if k in t},
                "n_test_windows": t["n_test_windows"],
                "calibration": t["calibration"]["method"]}

    def model_block(haz):
        x = xgb_res[haz]
        return {
            "model": "xgboost_monotone",
            "n_features": len(FEATS),
            "test_ap": x["test"]["ap"],
            "test_prevalence": x["test"]["prevalence"],
            "lift_over_prevalence": x["test"]["lift"],
            "val_ap": x["val"]["ap"],
            "best_baseline": x["best_baseline"],
            "best_baseline_ap": x["best_baseline_ap"],
            "vs_best_baseline": x["vs_best_baseline"],
            "n_train_pos": x["n_train_pos"],
            "n_test_pos": x["n_test_pos"],
            "best_iteration": x["best_iteration"],
            "top_features": x["top_features"][:8],
            "baselines": dict(sorted(x["baselines"].items(),
                                     key=lambda kv: -kv[1])[:6]),
            "deviation": deviation(haz),
            "tft": tft_block(haz),
        }

    def lead_row(haz, k):
        x, key = lead_res[f"{haz}_l{k}"], f"{haz}_l{k}"
        row = {"hazard": haz, "lead": k,
               "prevalence": x["test"]["prevalence"],
               "baseline": x["best_baseline"],
               "baseline_ap": x["best_baseline_ap"],
               "xgb_ap": x["test"]["ap"],
               "xgb_vs_base": x["vs_best_baseline"],
               "xgb_lift": x["test"]["lift"],
               "n_test_pos": x["n_test_pos"],
               "tft_ap": None, "tft_vs_base": None, "tft_lift": None}
        if has_tft:
            t = tft_lead_res[key]
            row |= {"tft_ap": t["test"]["ap"],
                    "tft_vs_base": t["vs_best_baseline"],
                    "tft_lift": t["test"]["lift"]}
        return row

    leads_block = [lead_row(haz, k) for haz in HAZARDS for k in LEADS]

    meta = county_meta(panel, geo)
    payload = {
        "version": "v2",
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observed_through": str(panel["DATE"].max().date()),
        "run": v2_models.provenance(),
        "bands": bands,
        "horizon_days": prm["horizon_days"],
        "leads": list(LEADS),
        "splits": prm["years"],
        "label_end": prm.get("label_end", {}),
        "target": ("a NOAA-recorded flood episode or HURDAT2 cyclone pass "
                   "begins in this county within the next 3 days"),
        "label_source": {
            "flood": "NOAA StormEvents flood episodes, 2000-2026",
            "storm": "HURDAT2 Atlantic best track, county passes through 2024",
        },
        "n_counties_modeled": sum(v[4] for v in meta.values()),
        "n_counties_mapped": len(meta),
        "n_counties_scored": live_n_counties(),
        "n_events_listed": int(len(shown)),
        "n_events_total": int(len(events)),
        "coverage": coverage,
        "episodes": episodes,
        "fwd_rain_ablation": ablation(),
        "uncertainty": uncertainty(),
        "patience_sweep": patience(),
        "forecast_rain": forecast_rain(),
        "architectures": ["xgboost_monotone"] + (
            ["temporal_fusion_transformer"] if has_tft else []),
        "tft_status": {
            "available": has_tft,
            "note": None if has_tft else (
                "Withdrawn. The only trained transformer used the superseded "
                "label window, so comparing it against these boosters would "
                "compare two label definitions rather than two architectures. "
                "Pending a GPU rerun on this panel."),
        },
        "flood": model_block("flood"),
        "storm": model_block("storm"),
        "lead_table": leads_block,
        "counties": meta,
    }

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "meta.json").write_text(json.dumps(payload, separators=(",", ":")))
    (DOCS / "events.json").write_text(json.dumps(ev_out, separators=(",", ":")))
    (DOCS / "history.json").write_text(
        json.dumps({"days": by_day}, separators=(",", ":")))

    for f in ("meta.json", "events.json", "history.json"):
        print(f"  {f}: {(DOCS / f).stat().st_size / 1024:.0f} KB")
    print(f"days covered: {len(by_day):,}   "
          f"counties modelled: {payload['n_counties_modeled']}")


if __name__ == "__main__":
    main()

"""Fetch live weather per county and score today + next 3 days.

Open-Meteo's forecast endpoint returns up to 92 past days alongside the
forecast, so a single request per county supplies both the antecedent windows
and the forward-looking rain the models need. No API key required.

Writes docs/data/live.json.
"""

import json
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import cyclone_features
from config import OUT_DIR
from v2_models import FEATS, LEADS, load_lead_scorers, load_scorers, provenance

API = "https://api.open-meteo.com/v1/forecast"
PAST = 92
FWD = 4

# Each scored day needs rain for the three days after it, so the last scored
# day needs forecast out to FWD + HORIZON. Requesting only FWD truncates the
# forward window of every day but the first: with 1 mm falling daily, the four
# days returned prcp_fwd3 of 3, 2, 1 and 0 mm instead of 3, 3, 3, 3. The
# feature is one of the model's strongest, so the error pushed later leads
# toward "calm" -- the one direction a hazard dashboard must not fail in.
HORIZON = 3
REQ_FWD = FWD + HORIZON
STATIC = ["lat", "lon", "n_stations", "AREA_SQMI", "E_TOTPOP", "pop_density",
          "p95_prev3", "p95_fwd3", "mean_prev30"]
REFDATA = pathlib.Path(__file__).parent / "refdata"
STATIC_CSV = REFDATA / "county_static.csv"
NATIONAL_CSV = REFDATA / "county_static_national.csv"
# AL, FL, LA, MS, TX. The dashboard was briefly national; the project was
# scoped back to the Gulf, so only these states are fetched and scored.
GULF_FIPS = ("01", "12", "22", "28", "48")

# Six of the model features come from the static table, directly or as the
# denominator of a ratio. A county missing any of them cannot be scored:
# XGBoost would still return a number, because it routes NaN down a learned
# default branch, and that number would be indistinguishable on the map from a
# real one. Skipping is the only honest option.
#
# The three CDC SVI themes used to be required here and left with the model. No
# county gains coverage from that: the 8 the national table still drops are
# missing rainfall climatology, not social vulnerability. The columns stay in
# the cached tables, unread.
REQUIRED = ["lat", "lon", "pop_density", "p95_prev3", "p95_fwd3", "mean_prev30"]

# Open-Meteo's free tier allows 600 calls/minute. Going national takes this from
# 356 requests to ~3,100, and at eight unthrottled workers the archive endpoint
# started returning 429s partway through, losing whole blocks of adjacent
# counties. Half the published ceiling costs about ten minutes and holds.
WORKERS = 6
MIN_INTERVAL = 0.2

# A throttled run returning a fraction of the counties would overwrite a good
# file and read on the map as the model losing coverage overnight, rather than
# as a failed fetch. Below this share of the attempted counties, keep the old
# file and exit non-zero so the scheduled run reports failure.
MIN_SUCCESS = 0.9

_gate = threading.Lock()
_next_at = [0.0]


def _throttle():
    """Space request starts across all workers, so the cap is a rate not a burst."""
    with _gate:
        wait = _next_at[0] - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _next_at[0] = max(time.monotonic(), _next_at[0]) + MIN_INTERVAL


def county_static(rebuild=False):
    """Per-county constants the live features need.

    These do not change between runs, so they are cached to a small CSV. The
    daily refresh runs from a clean checkout where the 60 MB panel is not
    committed, and this table was the only thing it needed out of it.

    `p95_prev3`, `p95_fwd3` and `mean_prev30` are per-county rainfall
    climatology read off the panel, and they move when the panel is rebuilt.
    They are the denominators of the ratio features, which carry the most weight
    in the flood model, so a cache from an older panel would quietly score every
    county against the wrong normal. Pass `rebuild=True` after a panel change.
    """
    src = NATIONAL_CSV if NATIONAL_CSV.exists() else STATIC_CSV
    if src.exists() and not rebuild:
        cs = pd.read_csv(src, dtype={"county_fips": str})
        cs["county_fips"] = cs["county_fips"].str.zfill(5)
        cs = cs[cs["county_fips"].str[:2].isin(GULF_FIPS)].reset_index(drop=True)
        full = cs[REQUIRED].notna().all(axis=1)
        if not full.all():
            print(f"{src.name}: {len(cs)} counties, "
                  f"{(~full).sum()} dropped for missing constants")
        return cs[full].reset_index(drop=True)

    panel = pd.read_parquet(
        OUT_DIR / "daily_panel.parquet", columns=["county_fips", "DATE"] + STATIC
    )
    cs = (
        panel.sort_values("DATE").groupby("county_fips")[STATIC].last().reset_index()
    )
    STATIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    cs.to_csv(STATIC_CSV, index=False)
    print(f"cached {len(cs)} county constants to {STATIC_CSV}")
    return cs


GAUGE_RATIO_JSON = REFDATA / "gauge_ratio.json"


def gauge_ratios(rebuild=False):
    """Median wet-day prcp_max_stn/prcp per county, read off the panel.

    Training's prcp_max_stn is the max across a county's gauges, but the live
    fetch has one grid value per county, and serving that value as both prcp
    and prcp_max_stn understates the feature everywhere the panel had more
    than one gauge -- which is the dense-gauge metro counties, exactly where
    hurricane risk concentrates. The understatement only pushes live scores
    down. Multiplying the grid value by the county's own training-time ratio
    restores the feature's scale. Counties outside the panel stay at 1.0,
    which is exact for the single-gauge majority (245 of 356 panel counties).
    Cached for the same reason as county_static: the daily refresh runs from
    a checkout without the panel. Rebuild after a panel change.
    """
    if GAUGE_RATIO_JSON.exists() and not rebuild:
        return json.loads(GAUGE_RATIO_JSON.read_text())
    panel = pd.read_parquet(OUT_DIR / "daily_panel.parquet",
                            columns=["county_fips", "prcp", "prcp_max_stn"])
    wet = panel[panel["prcp"] >= 1.0]
    r = (wet["prcp_max_stn"] / wet["prcp"]).groupby(wet["county_fips"]).median()
    out = {str(f): round(float(v), 4) for f, v in r.items() if np.isfinite(v)}
    GAUGE_RATIO_JSON.write_text(json.dumps(out, indent=0, sort_keys=True))
    print(f"cached {len(out)} gauge ratios to {GAUGE_RATIO_JSON}")
    return out


def fetch(row):
    """One county's rainfall history and forecast, retried on a soft failure.

    Open-Meteo throttles a burst of a few hundred requests, so roughly one
    county in twelve fails on the first attempt. Without the retry those
    counties drop off the map for the whole day with no visible reason, which
    reads as missing coverage rather than as a rate limit.
    """
    err = None
    for attempt in range(4):
        if attempt:
            time.sleep(1.5 * 2 ** (attempt - 1))
        _throttle()
        try:
            r = requests.get(
                API,
                params={
                    "latitude": round(row.lat, 4),
                    "longitude": round(row.lon, 4),
                    "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,"
                             "wind_speed_10m_max,wind_gusts_10m_max",
                    "timezone": "America/Chicago",
                    "past_days": PAST,
                    "forecast_days": REQ_FWD,
                },
                timeout=30,
            )
            # A 429 means the whole pool is over the cap, so backing off this one
            # thread is not enough; pause every worker by pushing the shared gate.
            if r.status_code == 429:
                err = "429 throttled"
                with _gate:
                    _next_at[0] = max(_next_at[0], time.monotonic() + 20)
                continue
            r.raise_for_status()
            return row.county_fips, r.json()["daily"], None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return row.county_fips, None, err


def build_rows(fips, daily, static, feats):
    """One feature row per forecast day for this county."""
    t = pd.Series(pd.to_datetime(daily["time"]))
    # Kept unfilled so a gap in the forward window is detectable. Zero-filling
    # it first would turn "the API returned nothing for Friday" into "no rain
    # fell on Friday", which is the same number with the opposite meaning.
    prcp_raw = pd.Series(daily["precipitation_sum"], dtype="float64")
    prcp = prcp_raw.fillna(0.0)
    tmax = pd.Series(daily["temperature_2m_max"], dtype="float64")
    tmin = pd.Series(daily["temperature_2m_min"], dtype="float64")

    # Index of "today" = first forecast day.
    i0 = PAST
    out = []
    for k in range(FWD):
        i = i0 + k
        # A day whose forward window runs off the end of the response is
        # dropped, not scored on a short sum: the model has no way to tell a
        # genuinely dry three days from three days it was never given.
        if i + HORIZON >= len(t):
            break
        past = prcp.iloc[:i + 1]
        fwd = prcp_raw.iloc[i + 1:i + 1 + HORIZON]
        if fwd.isna().any():
            break

        f = {
            "prcp": float(prcp.iloc[i]),
            "prcp_max_stn": float(prcp.iloc[i]) * float(static["gauge_ratio"]),
            "tmax": float(tmax.iloc[i]) if pd.notna(tmax.iloc[i]) else np.nan,
            "tmin": float(tmin.iloc[i]) if pd.notna(tmin.iloc[i]) else np.nan,
        }
        for w in (3, 7, 14, 30, 60, 90):
            f[f"prcp_prev{w}"] = float(past.iloc[-w:].sum())
        f["prcp_fwd1"] = float(fwd.iloc[0])
        f["prcp_fwd3"] = float(fwd.sum())
        f["prcp_fwd3_max"] = float(fwd.max())

        day = t.iloc[i]
        f["month_sin"] = float(np.sin(2 * np.pi * day.month / 12))
        f["month_cos"] = float(np.cos(2 * np.pi * day.month / 12))
        f["hurricane_season"] = int(6 <= day.month <= 11)

        for c in STATIC:
            f[c] = float(static[c]) if pd.notna(static[c]) else np.nan

        p95_3 = f["p95_prev3"] or np.nan
        p95_f = f["p95_fwd3"] or np.nan
        m30 = f["mean_prev30"] or np.nan
        f["ratio_prev3"] = f["prcp_prev3"] / p95_3 if p95_3 else np.nan
        f["ratio_fwd3"] = f["prcp_fwd3"] / p95_f if p95_f else np.nan
        f["ratio_prev30"] = f["prcp_prev30"] / m30 if m30 else np.nan
        f["log_fwd3"] = float(np.log1p(max(f["prcp_fwd3"], 0)))

        out.append((day.strftime("%Y-%m-%d"), f,
                    {"precip": round(float(prcp.iloc[i]), 1),
                     "tmax": None if pd.isna(tmax.iloc[i]) else round(float(tmax.iloc[i]), 1),
                     "tmin": None if pd.isna(tmin.iloc[i]) else round(float(tmin.iloc[i]), 1),
                     "wind": round(float(daily["wind_speed_10m_max"][i] or 0), 1),
                     "gust": round(float(daily["wind_gusts_10m_max"][i] or 0), 1),
                     "fwd3": round(f["prcp_fwd3"], 1)}))
    return out


def main(rebuild_static=False):
    scorers = load_scorers()
    per_lead = load_lead_scorers()
    cs = county_static(rebuild=rebuild_static)
    ratios = gauge_ratios(rebuild=rebuild_static)
    cs["gauge_ratio"] = cs["county_fips"].map(ratios).fillna(1.0)
    print(f"fetching live weather for {len(cs)} counties "
          f"({(cs['gauge_ratio'] > 1).sum()} with a gauge-ratio correction) ...",
          flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(fetch, list(cs.itertuples(index=False))))

    ok = {f: d for f, d, _ in results if d}
    print(f"  {len(ok)}/{len(cs)} succeeded in {(time.time()-t0)/60:.1f} min")
    for f, d, err in results:
        if not d:
            print(f"  gave up on {f}: {err}")
    if len(ok) < MIN_SUCCESS * len(cs):
        raise SystemExit(
            f"only {len(ok)}/{len(cs)} counties fetched "
            f"({len(ok)/len(cs):.0%} < {MIN_SUCCESS:.0%}); leaving live.json alone"
        )

    static_map = cs.set_index("county_fips")
    rows, keys = [], []
    weather = {}
    for fips, daily in ok.items():
        for day, f, w in build_rows(fips, daily, static_map.loc[fips], FEATS):
            rows.append(f)
            keys.append((fips, day))
            weather.setdefault(fips, {})[day] = w

    X = pd.DataFrame(rows)
    X.insert(0, "day", [d for _, d in keys])
    X.insert(0, "county_fips", [f for f, _ in keys])

    # The same county-days, from the official forecast that is on the wire right
    # now rather than from the archive the panel was built against.
    cyc = cyclone_features.live(cs[["county_fips", "lat", "lon"]],
                                sorted({d for _, d in keys}))
    cyc["day"] = cyc.pop("DATE").dt.strftime("%Y-%m-%d")
    n = len(X)
    X = X.merge(cyc, on=["county_fips", "day"], how="left")
    if len(X) != n:
        raise SystemExit(
            f"cyclone merge changed the row count, {n} -> {len(X)}: a duplicated "
            f"county-day would misalign every probability written below")
    # A null cyc_active means a storm was under advisory but its forecast did
    # not reach this date, so the cyclone block is unknown rather than quiet.
    # Dropped on the same principle as a day with a short rainfall window: the
    # booster never saw a null here and would route it down an arbitrary default
    # branch, returning a number indistinguishable on the map from a real one.
    unknown = X["cyc_active"].isna()
    if unknown.any():
        print(f"  cyclone: dropped {int(unknown.sum())} county-days beyond the "
              f"forecast horizon ({sorted(X.loc[unknown, 'day'].unique())})")
        X = X[~unknown].reset_index(drop=True)
    act = int(X["cyc_active"].sum())
    print(f"  cyclone: {act}/{len(X)} county-days with an active storm"
          + (f", closest {X['cyc_dist_min'].min():.0f} km" if act else ""))

    missing = [c for c in FEATS if c not in X.columns]
    if missing:
        raise SystemExit(f"live features missing {missing}")
    for haz, fn in scorers.items():
        X[f"p_{haz}"] = fn(X)
    for (haz, k), fn in per_lead.items():
        X[f"p_{haz}_l{k}"] = fn(X)

    # Same 13-wide row shape the history export uses, so the dashboard has one
    # row reader. The two ground-truth flags sit at 0: on a forecast day nothing
    # has happened yet, and claiming otherwise would read as a model output.
    cols = (["p_flood", "p_storm"]
            + [f"p_flood_l{k}" for k in LEADS] + [f"p_storm_l{k}" for k in LEADS])
    days = {}
    for fips, day, *vals in X[["county_fips", "day"] + cols].itertuples(
            index=False, name=None):
        pf, ps, *leads = (round(float(v), 5) for v in vals)
        days.setdefault(day, []).append(
            [fips, pf, ps, weather[fips][day]["fwd3"], None, 0, 0] + leads
        )

    payload = {
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Open-Meteo forecast, scored by the XGBoost models from run "
                  + provenance()["run_id"],
        "run_id": provenance()["run_id"],
        "n_counties": len(ok),
        "days": days,
    }
    p = pathlib.Path("docs/data/live.json")
    p.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)  days={sorted(days)}")


if __name__ == "__main__":
    import sys
    main(rebuild_static="--rebuild-static" in sys.argv)

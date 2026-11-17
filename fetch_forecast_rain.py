"""Archived forecast rain for the forward window, per county and lead.

The panel's prcp_fwd* features are built from the same GHCN record as the
labels, so they are the rain that caused the flood rather than the rain a
forecaster could have seen. Every skill number resting on them is an upper
bound that no deployed system can reach. This script fetches what was actually
forecast, so the bound can be measured.

Open-Meteo's previous-runs archive is the only source here that resolves lead
time: `precipitation_previous_dayK` at date D is the forecast for D issued K
days earlier. So the forward window at origin t is assembled from
previous_day1 at t+1, previous_day2 at t+2 and previous_day3 at t+3 -- three
values that all come from the model run on day t, which is exactly the
information a forecast issued that morning contained.

The archive starts 2024-01-25. That is the binding constraint on this whole
exercise: it leaves the flood label 19 test episodes to be scored against,
and the hurricane label only 6, which is too few to say anything. The
historical-forecast API reaches back further but does not resolve lead, so
using it for a three-day window would quietly credit the model with a shorter
lead than it claims -- the same error this script exists to remove.

Writes artifacts/forecast_rain.parquet, one row per county-day, resumable: a
second invocation fetches only the counties the first one did not finish.
"""
import pathlib
import sys
import time

import pandas as pd
import requests

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
# The first date the lead-resolved archive answers for. Established by probing
# the API, not documented; re-probe before extending the range backwards.
START = "2024-01-25"
LEADS = (1, 2, 3)
VARS = ",".join(f"precipitation_previous_day{k}" for k in LEADS)
# The panel's daily totals come from GHCN, which bins on local time, and the
# live path already requests Central. Asking for UTC here would shift every
# daily sum by six hours against the labels.
TZ = "America/Chicago"
DEST = "artifacts/forecast_rain.parquet"
# Small enough that one response stays a few MB and a failure costs little.
BATCH = 8
PANEL = "artifacts/daily_panel.parquet"


def county_points():
    p = pd.read_parquet(PANEL, columns=["county_fips", "DATE", "lat", "lon"])
    end = p["DATE"].max().date().isoformat()
    pts = (p.groupby("county_fips")[["lat", "lon"]].first()
           .round(4).reset_index())
    return pts, end


class Throttled(Exception):
    """The shared pool is over its cap, not a problem with this batch."""


def get(lats, lons, start, end):
    """One request for several counties, retried on throttling.

    A 429 here is the shared pool being over its cap rather than anything
    about this batch, so the backoff is long enough to be worth waiting out.
    """
    last = "throttled on every attempt"
    for attempt in range(5):
        if attempt:
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
        try:
            r = requests.get(API, timeout=180, params={
                "latitude": ",".join(lats), "longitude": ",".join(lons),
                "hourly": VARS, "start_date": start, "end_date": end,
                "timezone": TZ})
            if r.status_code == 429:
                continue
            r.raise_for_status()
            d = r.json()
            # A single-location request returns an object, not a list. Batches
            # here are always >1 in practice, but a trailing batch of one would
            # otherwise iterate over dict keys.
            return d if isinstance(d, list) else [d]
        except Exception as e:
            last = e
    raise Throttled(f"{lats[0]}..{lats[-1]}: {last}")


def main():
    pts, end = county_points()
    done = set()
    if pathlib.Path(DEST).exists():
        prev = pd.read_parquet(DEST)
        done = set(prev["county_fips"].unique())
        print(f"{DEST} already holds {len(done)} counties")
    todo = pts[~pts["county_fips"].isin(done)].reset_index(drop=True)
    if todo.empty:
        print("nothing to fetch")
        return
    # Year chunks keep each response a few MB rather than tens, which matters
    # because the whole batch is re-fetched when one request fails.
    spans = [(f"{y}-01-01", f"{y}-12-31") for y in
             range(int(START[:4]), int(end[:4]) + 1)]
    spans[0] = (START, spans[0][1])
    spans[-1] = (spans[-1][0], end)

    out = []

    def flush():
        """Checkpoint what has been fetched so far.

        Called as the fetch proceeds rather than once at the end: the whole
        run is an hour of requests against a throttled public API, and a
        failure in the last batch must not cost the first forty.
        """
        if not out:
            return
        got = pd.concat(([prev] if done else []) + out, ignore_index=True)
        got = got.sort_values(["county_fips", "DATE"]).reset_index(drop=True)
        got.to_parquet(DEST, index=False)
        return got

    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        b = todo.iloc[i:i + BATCH]
        lats = [f"{v:.4f}" for v in b["lat"]]
        lons = [f"{v:.4f}" for v in b["lon"]]
        per_county = {f: [] for f in b["county_fips"]}
        try:
            for s, e in spans:
                for fips, ent in zip(b["county_fips"], get(lats, lons, s, e)):
                    h = ent["hourly"]
                    df = pd.DataFrame({k: h[k] for k in h if k != "time"})
                    df["DATE"] = pd.to_datetime(h["time"]).floor("D")
                    # min_count so a day the archive has no run for stays null
                    # instead of summing to a confident zero.
                    per_county[fips].append(df.groupby("DATE").sum(min_count=1))
        except Throttled as exc:
            # The cap is hourly and the whole archive is a few hundred
            # counties, so running out of quota is the expected way this
            # finishes rather than a failure. Keep what completed and say how
            # to continue; a non-zero exit here would make a resumable fetch
            # look broken.
            got = flush()
            have = got["county_fips"].nunique() if got is not None else len(done)
            print(f"\nthrottled at {exc}")
            print(f"kept {have}/{len(pts)} counties in {DEST}; "
                  f"re-run after the hourly cap resets to continue")
            return
        for fips, parts in per_county.items():
            d = pd.concat(parts).reset_index()
            d.insert(0, "county_fips", fips)
            out.append(d)
        el = time.time() - t0
        n = i + len(b)
        print(f"  {n:>4}/{len(todo)} counties  {el:>5.0f}s  "
              f"eta {el / n * (len(todo) - n):>5.0f}s", flush=True)
        if n % (BATCH * 5) == 0:
            flush()

    got = flush()
    print(f"\nwrote {DEST}  {len(got):,} county-days  "
          f"{got['county_fips'].nunique()} counties  "
          f"{got['DATE'].min().date()}..{got['DATE'].max().date()}")


if __name__ == "__main__":
    sys.exit(main())

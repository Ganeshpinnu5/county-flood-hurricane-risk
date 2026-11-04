"""Rainfall climatology for counties outside the Gulf panel, from ERA5.

The 356 Gulf counties get `p95_prev3`, `p95_fwd3` and `mean_prev30` from the
GHCN station panel. Counties outside it have no panel, so the same three
constants are derived here from Open-Meteo's ERA5 archive over the identical
window, CLIM_YEARS in build_daily.py.

The two sources are not interchangeable. ERA5 is a gridded reanalysis and GHCN
is a station record, so a county's "normal" measured one way is not the same
number measured the other. These constants are the denominators of the ratio
features, which carry the most weight in the flood model, so the difference is
a real bias and not a rounding detail. It is measured in audit_climatology.py
and disclosed on the dashboard rather than hidden.

Resumable: every county is appended to the output CSV as it lands, and a rerun
skips what is already there. A throttle or a dropped connection costs the
counties in flight, not the run.
"""
import argparse
import csv
import pathlib
import threading
import time

import numpy as np
import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# Must match CLIM_YEARS in build_daily.py. The extra days past the window give
# the forward-looking rolling sum its last three days instead of truncating it.
CLIM_START = "2000-01-01"
CLIM_END = "2008-12-31"
FETCH_END = "2009-01-03"
CLIM_YEARS = (2000, 2008)

OUT = pathlib.Path(__file__).parent / "refdata" / "county_climatology.csv"
COLS = ["county_fips", "p95_prev3", "p95_fwd3", "mean_prev30", "n_days"]


def climatology(prcp, dates):
    """The three constants, by the recipe in build_daily.py lines 68-97."""
    s = pd.Series(prcp, dtype="float64").fillna(0.0)
    prev3 = s.rolling(3, min_periods=1).sum()
    prev30 = s.rolling(30, min_periods=10).sum()
    fwd3 = s.shift(-3).rolling(3, min_periods=1).sum()

    yr = pd.DatetimeIndex(dates).year
    m = (yr >= CLIM_YEARS[0]) & (yr <= CLIM_YEARS[1])
    return {
        "p95_prev3": float(prev3[m].quantile(0.95)),
        "p95_fwd3": float(fwd3[m].quantile(0.95)),
        "mean_prev30": float(prev30[m].mean()),
        "n_days": int(m.sum()),
    }


def fetch_one(row, session, tries=5):
    err = None
    for attempt in range(tries):
        if attempt:
            time.sleep(min(60, 2.0 * 2 ** (attempt - 1)))
        try:
            r = session.get(
                ARCHIVE,
                params={
                    "latitude": round(row.lat, 4),
                    "longitude": round(row.lon, 4),
                    "start_date": CLIM_START,
                    "end_date": FETCH_END,
                    "daily": "precipitation_sum",
                    "timezone": "America/Chicago",
                },
                timeout=120,
            )
            if r.status_code == 429:
                err = "429 throttled"
                time.sleep(30)
                continue
            r.raise_for_status()
            d = r.json()["daily"]
            out = climatology(d["precipitation_sum"], d["time"])
            out["county_fips"] = row.county_fips
            return out, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    return None, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="/tmp/clim_targets.csv",
                    help="CSV with county_fips, lat, lon")
    ap.add_argument("--out", default=str(OUT),
                    help="where to append results; the audit points this at a "
                         "scratch file so an overlap sample cannot contaminate "
                         "the real table")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_path = pathlib.Path(args.out)

    tgt = pd.read_csv(args.targets, dtype={"county_fips": str})
    tgt["county_fips"] = tgt["county_fips"].str.zfill(5)

    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path, dtype={"county_fips": str})
        done = set(prev["county_fips"].str.zfill(5))
    todo = tgt[~tgt["county_fips"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(tgt)} targets, {len(done)} already done, {len(todo)} to fetch",
          flush=True)
    if todo.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "t0": time.time()}
    local = threading.local()

    def worker(row):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        rec, err = fetch_one(row, local.s)
        with lock:
            if rec:
                w.writerow({k: rec[k] for k in COLS})
                fh.flush()
                state["ok"] += 1
            else:
                state["fail"] += 1
                print(f"  FAIL {row.county_fips}: {err}", flush=True)
            n = state["ok"] + state["fail"]
            if n % 50 == 0:
                el = time.time() - state["t0"]
                rate = n / el
                left = (len(todo) - n) / rate if rate else 0
                print(f"{n}/{len(todo)}  ok={state['ok']} fail={state['fail']}  "
                      f"{rate:.1f}/s  eta {left/60:.0f} min", flush=True)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(worker, todo.itertuples(index=False)))

    fh.close()
    el = time.time() - state["t0"]
    print(f"done in {el/60:.1f} min: {state['ok']} ok, {state['fail']} failed",
          flush=True)


if __name__ == "__main__":
    main()

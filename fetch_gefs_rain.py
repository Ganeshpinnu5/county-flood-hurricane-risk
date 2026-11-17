"""Archived GEFS forecast rain for the forward window, per county and lead.

Companion to fetch_forecast_rain.py, which does the same job from Open-Meteo's
previous-runs archive. That archive only reaches back to 2024-01-25, which
leaves the hurricane label 6 test episodes -- too few to measure anything. This
reads the operational GEFS output NOAA keeps on S3 instead, which reaches back
far enough to recover 19 of the 23 storm episodes and 60 of the 65 flood ones.

Both scripts emit the same three columns, so audit_forecast_rain.py scores
either one without changing. Two independent forecast products answering the
same question is the point: if the published skill survives one archive and not
the other, that is a fact about the archive rather than about the model.

Product choice, and what it costs
---------------------------------
`geavg` from the pgrb2ap5 set: the 0.5-degree ensemble mean of the 00Z run.
Three constraints pin this down.

- The ensemble mean, not the control member, because precipitation is the field
  the spread exists for and the mean is the calibrated summary of it.
- 2020-09-23 onward, because that is when GEFS v12 began and the bucket holds
  no ensemble mean before it -- only the control and the raw members, on a
  coarser grid. Averaging 21 members by hand to reach back nine further months
  would buy 4 storm episodes and cost a change of grid resolution halfway
  through the series, which is the kind of silent inhomogeneity that makes a
  comparison stop meaning anything.
- The 00Z run, so every value in the forward window was on the wire before the
  local day it forecasts had started.

Lead assembly
-------------
APCP arrives as non-overlapping 6-hour buckets, so local day t+k is the four
buckets ending at forecast hours 24k+12 through 24k+30. All twelve come from
the single run issued at 00Z on day t, which is exactly the information a
forecast published that morning contained -- the same property that made
Open-Meteo's lead-resolved archive usable and its historical-forecast API not.

Day boundaries are taken at 06Z, which is midnight Central Standard. GHCN bins
its daily totals on local time and the live path already requests Central, so
this matches the labels. It is an hour off during daylight saving, and no
product here can fix that: the shortest GEFS accumulation is 3-hourly, and
05Z is not a multiple of 3. Open-Meteo's hourly data does bin exactly, which
is one more reason to keep both arms rather than merge them.

Writes artifacts/gefs/<YYYY-MM>.parquet as it goes and merges them into
artifacts/gefs_forecast_rain.parquet at the end. Resumable: a second
invocation re-reads finished months off disk and fetches only what is missing.
"""
import concurrent.futures as cf
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import requests

import eccodes

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
KEY = "gefs.{d}/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f{h:03d}"
# First date the 0.5-degree ensemble mean exists in the bucket. Established by
# probing, not documented; re-probe before extending the range backwards.
START = os.environ.get("CRISK_GEFS_START", "2020-09-23")
END = os.environ.get("CRISK_GEFS_END", "2024-12-31")
LEADS = (1, 2, 3)
# Local day t+k is 06Z..06Z, so the four 6-hour buckets ending here.
HOURS = {k: [24 * k + 6 * j for j in (2, 3, 4, 5)] for k in LEADS}
ALL_HOURS = sorted({h for v in HOURS.values() for h in v})
WORKERS = int(os.environ.get("CRISK_GEFS_WORKERS", 16))
SHARDS = pathlib.Path(os.environ.get("CRISK_GEFS_SHARDS", "artifacts/gefs"))
DEST = pathlib.Path(os.environ.get("CRISK_GEFS_PARQUET",
                                  "artifacts/gefs_forecast_rain.parquet"))
PANEL = "artifacts/daily_panel.parquet"

SESSION = requests.Session()


def county_points():
    p = pd.read_parquet(PANEL, columns=["county_fips", "lat", "lon"])
    return p.groupby("county_fips")[["lat", "lon"]].first().reset_index()


def _get(url, headers=None):
    """GET with retries. A missing key is a real answer, not a failure.

    Model outages leave holes in the archive, and a hole has to come back as
    None so the day lands as null rather than as a confident zero.
    """
    for attempt in range(4):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            r = SESSION.get(url, headers=headers, timeout=120)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 3:
                return None
    return None


def grib_bytes(date, hour):
    """The one APCP record out of a file holding dozens of fields.

    The .idx sidecar gives each record's byte offset, so the record is pulled
    with a range request. Downloading whole files instead would move about a
    hundred times as many bytes for the same four numbers.
    """
    url = f"{BASE}/{KEY.format(d=date, h=hour)}"
    idx = _get(url + ".idx")
    if idx is None:
        return None
    lines = [ln for ln in idx.decode().splitlines() if ln.strip()]
    offs = [int(ln.split(":")[1]) for ln in lines]
    for i, ln in enumerate(lines):
        if ln.split(":")[3] == "APCP":
            end = offs[i + 1] - 1 if i + 1 < len(offs) else ""
            return _get(url, headers={"Range": f"bytes={offs[i]}-{end}"})
    return None


def fetch_day(date):
    return date, {h: grib_bytes(date, h) for h in ALL_HOURS}


_GRID = {}


def county_cells(msg, pts):
    """Flat grid index per county, derived from the file's own header.

    Hardcoding the GEFS grid would work until the day it silently changes, so
    the geometry is read off the message and then checked against the
    coordinate arrays the same message carries. If the arithmetic and the file
    disagree the run stops rather than quietly sampling the wrong ocean.
    """
    if "idx" in _GRID:
        return _GRID["idx"]
    h = eccodes.codes_new_from_message(msg)
    try:
        g = {k: eccodes.codes_get(h, k) for k in
             ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
              "longitudeOfFirstGridPointInDegrees",
              "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees")}
        lats = eccodes.codes_get_array(h, "latitudes")
        lons = eccodes.codes_get_array(h, "longitudes")
    finally:
        eccodes.codes_release(h)

    ni, nj = int(g["Ni"]), int(g["Nj"])
    di, dj = g["iDirectionIncrementInDegrees"], g["jDirectionIncrementInDegrees"]
    lat0 = g["latitudeOfFirstGridPointInDegrees"]
    lon0 = g["longitudeOfFirstGridPointInDegrees"]
    # The grid runs north to south, so the j index counts down from lat0.
    j = np.rint((lat0 - pts["lat"].to_numpy()) / dj).astype(int)
    i = np.rint(((pts["lon"].to_numpy() - lon0) % 360) / di).astype(int)
    if not ((0 <= j).all() and (j < nj).all() and (0 <= i).all() and (i < ni).all()):
        raise SystemExit("county fell outside the GEFS grid -- check lat/lon")
    flat = j * ni + i
    off = np.abs(lats[flat] - pts["lat"].to_numpy())
    dlon = np.abs((lons[flat] - pts["lon"].to_numpy() + 180) % 360 - 180)
    worst = max(off.max(), dlon.max())
    if worst > max(di, dj):
        raise SystemExit(f"grid index disagrees with the file by {worst:.3f} deg")
    print(f"grid {ni}x{nj} at {di} deg; worst county offset {worst:.3f} deg")
    _GRID["idx"] = flat
    _GRID["size"] = len(lats)
    return flat


def reduce_day(date, blobs, pts):
    """One run's twelve buckets collapsed to three per-county daily totals.

    Decoding happens on the calling thread on purpose. eccodes handles are not
    safe to create concurrently, so the pool downloads bytes and nothing else.
    """
    vals = {}
    for h, b in blobs.items():
        if b is None:
            continue
        hh = eccodes.codes_new_from_message(b)
        try:
            cells = county_cells(b, pts)
            v = eccodes.codes_get_values(hh)
            # The county index is derived once and reused, so a grid that
            # changed partway through the archive would be sampled at stale
            # offsets. Comparing the point count catches that for the price of
            # a len().
            if len(v) != _GRID["size"]:
                raise SystemExit(
                    f"{date} f{h}: grid is {len(v)} points, expected "
                    f"{_GRID['size']} -- the archive changed grid mid-series")
            vals[h] = v[cells]
        finally:
            eccodes.codes_release(hh)

    origin = pd.Timestamp(date)
    rows = []
    for k in LEADS:
        got = [vals[h] for h in HOURS[k] if h in vals]
        # A day missing one of its four buckets is not a daily total.
        if len(got) < len(HOURS[k]):
            continue
        rows.append(pd.DataFrame({
            "county_fips": pts["county_fips"].to_numpy(),
            "DATE": origin + pd.Timedelta(days=k),
            # Which run produced the value. Kept so a shard can be checked for
            # completeness instead of trusted because the file exists: a run
            # over a partial date range would otherwise leave behind a month
            # shard that a later full run accepts and never refetches.
            "origin": origin,
            "lead": k,
            "mm": np.sum(got, axis=0)}))
    return rows


def run_month(month, dates, pts):
    shard = SHARDS / f"{month}.parquet"
    want = {pd.Timestamp(d) for d in dates}
    if shard.exists():
        got = pd.read_parquet(shard)
        missing = want - set(got["origin"].unique())
        if not missing:
            return got, True
        print(f"  {month}: shard has {len(want) - len(missing)}/{len(want)} "
              f"origin dates, refetching")
    rows, t0 = [], time.time()
    # Chunked rather than one pool over the whole month: the pool would
    # otherwise hold every day's bytes in memory before the first decode.
    for i in range(0, len(dates), WORKERS):
        chunk = dates[i:i + WORKERS]
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            for date, blobs in ex.map(fetch_day, chunk):
                rows.extend(reduce_day(date, blobs, pts))
    if not rows:
        print(f"  {month}: nothing in the archive")
        return None, False
    got = pd.concat(rows, ignore_index=True)
    got.to_parquet(shard, index=False)
    print(f"  {month}: {len(got):,} rows  {time.time() - t0:.0f}s")
    return got, False


def main():
    SHARDS.mkdir(parents=True, exist_ok=True)
    pts = county_points()
    days = pd.date_range(START, END, freq="D")
    print(f"{len(pts)} counties  {len(days)} origin dates  "
          f"{START}..{END}  {len(ALL_HOURS)} records per date")

    for month, grp in days.to_series().groupby(days.strftime("%Y-%m")):
        _, cached = run_month(month, [d.strftime("%Y%m%d") for d in grp], pts)
        if cached:
            print(f"  {month}: cached")

    # Merged from every shard on disk rather than from the months this
    # invocation happened to cover. The archive is cumulative, so a run that
    # only extends the end of the range must not rewrite the parquet with its
    # own slice and silently drop the years already fetched.
    shards = sorted(SHARDS.glob("*.parquet"))
    if not shards:
        raise SystemExit("fetched nothing -- re-probe the bucket layout")
    print(f"merging {len(shards)} monthly shards")
    long = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    wide = (long.pivot_table(index=["county_fips", "DATE"], columns="lead",
                             values="mm", aggfunc="first")
            .rename(columns={k: f"precipitation_previous_day{k}" for k in LEADS})
            .reset_index().sort_values(["county_fips", "DATE"]))
    wide.columns.name = None
    wide.to_parquet(DEST, index=False)
    print(f"\nwrote {DEST}  {len(wide):,} county-days  "
          f"{wide['county_fips'].nunique()} counties  "
          f"{wide['DATE'].min().date()}..{wide['DATE'].max().date()}")


if __name__ == "__main__":
    sys.exit(main())

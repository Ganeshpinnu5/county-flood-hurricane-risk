"""NHC official forecasts as they were issued, from the ATCF a-decks.

Written because the storm model had no cyclone features at all: it saw rainfall,
temperature, month and lat/lon, and was asked whether a hurricane would reach a
county. That is why it early-stopped after a handful of rounds -- there was
nothing in the feature matrix about a storm. This fetches the information a
forecaster actually had, and it is what took hurricane AP from 0.15570 to
0.53294.

a-decks, not b-decks. The b-deck is the post-season best track -- the truth, and
the right thing to build a *label* from, which is what labels.py does. The
a-deck holds every forecast issued during the storm, stamped with the time it
was issued, so a feature built from it contains only what was knowable at the
origin date. Using the best track as a feature would be the forward-rain defect
again in a worse form: perfect foreknowledge of the track.

`OFCL` only, out of the dozens of techs in the file. It is NHC's official
forecast, the one the public cone is drawn from, so a model fed it is being
given the operational product rather than a model output nobody published.
Advisories where OFCL is absent are left missing rather than backfilled from a
guidance model, because "NHC had no forecast for this storm yet" is a real state
and a guidance aid is not what was on the wire.

Coverage, established by probing rather than documentation:

- OFCL track and intensity run from 2000, the start of the panel.
- Forecast *wind radii* begin in 2004. 2000-2003 therefore carry track and
  intensity but null R34/R50/R64, which XGBoost handles natively as missing.
  Those four years are part of the training window only, never the test window.
- 2000-2025 live under atcf/archive/<year>/. The current season is not archived
  until it closes, so 2026 comes from atcf/aid_public/.

Writes artifacts/nhc_adeck.parquet, one row per storm, issue time and forecast
hour, with the three wind-radius tiers pivoted into columns. Resumable: the
compressed decks are cached under artifacts/adeck/ and a second invocation
re-reads them instead of refetching.
"""
import gzip
import io
import os
import pathlib
import re
import sys
import time

import pandas as pd
import requests

ARCHIVE = "https://ftp.nhc.noaa.gov/atcf/archive/{year}/"
LIVE = "https://ftp.nhc.noaa.gov/atcf/aid_public/"
# Atlantic only. Every cyclone that has ever reached the Gulf Coast panel is an
# Atlantic-basin storm; eastern-Pacific systems that cross into Texas arrive as
# remnant moisture, which the rainfall features already carry.
BASIN = "al"
# NHC's official forecast. CARQ is the analysis, the rest are guidance aids.
TECH = "OFCL"
# Invest numbers. They are pre-genesis areas with no official forecast, so they
# drop out on the OFCL filter anyway -- excluded here to avoid fetching 20 files
# a season that contribute nothing.
INVEST = range(90, 100)
START = int(os.environ.get("CRISK_ADECK_START", 2000))
END = int(os.environ.get("CRISK_ADECK_END", 2026))
CACHE = pathlib.Path(os.environ.get("CRISK_ADECK_CACHE", "artifacts/adeck"))
DEST = pathlib.Path(os.environ.get("CRISK_ADECK_PARQUET",
                                   "artifacts/nhc_adeck.parquet"))
TIERS = (34, 50, 64)
QUADS = ("ne", "se", "sw", "nw")

SESSION = requests.Session()


def _get(url, tries=4):
    for attempt in range(tries):
        if attempt:
            time.sleep(2 ** attempt)
        try:
            r = SESSION.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == tries - 1:
                return None
    return None


def deck_names(year):
    """Which storms exist in a season, read off the listing rather than guessed.

    Season storm counts vary and a gap in the numbering is possible, so probing
    1..40 would mean dozens of wasted 404s per year.
    """
    base = ARCHIVE.format(year=year) if year < END else LIVE
    body = _get(base)
    if body is None:
        return base, []
    pat = rf"a{BASIN}(\d{{2}}){year}\.dat\.gz"
    nums = sorted({int(m) for m in re.findall(pat, body.decode(errors="replace"))})
    return base, [n for n in nums if n not in INVEST]


def fetch_deck(base, year, num):
    name = f"a{BASIN}{num:02d}{year}.dat.gz"
    local = CACHE / name
    if local.exists():
        return local.read_bytes()
    body = _get(base + name)
    if body is None:
        return None
    local.write_bytes(body)
    return body


def _tenths(tok):
    """ATCF latitude and longitude: tenths of a degree with a hemisphere suffix."""
    tok = tok.strip()
    if not tok or not tok[:-1].lstrip("-").isdigit():
        return None
    v = int(tok[:-1]) / 10.0
    return -v if tok[-1] in "SW" else v


def parse_deck(body, storm_id):
    """OFCL rows of one deck, as (issue, tau) keyed records.

    A forecast appears once per wind-radius tier, so the same position arrives
    on up to three lines that differ only in the RAD column. They are collapsed
    here rather than downstream, because a half-populated set of tiers is the
    normal case and only this function knows which column distinguished them.
    """
    recs = {}
    for line in gzip.GzipFile(fileobj=io.BytesIO(body)):
        f = [c.strip() for c in line.decode(errors="replace").split(",")]
        if len(f) < 20 or f[4] != TECH:
            continue
        lat, lon = _tenths(f[6]), _tenths(f[7])
        if lat is None or lon is None:
            continue
        key = (f[2], int(f[5]))
        r = recs.setdefault(key, {
            "storm_id": storm_id,
            "issued": pd.to_datetime(f[2], format="%Y%m%d%H"),
            "tau": int(f[5]), "lat": lat, "lon": lon,
            "vmax": float(f[8]) if f[8].isdigit() else None,
            "mslp": float(f[9]) if f[9].isdigit() else None,
            "ty": f[10],
            # DIR and SPEED sit past the optional columns and are often blank.
            "bearing": float(f[25]) if len(f) > 25 and f[25].isdigit() else None,
            "fwd_kt": float(f[26]) if len(f) > 26 and f[26].isdigit() else None,
            "name": f[27] if len(f) > 27 else "",
        })
        rad = int(f[11]) if f[11].lstrip("-").isdigit() else 0
        if rad in TIERS:
            for q, v in zip(QUADS, f[13:17]):
                # 0 is "no wind field in this quadrant", which for a radius is
                # the same statement as absent, so both land as missing.
                r[f"r{rad}_{q}"] = float(v) if v.isdigit() and float(v) > 0 else None
    return list(recs.values())


def _pad_radii(a):
    """Guarantee a column per tier/quadrant, since the builder indexes them."""
    for c in [f"r{t}_{q}" for t in TIERS for q in QUADS]:
        if c not in a.columns:
            a[c] = float("nan")
    return a


def live_adeck(year=None, stale_days=2, max_tau=120):
    """The current season's OFCL forecasts, refetched rather than read from cache.

    `fetch_deck` caches, which is right for a closed season: an archived deck
    never changes. The current season's deck grows with every advisory, so a
    cached copy would pin the dashboard's view of an approaching storm to
    whatever hour it was first fetched -- a hurricane would appear to stop
    moving. That is the one direction a hazard page must not fail in, so the
    cache is bypassed here.

    Trimmed to advisories from the last `stale_days`, which is what decides
    whether a storm is still in play. Older ones only cost time: every storm of
    the season is listed, and all but the active ones dissipated months ago.
    """
    now = pd.Timestamp.now("UTC").tz_localize(None)
    year = year or now.year
    body = _get(LIVE)
    if body is None:
        raise SystemExit(f"cannot list {LIVE} -- no live cyclone forecast")
    nums = sorted({int(m) for m in
                   re.findall(rf"a{BASIN}(\d{{2}}){year}\.dat\.gz",
                              body.decode(errors="replace"))})
    rows = []
    for num in (n for n in nums if n not in INVEST):
        deck = _get(f"{LIVE}a{BASIN}{num:02d}{year}.dat.gz")
        if deck is None:
            continue
        rows.extend(parse_deck(deck, f"{BASIN.upper()}{num:02d}{year}"))
    if not rows:
        # An empty season is a real state, not a failure: the dashboard still
        # has to render, with every county quiet.
        return _pad_radii(pd.DataFrame(
            columns=["storm_id", "issued", "tau", "lat", "lon", "vmax", "mslp",
                     "ty", "bearing", "fwd_kt", "name"]))
    a = _pad_radii(pd.DataFrame(rows))
    a = a[a["issued"] >= now - pd.Timedelta(days=stale_days)]
    a = a[a["tau"].between(0, max_tau)]
    return a.sort_values(["storm_id", "issued", "tau"]).reset_index(drop=True)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for year in range(START, END + 1):
        base, nums = deck_names(year)
        if not nums:
            missing.append(year)
            print(f"{year}: no decks listed")
            continue
        got = 0
        for num in nums:
            body = fetch_deck(base, year, num)
            if body is None:
                continue
            r = parse_deck(body, f"{BASIN.upper()}{num:02d}{year}")
            rows.extend(r)
            got += bool(r)
        print(f"{year}: {got}/{len(nums)} decks with an official forecast, "
              f"{len(rows):,} rows so far", flush=True)

    if not rows:
        raise SystemExit("no OFCL rows parsed -- re-probe the archive layout")
    # A tier absent from every parsed line leaves no column at all, and the
    # feature builder must see a column of nulls rather than a KeyError.
    a = _pad_radii(pd.DataFrame(rows))
    a = a.sort_values(["storm_id", "issued", "tau"]).reset_index(drop=True)
    a.to_parquet(DEST, index=False)

    rad = a[[f"r{t}_{q}" for t in TIERS for q in QUADS]].notna().any(axis=1)
    print(f"\nwrote {DEST}  {len(a):,} forecast positions  "
          f"{a['storm_id'].nunique()} storms  "
          f"{a['issued'].min().date()}..{a['issued'].max().date()}")
    print(f"  {rad.mean() * 100:.1f}% carry a wind radius "
          f"(first {a.loc[rad, 'issued'].min().date()})")
    print(f"  taus present: {sorted(a['tau'].unique())}")
    if missing:
        print(f"  no decks for: {missing}")


if __name__ == "__main__":
    sys.exit(main())

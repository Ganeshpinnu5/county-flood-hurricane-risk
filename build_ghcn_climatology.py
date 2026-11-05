"""County rainfall climatology for the whole country, from GHCN-Daily.

The 356 Gulf counties take `p95_prev3`, `p95_fwd3` and `mean_prev30` from the
station panel in gulf_coast_historical.parquet. Scoring the rest of the country
needs the same three constants for ~2,750 more counties, and the honest way to
get them is the same source and the same recipe rather than a different one.

An earlier attempt pulled them from Open-Meteo's ERA5 archive. Two problems.
ERA5 is a gridded reanalysis, so a county's "normal" measured that way is a
different quantity from the station mean the panel was built on -- and these
constants are the denominators of the ratio features, which carry the most
weight in the flood model, so the gap would have biased every county outside
the Gulf in the same direction. It was also capped at roughly 220 counties an
hour, which made national coverage a week's work.

Reading NOAA's year files directly fixes both. Each US station is assigned to a
county by point-in-polygon, then aggregated exactly as build_daily.py does:
county-day mean across the stations reporting that day, the grid completed so
rolling windows span calendar time rather than report days, then the same
windows and the same CLIM_YEARS span.

Counties with no station in the window keep no climatology and stay unscored.
That is the same rule the Gulf panel already followed -- it covers 356 of the
roughly 400 counties in those five states -- and it is preferable to handing a
county a neighbour's normal.

Units: the by_year archive reports PRCP in tenths of a millimetre and the panel
is in millimetres, so values are divided by 10. `verify_units` checks that
against the panel on stations the two share, because getting it wrong would
scale every national denominator by ten and still look entirely plausible.
"""
import argparse
import gzip
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
GHCN = ROOT / "refdata" / "cache" / "ghcn"
COUNTIES = ROOT / "artifacts" / "stage4" / "geo_src" / "counties.json"
PANEL = pathlib.Path.home() / "ganesh project" / "gulf_coast_historical.parquet"
OUT = ROOT / "refdata" / "county_climatology_ghcn.csv"

# Must match CLIM_YEARS in build_daily.py, which defaults to the first nine
# years of the panel window (PANEL_START=2000).
CLIM_YEARS = (2000, 2008)

# 2009 is read but never aggregated. build_daily.py computes the rolling windows
# over the whole 2000-2026 panel and only then masks to CLIM_YEARS, so its
# forward-looking sum on 2008-12-31 sees into January 2009. Stopping the grid at
# 2008-12-31 would truncate the last three days of p95_fwd3 instead.
YEARS = range(CLIM_YEARS[0], CLIM_YEARS[1] + 2)

# Tenths of a millimetre to millimetres.
PRCP_SCALE = 0.1

# The Gulf panel's 523 stations are all COOP (USC) or airport/WBAN (USW): the
# long-record, quality-controlled networks. GHCN also carries US1 (CoCoRaHS,
# which barely existed before 2009), USR (RAWS) and USS (SNOTEL). Including
# those would build the national counties from a different kind of network than
# the Gulf counties, which is the bias this script exists to avoid.
NETWORKS = ("USC", "USW")

# Minimum share of days in the window a station must report to be used. The
# county-day value is a mean across stations, so a station that reports for one
# season pulls that county's normal toward that season's weather.
MIN_COVERAGE = 0.8

COLS = ["county_fips", "p95_prev3", "p95_fwd3", "mean_prev30", "n_days",
        "n_stations_ghcn", "clim_source"]

# Stations borrowed from outside the county when it contains none of its own,
# and the distance past which borrowing is refused. A rainfall *normal* varies
# far more smoothly in space than daily rainfall does, so a gauge one county
# over is a fair estimate of the local baseline where a gauge across a mountain
# range is not. Counties with nothing inside this radius stay unscored.
FALLBACK_K = 2
FALLBACK_MAX_KM = 150.0


def stations():
    """US stations and their coordinates, from the fixed-width inventory."""
    path = GHCN / "ghcnd-stations.txt"
    df = pd.read_fwf(
        path,
        colspecs=[(0, 11), (12, 20), (21, 30), (38, 40)],
        names=["STATION", "lat", "lon", "state"],
        dtype={"STATION": str, "state": str},
    )
    # GHCN ids are prefixed by country. US stations carry the 48 states plus DC;
    # nearby Canadian and Mexican stations would land outside every polygon
    # anyway, but dropping them first keeps the spatial join small.
    df = df[df["STATION"].str[:3].isin(NETWORKS)]
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def station_counties(stn):
    """Assign each station to the county polygon containing it."""
    import shapely
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    feats = json.loads(COUNTIES.read_text())["features"]
    fips = [f["properties"]["STATE"] + f["properties"]["COUNTY"] for f in feats]
    geoms = [shape(f["geometry"]) for f in feats]
    tree = STRtree(geoms)

    pts = shapely.points(stn["lon"].to_numpy(), stn["lat"].to_numpy())
    si, gi = tree.query(pts, predicate="within")

    out = stn.iloc[si].copy()
    out["county_fips"] = [fips[i] for i in gi]
    # A point on a shared boundary can match twice; the first is deterministic
    # because STRtree returns input indices in order.
    out = out.drop_duplicates("STATION", keep="first")
    print(f"{len(out)} of {len(stn)} US stations fell inside a county")
    return out[["STATION", "county_fips"]]


def _iter_prcp(year, wanted):
    """Quality-passed PRCP rows for `wanted` stations, one chunk at a time.

    The year files are ~35M rows, and only about a tenth of that is PRCP from a
    station we use. Filtering inside the loop keeps peak memory to one chunk.
    """
    with gzip.open(GHCN / f"{year}.csv.gz", "rt") as fh:
        for ch in pd.read_csv(
            fh,
            header=None,
            usecols=[0, 1, 2, 3, 5],
            names=["STATION", "DATE", "ELEMENT", "VALUE", "QFLAG"],
            dtype={"STATION": str, "DATE": str, "ELEMENT": str,
                   "VALUE": "float64", "QFLAG": str},
            chunksize=2_000_000,
        ):
            # A non-empty quality flag means the value failed one of NOAA's
            # checks, so it is dropped rather than averaged in.
            ch = ch[(ch["ELEMENT"] == "PRCP") & ch["QFLAG"].isna()
                    & ch["STATION"].isin(wanted)]
            if len(ch):
                yield ch


def station_coverage(wanted, tag=None):
    """Days reported per station across the window (pass one).

    Cached on `tag`, because this is ten full gzip passes and the selection
    rules downstream of it are exactly what we want to iterate on cheaply.
    """
    cache = GHCN / f"coverage_{tag}.csv" if tag else None
    if cache and cache.exists():
        s = pd.read_csv(cache, index_col=0).iloc[:, 0]
        print(f"  coverage from cache: {len(s)} stations ({cache.name})")
        return s

    total = pd.Series(dtype="int64")
    for y in YEARS:
        n = pd.Series(dtype="int64")
        for ch in _iter_prcp(y, wanted):
            n = n.add(ch["STATION"].value_counts(), fill_value=0)
        total = total.add(n, fill_value=0)
        print(f"  {y}: {int(n.sum()):,} obs from {len(n)} stations", flush=True)
    if cache:
        total.to_csv(cache)
    return total


def cap_stations(keep, cov, cap):
    """Keep only the `cap` best-reporting stations in each county.

    The Gulf panel averages 1.47 stations per county; a raw polygon join
    averages 7.9. That matters because the county-day value is a mean across
    stations, so a denser network smooths it and drags the 95th percentile
    down -- measurably, 2.6% low against the published constants. Capping the
    count makes the national network as sparse as the one the models were
    trained on, so the climatology stays commensurate instead of needing a
    fitted correction factor.
    """
    k = keep.copy()
    k["_cov"] = k["STATION"].map(cov).fillna(0)
    k = (k.sort_values(["county_fips", "_cov", "STATION"],
                       ascending=[True, False, True])
          .groupby("county_fips", sort=False).head(cap))
    return k.drop(columns="_cov")


def read_year(year, keep, wanted):
    """County-day mean rainfall for one year file (pass two)."""
    chunks = []
    for ch in _iter_prcp(year, wanted):
        ch = ch.merge(keep, on="STATION", how="inner")
        if len(ch):
            chunks.append(ch[["county_fips", "DATE", "VALUE"]])
    if not chunks:
        return pd.DataFrame(columns=["county_fips", "DATE", "prcp", "n_stn"])

    d = pd.concat(chunks, ignore_index=True)
    d["VALUE"] *= PRCP_SCALE
    g = (d.groupby(["county_fips", "DATE"])["VALUE"]
          .agg(prcp="mean", n_stn="size").reset_index())
    g["DATE"] = pd.to_datetime(g["DATE"], format="%Y%m%d")
    print(f"  {year}: {len(d):,} obs -> {len(g):,} county-days, "
          f"{g['county_fips'].nunique()} counties", flush=True)
    return g


def climatology(d):
    """The three constants, by the recipe in build_daily.py lines 66-97.

    A day with no station report stays NaN rather than becoming zero. That is
    what build_daily.py does, and the distinction is not cosmetic: zero-filling
    would tell `mean_prev30` that a county with reporting gaps is drier than it
    is, which inflates `ratio_prev30` for exactly the counties whose data is
    weakest. pandas skips NaN in both `rolling().sum()` and the aggregations, so
    `min_periods` decides when a window is too empty to use at all.
    """
    out = []
    # The grid is completed so a rolling window spans calendar time rather than
    # the days a station happened to report.
    days = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")
    in_clim = (days.year >= CLIM_YEARS[0]) & (days.year <= CLIM_YEARS[1])
    for fips, g in d.groupby("county_fips", sort=True):
        s = g.set_index("DATE")["prcp"].reindex(days)
        prev3 = s.rolling(3, min_periods=1).sum()
        prev30 = s.rolling(30, min_periods=10).sum()
        fwd3 = s.shift(-3).rolling(3, min_periods=1).sum()
        rep = g[g["DATE"].dt.year.between(*CLIM_YEARS)]
        out.append({
            "county_fips": fips,
            "p95_prev3": float(prev3[in_clim].quantile(0.95)),
            "p95_fwd3": float(fwd3[in_clim].quantile(0.95)),
            "mean_prev30": float(prev30[in_clim].mean()),
            "n_days": int(rep["DATE"].nunique()),
            "n_stations_ghcn": int(rep["n_stn"].max()) if len(rep) else 0,
        })
    return pd.DataFrame(out)


def verify_units(keep):
    """Check the tenths-to-mm scaling against the panel on shared stations.

    This is the one step that cannot be reasoned about safely. A factor of ten
    on every national denominator would leave the ratio features an order of
    magnitude too small, every county outside the Gulf would read as calmer
    than it is, and nothing on the dashboard would look broken.
    """
    if not PANEL.exists():
        print("  SKIP: panel not present, cannot verify units")
        return None
    w = pd.read_parquet(PANEL, columns=["STATION", "DATE", "PRCP"])
    w = w[w["DATE"].dt.year.between(*CLIM_YEARS)].dropna(subset=["PRCP"])

    year = CLIM_YEARS[0]
    raw = []
    with gzip.open(GHCN / f"{year}.csv.gz", "rt") as fh:
        for ch in pd.read_csv(
            fh, header=None, usecols=[0, 1, 2, 3, 5],
            names=["STATION", "DATE", "ELEMENT", "VALUE", "QFLAG"],
            dtype={"STATION": str, "DATE": str, "ELEMENT": str,
                   "VALUE": "float64", "QFLAG": str},
            chunksize=2_000_000,
        ):
            ch = ch[(ch["ELEMENT"] == "PRCP") & ch["QFLAG"].isna()]
            if len(ch):
                raw.append(ch[["STATION", "DATE", "VALUE"]])
    r = pd.concat(raw, ignore_index=True)
    r["DATE"] = pd.to_datetime(r["DATE"], format="%Y%m%d")

    m = w.merge(r, on=["STATION", "DATE"], how="inner")
    if m.empty:
        print("  no shared station-days: cannot verify units")
        return None

    nz = m[(m["PRCP"] > 0) & (m["VALUE"] > 0)]
    ratio = (nz["VALUE"] / nz["PRCP"]).median()
    scaled = (nz["VALUE"] * PRCP_SCALE - nz["PRCP"]).abs()
    print(f"  {len(m):,} shared station-days in {year}, "
          f"{nz['STATION'].nunique()} stations")
    print(f"  median raw/panel ratio: {ratio:.3f} "
          f"(10 means the archive is in tenths, as assumed)")
    print(f"  after x{PRCP_SCALE}: mean abs diff {scaled.mean():.4f} mm, "
          f"max {scaled.max():.3f} mm")
    ok = abs(ratio - 1.0 / PRCP_SCALE) < 0.05 and scaled.mean() < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}: scaling {PRCP_SCALE} "
          f"{'confirmed' if ok else 'DOES NOT HOLD -- do not use this output'}")
    return ok


def gulf_mapping():
    """The panel's own 523 station-to-county assignments."""
    m = pd.read_csv(pathlib.Path.home() / "ganesh project" /
                    "station_county_mapping (1).csv", dtype={"county_fips": str})
    m["county_fips"] = m["county_fips"].astype(str).str.zfill(5)
    return m[["STATION", "county_fips"]]


def compare_gulf(clim, label):
    """Score a climatology against the 356 Gulf rows the panel was built from.

    The Gulf counties are a known answer this pipeline never sees, so they are
    the only way to tell whether the recipe and the station-selection rule are
    right before trusting the national numbers.
    """
    known = pd.read_csv(ROOT / "refdata" / "county_static.csv",
                        dtype={"county_fips": str})
    known["county_fips"] = known["county_fips"].str.zfill(5)
    m = known.merge(clim, on="county_fips", suffixes=("_known", "_mine"))
    print(f"\n  {label}: matched {len(m)} of {len(known)} Gulf counties")
    if m.empty:
        return None
    print(f"  {'constant':14s}{'known':>9s}{'mine':>9s}{'ratio':>8s}"
          f"{'mean|d|':>9s}{'corr':>8s}")
    worst = 0.0
    for c in ("p95_prev3", "p95_fwd3", "mean_prev30"):
        a, b = m[c + "_known"], m[c + "_mine"]
        ok = a.notna() & b.notna()
        rel = ((b[ok] - a[ok]).abs() / a[ok]).mean()
        worst = max(worst, rel)
        print(f"  {c:14s}{a[ok].mean():9.2f}{b[ok].mean():9.2f}"
              f"{b[ok].mean()/a[ok].mean():8.3f}{(a[ok]-b[ok]).abs().mean():9.2f}"
              f"{a[ok].corr(b[ok]):8.4f}")
    print(f"  mean relative error {worst:.1%}")
    return worst


def read_station_level(keep, wanted):
    """Station-day rainfall for the whole window, kept in memory.

    Every candidate station rule differs only in which stations it keeps, so
    the gzip files are read once here and each rule is then evaluated without
    touching disk again. Only viable for a regional subset -- the national set
    is ten times larger and goes through read_year instead.
    """
    parts = []
    for y in YEARS:
        rows = []
        for ch in _iter_prcp(y, wanted):
            rows.append(ch[["STATION", "DATE", "VALUE"]])
        if not rows:
            continue
        d = pd.concat(rows, ignore_index=True)
        d["VALUE"] *= PRCP_SCALE
        d["DATE"] = pd.to_datetime(d["DATE"], format="%Y%m%d")
        parts.append(d)
        print(f"  {y}: {len(d):,} station-days", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d = d.merge(keep[["STATION", "county_fips"]], on="STATION", how="inner")
    d["STATION"] = d["STATION"].astype("category")
    d["county_fips"] = d["county_fips"].astype("category")
    return d


def climatology_from(sl, stations_keep):
    """County-day means over one station selection, then the constants."""
    d = sl[sl["STATION"].isin(stations_keep)]
    g = (d.groupby(["county_fips", "DATE"], observed=True)["VALUE"]
          .agg(prcp="mean", n_stn="size").reset_index())
    g["county_fips"] = g["county_fips"].astype(str)
    print(f"  {len(d):,} obs -> {len(g):,} county-days, "
          f"{g['county_fips'].nunique()} counties")
    return climatology(g)


def build(keep, wanted):
    parts = [read_year(y, keep, wanted) for y in YEARS]
    d = pd.concat(parts, ignore_index=True)
    d = (d.groupby(["county_fips", "DATE"])
          .agg(prcp=("prcp", "mean"), n_stn=("n_stn", "max")).reset_index())
    print(f"\n{len(d):,} county-days, {d['county_fips'].nunique()} counties")
    return climatology(d)


def county_centroids():
    """Centroid lat/lon for every county, from the national static table."""
    d = pd.read_csv(ROOT / "refdata" / "county_static_national.csv",
                    dtype={"county_fips": str})
    d["county_fips"] = d["county_fips"].str.zfill(5)
    return d.dropna(subset=["lat", "lon"])[["county_fips", "lat", "lon"]]


def nearest_fallback(stn, qualifying, have):
    """Assign the FALLBACK_K nearest outside stations to each stationless county.

    About 290 counties contain no gauge that reports on MIN_COVERAGE of the
    window, so a polygon join leaves them with no normal and the live scorer
    drops them. Borrowing the closest qualifying gauges is defensible where
    imputing a daily value would not be: these three constants are long-run
    summaries, and the 95th percentile of three-day rainfall changes slowly
    across a hundred kilometres of the same climate.

    Marked `nearest` in the output so the dashboard can disclose it rather than
    present a borrowed baseline as if the county had measured it.
    """
    cen = county_centroids()
    missing = cen[~cen["county_fips"].isin(have)]
    if missing.empty:
        return pd.DataFrame(columns=["STATION", "county_fips"])

    src = stn[stn["STATION"].isin(qualifying)].reset_index(drop=True)
    slat = np.radians(src["lat"].to_numpy())
    slon = np.radians(src["lon"].to_numpy())
    clat = np.radians(missing["lat"].to_numpy())[:, None]
    clon = np.radians(missing["lon"].to_numpy())[:, None]

    # Haversine. 290 x 7,400 is small enough to do as one dense block.
    dlat, dlon = slat[None, :] - clat, slon[None, :] - clon
    a = (np.sin(dlat / 2) ** 2
         + np.cos(clat) * np.cos(slat[None, :]) * np.sin(dlon / 2) ** 2)
    km = 6371.0 * 2 * np.arcsin(np.sqrt(a))

    idx = np.argsort(km, axis=1)[:, :FALLBACK_K]
    rows, refused = [], 0
    for i, fips in enumerate(missing["county_fips"].to_numpy()):
        picked = [j for j in idx[i] if km[i, j] <= FALLBACK_MAX_KM]
        if not picked:
            refused += 1
            continue
        for j in picked:
            rows.append((src.at[j, "STATION"], fips))

    out = pd.DataFrame(rows, columns=["STATION", "county_fips"])
    n = out["county_fips"].nunique()
    print(f"  fallback: {n} of {len(missing)} stationless counties borrow "
          f"{len(out)} stations within {FALLBACK_MAX_KM:.0f} km")
    if refused:
        print(f"  {refused} counties have no gauge within "
              f"{FALLBACK_MAX_KM:.0f} km and stay unscored")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--gulf-check", action="store_true",
                    help="rebuild the Gulf counties from the panel's own 523 "
                         "stations and from the polygon join, and compare both "
                         "against the published constants")
    ap.add_argument("--caps", default="", help="comma-separated station caps to "
                    "test on the Gulf counties, e.g. 1,2,3")
    ap.add_argument("--cap", type=int, default=0,
                    help="stations per county for the national build")
    ap.add_argument("--fill-missing", action="store_true",
                    help="keep the counties already built and only add the "
                         "stationless ones via the nearest-gauge fallback")
    args = ap.parse_args()

    stn = stations()
    print(f"{len(stn)} {'/'.join(NETWORKS)} stations in the inventory")
    keep = station_counties(stn)

    print("\nunits check")
    ok = verify_units(keep)
    if args.verify_only:
        return
    if ok is False:
        raise SystemExit("unit check failed; refusing to write climatology")

    n_days = len(pd.date_range(f"{CLIM_YEARS[0]}-01-01",
                               f"{CLIM_YEARS[1]}-12-31", freq="D"))

    if args.gulf_check or args.caps:
        gulf = set(pd.read_csv(ROOT / "refdata" / "county_static.csv",
                               dtype={"county_fips": str})["county_fips"]
                   .str.zfill(5))
        sub = keep[keep["county_fips"].isin(gulf)]
        cov = station_coverage(set(sub["STATION"]), tag="gulf")
        good = cov[cov >= MIN_COVERAGE * n_days].index
        covered = sub[sub["STATION"].isin(set(good))]

        if args.gulf_check:
            gm = gulf_mapping()
            print("\n--- A. the panel's own 523 stations: tests the recipe ---")
            compare_gulf(build(gm, set(gm["STATION"])), "panel stations")

            print(f"\n--- B. polygon join, coverage >= {MIN_COVERAGE:.0%}: "
                  f"{len(good)} of {len(sub)} stations ---")
            compare_gulf(build(covered, set(good)),
                         f"polygon join, {MIN_COVERAGE:.0%} coverage")

        caps = [int(c) for c in args.caps.split(",") if c.strip()]
        if caps:
            print(f"\nreading {len(good)} covered Gulf stations once")
            sl = read_station_level(covered, set(good))
            for cap in caps:
                c = cap_stations(covered, cov, cap)
                per = len(c) / c["county_fips"].nunique()
                print(f"\n--- C. polygon join + {MIN_COVERAGE:.0%} coverage, "
                      f"max {cap} station(s)/county: {len(c)} stations, "
                      f"{per:.2f} per county (panel: 1.47) ---")
                compare_gulf(climatology_from(sl, set(c["STATION"])),
                             f"capped at {cap}")
        return

    print("\npass 1: station coverage")
    cov = station_coverage(set(keep["STATION"]), tag="national")
    good = set(cov[cov >= MIN_COVERAGE * n_days].index)
    print(f"{len(good)} of {len(keep)} stations report on "
          f">={MIN_COVERAGE:.0%} of {n_days} days")

    sel = keep[keep["STATION"].isin(good)]
    if args.cap:
        sel = cap_stations(sel, cov, args.cap)
        print(f"capped at {args.cap}/county: {len(sel)} stations, "
              f"{len(sel) / sel['county_fips'].nunique():.2f} per county")

    # Only the stationless counties are rebuilt, so a finished national run is
    # topped up rather than repeated: pass 2 is ten gzip passes either way, but
    # this keeps the counties that have their own gauge bit-identical.
    if args.fill_missing:
        if not OUT.exists():
            raise SystemExit(f"{OUT.name} does not exist; run without "
                             "--fill-missing first")
        done = pd.read_csv(OUT, dtype={"county_fips": str})
        done["county_fips"] = done["county_fips"].str.zfill(5)
        if "clim_source" not in done.columns:
            done["clim_source"] = "county"
        have = set(done.loc[done["clim_source"] == "county", "county_fips"])
        print(f"\n{len(have)} counties already have their own gauge")

        print("\nnearest-station fallback")
        fb = nearest_fallback(stn, good, have)
        if fb.empty:
            print("nothing to fill")
            return

        print("\npass 2: county-day aggregation for borrowed counties only")
        add = build(fb, set(fb["STATION"]))
        add["clim_source"] = "nearest"
        clim = (pd.concat([done[done["clim_source"] == "county"], add],
                          ignore_index=True)
                  .drop_duplicates("county_fips", keep="first")
                  .sort_values("county_fips").reset_index(drop=True))
    else:
        print("\nnearest-station fallback")
        fb = nearest_fallback(stn, good, set(sel["county_fips"]))

        print("\npass 2: county-day aggregation")
        own = build(sel, set(sel["STATION"]))
        own["clim_source"] = "county"
        clim = own
        if not fb.empty:
            add = build(fb, set(fb["STATION"]))
            add["clim_source"] = "nearest"
            clim = (pd.concat([own, add], ignore_index=True)
                      .drop_duplicates("county_fips", keep="first")
                      .sort_values("county_fips").reset_index(drop=True))

    clim.to_csv(OUT, index=False, columns=COLS)
    n_own = (clim["clim_source"] == "county").sum()
    print(f"\nwrote {OUT} ({len(clim)} counties: "
          f"{n_own} own gauge, {len(clim) - n_own} borrowed)")
    for c in ("p95_prev3", "p95_fwd3", "mean_prev30"):
        print(f"  {c:12s} {clim[c].min():7.1f} - {clim[c].max():7.1f}  "
              f"mean {clim[c].mean():7.1f}")
    compare_gulf(clim, "national build vs Gulf panel")


if __name__ == "__main__":
    main()

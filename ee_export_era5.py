"""Pull the national county-day hydrology panel out of ERA5-Land.

The interactive endpoint measured 66 s for a single day over 3,109 counties,
so a 27-year pull through it would take a week of wall clock. Batch Export
tasks get a different, parallel pool, so the work is submitted as one job per
year-month and left to run.

One month per job rather than one year: a year is ~1.1M output rows, which is
inside Drive's limits but close enough to EE's aggregation ceiling that a
single failure would cost a whole year of work. A month is ~93k rows and a
failure is cheap to resubmit.

What the bands are for, since several are new to this project:

    total_precipitation_sum       water in
    tp_max                        the same field, spatial max over the county.
                                  County-mean rain hides flash flooding: a
                                  10 mm county average can be a 90 mm cell
                                  over one town.
    volumetric_soil_water_1..3    water already in storage. This is the
                                  variable the panel has never had, and the
                                  reason an identical rain day floods one
                                  county and not another.
    snow_depth_water_equivalent   water held as snow
    snowmelt_sum                  that water being released
    snowfall_sum                  precipitation that did not run off today
    runoff_sum, surface_runoff    water the land has already shed
    temperature_2m                melt driver, and the rain-vs-snow split

Deliberately not pulled: soil layer 4 (0.3-3 m, too slow to matter at a 3-day
horizon) and snow_depth (redundant with its own water equivalent, which is
the flood-relevant quantity).

Note the day boundary is UTC, which the GHCN station panel's is not. That
offset is real and has to be disclosed, not hidden, when the two are compared.

Usage:
    python ee_export_era5.py --check          verify one month computes
    python ee_export_era5.py --submit         queue every outstanding month
    python ee_export_era5.py --status         poll the task list
"""
import argparse
import sys

import ee
import pandas as pd

PROJECT = "climate-514021"
COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
SCALE = 11132          # native ERA5-Land grid; resampling buys nothing
FOLDER = "crisk_era5"
COUNTY_CSV = "refdata/county_static_national.csv"

# ERA5-Land Daily Aggregates currently ends 2026-10-31. Asking for November
# yields empty months that look like failures.
START = (2000, 1)
END = (2026, 10)

# The national static table was built on a pre-2015 FIPS vintage, so two of
# its codes no longer exist in TIGER. Caught here rather than at merge time,
# where it would have looked like missing weather instead of a stale code.
#   46113 Shannon County SD was renamed Oglala Lakota County (46102) in 2015;
#         same polygon, so the data is simply fetched under the new code and
#         relabelled with the one the panel uses.
#   51515 Bedford city VA was absorbed into Bedford County (51019) in 2013.
#         There is no polygon to fetch; it is not a county any more.
FIPS_ALIAS = {"46102": "46113"}
FIPS_RETIRED = {"51515"}

MEAN_BANDS = [
    "total_precipitation_sum",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "snow_depth_water_equivalent",
    "snowmelt_sum",
    "snowfall_sum",
    "runoff_sum",
    "surface_runoff_sum",
    "temperature_2m",
]
MAX_BANDS = ["tp_max"]
OUT_COLS = ["county_fips", "date"] + MEAN_BANDS + MAX_BANDS


def counties():
    """TIGER county polygons, cut to the counties the panel actually uses.

    TIGER carries 3,233 features; the panel carries 3,109. The difference is
    territories plus the two retired codes handled by FIPS_ALIAS and
    FIPS_RETIRED. Joining to the panel's own list keeps the two in step
    instead of discovering the mismatch at merge time.

    Features come back keyed by the panel's county_fips, not TIGER's GEOID,
    so the exported CSVs join to the panel without further translation.
    """
    want = set(pd.read_csv(COUNTY_CSV,
                           dtype={"county_fips": str})["county_fips"])
    want -= FIPS_RETIRED
    back = {v: k for k, v in FIPS_ALIAS.items()}
    geoids = sorted({back.get(c, c) for c in want})
    relabel = ee.Dictionary(FIPS_ALIAS)
    fc = ee.FeatureCollection("TIGER/2018/Counties").filter(
        ee.Filter.inList("GEOID", ee.List(geoids)))
    return fc.map(lambda f: f.set(
        "county_fips", relabel.get(f.get("GEOID"), f.get("GEOID"))))


def reducer():
    # Two forEach reducers with sharedInputs=False: the first ten inputs get
    # the mean, the eleventh gets the max. Output names come straight from the
    # band names, so the CSV header needs no translation.
    return (ee.Reducer.mean().forEach(MEAN_BANDS)
            .combine(ee.Reducer.max().forEach(MAX_BANDS), sharedInputs=False))


def month_fc(fc_counties, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    ic = (ee.ImageCollection(COLLECTION)
          .filterDate(start, start.advance(1, "month")))
    red = reducer()

    def per_day(img):
        date = img.date().format("YYYY-MM-dd")
        stack = img.select(MEAN_BANDS).addBands(
            img.select(["total_precipitation_sum"]).rename("tp_max"))
        return stack.reduceRegions(
            collection=fc_counties, reducer=red, scale=SCALE, tileScale=4,
        ).map(lambda f: f.set("date", date))

    return ic.map(per_day).flatten()


def months():
    out = []
    y, m = START
    while (y, m) <= END:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def name(year, month):
    return f"era5_{year}{month:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", metavar="YYYY-MM",
                    help="submit a single month; for smoke tests and for "
                         "resubmitting one that failed")
    a = ap.parse_args()
    ee.Initialize(project=PROJECT)

    if a.status:
        tasks = [t for t in ee.data.getTaskList()
                 if str(t.get("description", "")).startswith("era5_")]
        tally = {}
        for t in tasks:
            tally[t["state"]] = tally.get(t["state"], 0) + 1
        print(f"{len(tasks)} era5 tasks: {tally}")
        for t in tasks:
            if t["state"] == "FAILED":
                print(f"  FAILED {t['description']}: "
                      f"{t.get('error_message', '')[:200]}")
        return

    cty = counties()
    n = cty.size().getInfo()
    print(f"counties matched: {n}")
    if n != 3108:
        sys.exit(f"{n} counties matched TIGER, expected 3108 "
                 f"(3109 in the panel, less Bedford city); refusing to export")

    if a.check:
        # Two days over a handful of counties. The point is to prove the
        # reducer and band stack compute at all before queueing 300 jobs,
        # not to validate the numbers.
        probe = cty.limit(5)
        start = ee.Date.fromYMD(2020, 8, 1)
        ic = (ee.ImageCollection(COLLECTION)
              .filterDate(start, start.advance(2, "day")))
        red = reducer()

        def per_day(img):
            d = img.date().format("YYYY-MM-dd")
            stack = img.select(MEAN_BANDS).addBands(
                img.select(["total_precipitation_sum"]).rename("tp_max"))
            return stack.reduceRegions(
                collection=probe, reducer=red, scale=SCALE, tileScale=4,
            ).map(lambda f: f.set("date", d))

        rows = (ic.map(per_day).flatten()
                .select(OUT_COLS, None, False).getInfo()["features"])
        print(f"probe rows: {len(rows)} (expect 10)")
        for r in rows[:2]:
            print(" ", r["properties"])
        return

    if not (a.submit or a.only):
        ap.error("pass --check, --submit, --only or --status")

    if a.only:
        y, m = (int(x) for x in a.only.split("-"))
        nm = name(y, m)
        task = ee.batch.Export.table.toDrive(
            collection=month_fc(cty, y, m), description=nm, folder=FOLDER,
            fileNamePrefix=nm, fileFormat="CSV", selectors=OUT_COLS)
        task.start()
        print(f"queued {nm} (id {task.id})")
        return

    existing = {t["description"]: t["state"] for t in ee.data.getTaskList()}
    queued = skipped = 0
    for y, m in months():
        nm = name(y, m)
        st = existing.get(nm)
        if st in ("COMPLETED", "READY", "RUNNING"):
            skipped += 1
            continue
        task = ee.batch.Export.table.toDrive(
            collection=month_fc(cty, y, m), description=nm, folder=FOLDER,
            fileNamePrefix=nm, fileFormat="CSV", selectors=OUT_COLS)
        task.start()
        queued += 1
        print(f"queued {nm}")
    print(f"\n{queued} queued, {skipped} already done or running")


if __name__ == "__main__":
    main()

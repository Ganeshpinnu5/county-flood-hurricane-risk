"""Fold the monthly Earth Engine CSVs into one national county-day table.

ee_export_era5.py writes 322 monthly CSVs to Drive in native ERA5 units. This
reads them back, converts to the units the rest of the project already speaks
(mm and degrees C, not metres and kelvin), checks the grid is complete, and
writes a single parquet.

Feature engineering deliberately does not happen here. This produces the raw
county-day observations; rolling windows, ratios and lags stay in build_daily
alongside the ones that already exist, so there is one place where a feature
is defined rather than two.

The one thing worth knowing about the output: `snow_depth_water_equivalent`
and the soil moisture layers are *state* variables, not accumulations. They
describe the ground at a moment. The 7-day lag that makes them honest at
serve time (see the serve-time equivalence task) is applied downstream, not
here, because the raw table should stay a faithful record of what ERA5 said.

Usage:
    python build_era5_panel.py [--src DIR] [--out PATH]
"""
import argparse
import pathlib
import sys

import pandas as pd

SRC = pathlib.Path.home() / "ganesh project" / "era5_export"
OUT = pathlib.Path("artifacts/era5_county_daily.parquet")
COUNTY_CSV = "refdata/county_static_national.csv"

FIRST, LAST = "2000-01-01", "2026-10-31"

# ERA5-Land depths and accumulations are metres. Everything the project
# already speaks is millimetres.
M_TO_MM = [
    "total_precipitation_sum", "tp_max", "snowfall_sum", "snowmelt_sum",
    "runoff_sum", "surface_runoff_sum", "snow_depth_water_equivalent",
]
KELVIN = ["temperature_2m"]
SOIL = ["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3"]
VALUE_COLS = M_TO_MM + KELVIN + SOIL

RENAME = {
    "total_precipitation_sum": "era5_prcp",
    "tp_max": "era5_prcp_max",
    "snowfall_sum": "era5_snowfall",
    "snowmelt_sum": "era5_snowmelt",
    "runoff_sum": "era5_runoff",
    "surface_runoff_sum": "era5_runoff_surface",
    "snow_depth_water_equivalent": "era5_swe",
    "temperature_2m": "era5_tmean",
    "volumetric_soil_water_layer_1": "era5_soil1",
    "volumetric_soil_water_layer_2": "era5_soil2",
    "volumetric_soil_water_layer_3": "era5_soil3",
}


def read_all(src):
    files = sorted(src.glob("era5_*.csv"))
    if not files:
        sys.exit(f"no era5_*.csv under {src}")
    print(f"reading {len(files)} monthly files from {src}")

    frames = []
    for i, f in enumerate(files, 1):
        d = pd.read_csv(f, dtype={"county_fips": str})
        missing = [c for c in ["county_fips", "date"] + VALUE_COLS
                   if c not in d.columns]
        if missing:
            sys.exit(f"{f.name} is missing columns: {missing}")
        frames.append(d[["county_fips", "date"] + VALUE_COLS])
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=pathlib.Path, default=SRC)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    a = ap.parse_args()

    d = read_all(a.src)
    d["county_fips"] = d["county_fips"].str.zfill(5)
    d["DATE"] = pd.to_datetime(d["date"])
    d = d.drop(columns=["date"])

    dup = d.duplicated(subset=["county_fips", "DATE"]).sum()
    if dup:
        # Resubmitting a month without deleting the old CSV is the way this
        # happens, and a silent duplicate would double-count rainfall.
        sys.exit(f"{dup:,} duplicate county-days; check for resubmitted "
                 f"months left alongside their replacements in {a.src}")

    for c in M_TO_MM:
        d[c] = d[c] * 1000.0
    for c in KELVIN:
        d[c] = d[c] - 273.15
    # ERA5 carries tiny negative accumulations (order 1e-25) as numerical
    # noise. Clipping is cosmetic for rain but matters for SWE, where a
    # negative would otherwise mean "less than no snow".
    for c in M_TO_MM:
        d[c] = d[c].clip(lower=0)

    d = d.rename(columns=RENAME).sort_values(["county_fips", "DATE"])
    d = d.reset_index(drop=True)

    want = set(pd.read_csv(COUNTY_CSV, dtype={"county_fips": str})
               ["county_fips"]) - {"51515"}
    got = set(d["county_fips"])
    if got != want:
        print(f"  WARNING: {len(want - got)} expected counties absent, "
              f"{len(got - want)} unexpected present")
        print(f"    absent sample: {sorted(want - got)[:8]}")

    span = pd.date_range(FIRST, LAST, freq="D")
    have = d["DATE"].nunique()
    if have != len(span):
        gaps = sorted(set(span) - set(d["DATE"].unique()))
        print(f"  WARNING: {len(span) - have} dates missing of {len(span)}")
        print(f"    first gaps: {[str(g.date()) for g in gaps[:8]]}")

    expect = len(got) * have
    print(f"\nrows {len(d):,} (counties {len(got)} x dates {have} "
          f"= {expect:,}; {'complete' if len(d) == expect else 'RAGGED'})")
    print(d[["era5_prcp", "era5_swe", "era5_soil1", "era5_tmean"]]
          .describe().T[["mean", "min", "max"]])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(a.out, index=False)
    print(f"\nwrote {a.out}  ({a.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

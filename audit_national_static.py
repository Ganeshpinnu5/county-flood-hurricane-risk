"""Check the two substitutions behind national coverage.

Extending the dashboard past the Gulf panel swapped the source of two inputs,
and each swap is a chance to be quietly wrong:

1. Social vulnerability. CDC only publishes county files ranked within a state,
   so the national ranks are rebuilt from the raw EP_ percentages. The rebuild
   is checked against the 534 Gulf rows the panel was actually built from,
   which are a known answer it never sees.

2. Rainfall climatology. Gulf counties take `p95_prev3`, `p95_fwd3` and
   `mean_prev30` from the GHCN station panel; everywhere else they come from
   ERA5. These are the denominators of the ratio features, so a systematic gap
   between the two sources biases every score outside the Gulf in the same
   direction. This measures the gap on counties that have both, which is the
   only honest way to state its size.
"""
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
GULF_SVI = pathlib.Path.home() / "ganesh project" / "svi_gulf_coast_counties (1).csv"
STATIC = ROOT / "refdata" / "county_static.csv"
CLIM = ROOT / "refdata" / "county_climatology.csv"


def check_svi():
    print("=" * 68)
    print("1. national SVI ranks vs the 534 Gulf rows the panel was built from")
    print("=" * 68)
    if not GULF_SVI.exists():
        print(f"  SKIP: {GULF_SVI} not present")
        return None

    import build_national_static as bns

    rebuilt = bns.svi_national()
    gulf = pd.read_csv(GULF_SVI, dtype={"FIPS": str})
    gulf["county_fips"] = gulf["FIPS"].str.zfill(5)
    m = gulf.merge(rebuilt, on="county_fips", suffixes=("_pub", "_new"))
    print(f"  matched {len(m)} of {len(gulf)} rows\n")

    print(f"  {'column':14s}{'max abs':>10s}{'mean abs':>11s}{'corr':>9s}")
    worst = 0.0
    for c in ["RPL_THEME1", "RPL_THEME4", "RPL_THEMES"]:
        a = pd.to_numeric(m[c + "_pub"], errors="coerce").mask(lambda s: s <= -999)
        b = m[c + "_new"]
        ok = a.notna() & b.notna()
        d = (a[ok] - b[ok]).abs()
        worst = max(worst, d.mean())
        print(f"  {c:14s}{d.max():10.6f}{d.mean():11.6f}{a[ok].corr(b[ok]):9.5f}")
    verdict = "PASS" if worst < 1e-3 else "FAIL"
    print(f"\n  {verdict}: mean abs error {worst:.2e} "
          f"(CDC publishes 4 decimals, so ~5e-05 is the floor)")
    return worst


def check_climatology():
    print()
    print("=" * 68)
    print("2. ERA5 vs GHCN climatology on counties that have both")
    print("=" * 68)

    overlap = ROOT / "refdata" / "cache" / "overlap_climatology.csv"
    src = overlap if overlap.exists() else CLIM
    if not src.exists():
        print("  SKIP: no climatology file yet")
        return None

    panel = pd.read_csv(STATIC, dtype={"county_fips": str})
    panel["county_fips"] = panel["county_fips"].str.zfill(5)
    era = pd.read_csv(src, dtype={"county_fips": str})
    era["county_fips"] = era["county_fips"].str.zfill(5)
    m = panel.merge(era, on="county_fips", suffixes=("_ghcn", "_era"))
    if m.empty:
        print("  no overlap: every ERA5 county is outside the panel.")
        print("  Run with --overlap to fetch ERA5 for a sample of Gulf counties.")
        return None

    print(f"  source: {src.name}, {len(m)} counties with both\n")
    print(f"  {'constant':14s}{'GHCN mean':>11s}{'ERA5 mean':>11s}"
          f"{'ERA5/GHCN':>11s}{'corr':>8s}")
    for c in ["p95_prev3", "p95_fwd3", "mean_prev30"]:
        a, b = m[c + "_ghcn"], m[c + "_era"]
        ok = a.notna() & b.notna()
        print(f"  {c:14s}{a[ok].mean():11.2f}{b[ok].mean():11.2f}"
              f"{b[ok].mean()/a[ok].mean():11.3f}{a[ok].corr(b[ok]):8.4f}")

    r = m["p95_prev3_era"] / m["p95_prev3_ghcn"]
    print(f"\n  p95_prev3 ratio per county: median {r.median():.3f}, "
          f"IQR {r.quantile(.25):.3f}-{r.quantile(.75):.3f}")
    print("  A ratio above 1 means ERA5 calls the county wetter than the "
          "stations do,\n  so ratio_prev3 comes out lower and the model reads "
          "the same rain as calmer.")
    return m


def fetch_overlap(n=40):
    """ERA5 for a sample of Gulf counties, so the two sources can be compared."""
    panel = pd.read_csv(STATIC, dtype={"county_fips": str})
    panel["county_fips"] = panel["county_fips"].str.zfill(5)
    s = panel.sample(n, random_state=0)[["county_fips", "lat", "lon"]]
    tgt = ROOT / "refdata" / "cache" / "overlap_targets.csv"
    tgt.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(tgt, index=False)
    out = ROOT / "refdata" / "cache" / "overlap_climatology.csv"
    print(f"fetching ERA5 for {n} Gulf counties -> {out}")
    subprocess.run(
        [sys.executable, str(ROOT / "fetch_climatology.py"),
         "--targets", str(tgt), "--out", str(out), "--workers", "5"],
        check=True,
    )
    return out


if __name__ == "__main__":
    if "--overlap" in sys.argv:
        fetch_overlap()
    check_svi()
    check_climatology()

"""Extend the per-county constants from the 356 Gulf counties to all 3,109.

`fetch_live.py` scores a county from two things: rainfall pulled live from
Open-Meteo, which works anywhere, and twelve constants that until now existed
only for the counties in the Gulf panel. This builds those constants nationally.

Two rules shape the result.

The 356 Gulf rows are copied through byte for byte. They are what the sealed run
trained and scored on, so re-deriving them from a different source would quietly
move the published numbers.

The social-vulnerability ranks have to be national. CDC publishes county files
ranked within each state, and a state rank is a different quantity: the Gulf
extract the panel was built from averages 0.73 on RPL_THEMES because Gulf
counties really are vulnerable relative to the whole country, where a
within-state ranking would average 0.5 by construction. So the national ranks
are rebuilt here from the raw EP_ percentages by CDC's documented method, and
`audit_national_static.py` checks the rebuild against the 534 Gulf rows that
already have a known answer.
"""
import concurrent.futures as cf
import json
import pathlib

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).parent
GEO = ROOT / "docs" / "data" / "geo.json"
STATIC = ROOT / "refdata" / "county_static.csv"
CLIM = ROOT / "refdata" / "county_climatology_ghcn.csv"
CACHE = ROOT / "refdata" / "cache" / "svi"
SVI_URL = "https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/{}_county.csv"

STATES = """Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware
DistrictofColumbia Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky
Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana
Nebraska Nevada NewHampshire NewJersey NewMexico NewYork NorthCarolina NorthDakota Ohio
Oklahoma Oregon Pennsylvania RhodeIsland SouthCarolina SouthDakota Tennessee Texas Utah
Vermont Virginia Washington WestVirginia Wisconsin Wyoming""".split()

THEMES = {
    1: ["EP_POV150", "EP_UNEMP", "EP_HBURD", "EP_NOHSDP", "EP_UNINSUR"],
    2: ["EP_AGE65", "EP_AGE17", "EP_DISABL", "EP_SNGPNT", "EP_LIMENG"],
    3: ["EP_MINRTY"],
    4: ["EP_MUNIT", "EP_MOBILE", "EP_CROWD", "EP_NOVEH", "EP_GROUPQ"],
}
ALL_EP = [v for vs in THEMES.values() for v in vs]

# geo.json predates the 2022 SVI vintage, so three counties it draws were
# renumbered upstream. These two are pure renames, so the successor's SVI is the
# same place measured under a new code. Connecticut is not: its 8 counties became
# 9 planning regions with no 1:1 correspondence, so those 8 stay unscored rather
# than take a neighbour's vulnerability.
FIPS_SUCCESSOR = {
    "46113": "46102",  # Shannon -> Oglala Lakota, 2015
    "51515": "51019",  # Bedford city merged into Bedford County, 2013
}
OUT_COLS = ["county_fips", "lat", "lon", "n_stations", "AREA_SQMI", "E_TOTPOP",
            "RPL_THEME1", "RPL_THEME4", "RPL_THEMES", "pop_density",
            "p95_prev3", "p95_fwd3", "mean_prev30"]


def pctile(s):
    """CDC's percentile rank: ties take the low rank, scaled over N-1.

    Verified against the published Gulf values to a mean absolute error of
    5.4e-05, which is the rounding in CDC's own four-decimal CSV.
    """
    return (s.rank(method="min") - 1) / (s.notna().sum() - 1)


def svi_national():
    CACHE.mkdir(parents=True, exist_ok=True)

    def grab(name):
        p = CACHE / f"{name}.csv"
        if not p.exists() or p.stat().st_size < 1500:
            r = requests.get(SVI_URL.format(name), timeout=90)
            r.raise_for_status()
            p.write_bytes(r.content)
        return pd.read_csv(p, dtype={"FIPS": str}, low_memory=False)

    with cf.ThreadPoolExecutor(8) as ex:
        us = pd.concat(list(ex.map(grab, STATES)), ignore_index=True)
    us["FIPS"] = us["FIPS"].str.zfill(5)

    # -999 is CDC's missing sentinel; left raw it would rank as least vulnerable.
    ep = {c: pd.to_numeric(us[c], errors="coerce").mask(lambda s: s <= -999)
          for c in ALL_EP}
    epl = pd.DataFrame({c: pctile(v) for c, v in ep.items()})

    spl = pd.DataFrame({
        t: epl[vs].sum(axis=1, min_count=len(vs)) for t, vs in THEMES.items()
    })
    out = pd.DataFrame({"county_fips": us["FIPS"].values})
    for t in THEMES:
        out[f"RPL_THEME{t}"] = pctile(spl[t]).values
    out["RPL_THEMES"] = pctile(spl.sum(axis=1, min_count=len(THEMES))).values
    out["AREA_SQMI"] = pd.to_numeric(us["AREA_SQMI"], errors="coerce").values
    out["E_TOTPOP"] = pd.to_numeric(us["E_TOTPOP"], errors="coerce").values
    return out


def main():
    geo = json.loads(GEO.read_text())
    names = geo["names"]
    base = pd.DataFrame(
        [(f.zfill(5), v[2], v[3]) for f, v in names.items()],
        columns=["county_fips", "lat", "lon"],
    ).sort_values("county_fips")
    print(f"{len(base)} mapped counties")

    svi = svi_national()
    print(f"{len(svi)} SVI counties, national ranks rebuilt")

    # Ranked nationally first, then relabelled: the successor's rank is its rank
    # among all counties, which is what the map's older code should inherit.
    alias = svi[svi["county_fips"].isin(FIPS_SUCCESSOR.values())].copy()
    back = {v: k for k, v in FIPS_SUCCESSOR.items()}
    alias["county_fips"] = alias["county_fips"].map(back)
    svi = pd.concat([svi, alias], ignore_index=True)
    print(f"  + {len(alias)} renamed-county aliases")

    d = base.merge(svi, on="county_fips", how="left")
    d["pop_density"] = d["E_TOTPOP"] / d["AREA_SQMI"].replace(0, np.nan)
    d["n_stations"] = np.nan

    if CLIM.exists():
        clim = pd.read_csv(CLIM, dtype={"county_fips": str})
        clim["county_fips"] = clim["county_fips"].str.zfill(5)
        clim = clim.drop_duplicates("county_fips", keep="last")
        d = d.merge(clim[["county_fips", "p95_prev3", "p95_fwd3", "mean_prev30"]],
                    on="county_fips", how="left")
        print(f"{clim.shape[0]} counties with GHCN climatology")
    else:
        for c in ("p95_prev3", "p95_fwd3", "mean_prev30"):
            d[c] = np.nan

    # The Gulf rows win on every column: they are the sealed run's inputs.
    gulf = pd.read_csv(STATIC, dtype={"county_fips": str})
    gulf["county_fips"] = gulf["county_fips"].str.zfill(5)
    d = d[~d["county_fips"].isin(set(gulf["county_fips"]))]
    d = pd.concat([gulf[OUT_COLS], d[OUT_COLS]], ignore_index=True)
    d = d.sort_values("county_fips").reset_index(drop=True)

    done = d[["p95_prev3", "mean_prev30"]].notna().all(axis=1).sum()
    print(f"\n{len(d)} rows, {len(gulf)} carried from the panel unchanged")
    print(f"{done} scoreable, {len(d) - done} still missing climatology")
    for c in OUT_COLS[1:]:
        print(f"  {c:12s} {d[c].notna().sum():5d} present  "
              f"{d[c].isna().sum():4d} missing")
    return d


if __name__ == "__main__":
    out = main()
    dest = ROOT / "refdata" / "county_static_national.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")

"""
density_revenue.py
-------------------
Urban3-style "value-per-acre" analysis for San Francisco neighborhoods.

Compares total assessed value (land + improvements) per acre across SF's
41 Analysis Neighborhoods, using the same underlying dataset your
sf-property-tool app already queries:

    Assessor Historical Secured Property Tax Rolls  (data.sfgov.org, id: wv5m-vpq2)
    Analysis Neighborhoods (boundaries/acreage)      (data.sfgov.org, id: j2bu-swwd)

This is the core Urban3/Minicozzi metric: total taxable value divided by
land area, NOT by parcel count or population. It's the number that shows
whether a dense mixed-use block is outproducing a low-density area on a
per-acre (i.e. per-unit-of-infrastructure) basis.

Usage:
    python density_revenue.py --year 2023
    python density_revenue.py --year 2023 --top 15 --plot

Requires: requests, pandas  (pip install requests pandas --break-system-packages)
Optional: matplotlib for --plot
"""

import argparse
import sys
from dataclasses import dataclass

import pandas as pd
import requests

ASSESSOR_ENDPOINT = "https://data.sfgov.org/resource/wv5m-vpq2.json"
NEIGHBORHOODS_ENDPOINT = "https://data.sfgov.org/resource/j2bu-swwd.json"

# SoQL field names on the assessor dataset as of the 2023-2024 schema.
# Verify against https://dev.socrata.com/foundry/data.sfgov.org/wv5m-vpq2
# if the city has re-published the dataset with renamed columns.
FIELD_YEAR = "closed_roll_fiscal_year"
FIELD_NEIGHBORHOOD = "analysis_neighborhood"
FIELD_LAND_VALUE = "assessed_land_value"
FIELD_IMPROVEMENT_VALUE = "assessed_improvement_value"
FIELD_EXEMPT_VALUE = "assessed_fixtures_value"  # adjust if you're tracking exemptions separately


@dataclass
class NeighborhoodMetric:
    neighborhood: str
    total_assessed_value: float
    acres: float
    value_per_acre: float
    parcel_count: int


def fetch_assessed_value_by_neighborhood(year: int, app_token: str | None = None) -> pd.DataFrame:
    """
    Pulls aggregated assessed value totals grouped by analysis_neighborhood
    for a given closed-roll fiscal year, using server-side SoQL aggregation
    so we don't have to page through ~220k parcel rows client-side.
    """
    params = {
        "$select": (
            f"{FIELD_NEIGHBORHOOD} as neighborhood, "
            f"sum({FIELD_LAND_VALUE} + {FIELD_IMPROVEMENT_VALUE}) as total_assessed_value, "
            f"count(*) as parcel_count"
        ),
        "$where": f"{FIELD_YEAR} = {year} AND {FIELD_NEIGHBORHOOD} IS NOT NULL",
        "$group": "neighborhood",
        "$limit": 100,  # there are only 41 neighborhoods; 100 is a safe ceiling
    }
    headers = {"X-App-Token": app_token} if app_token else {}

    resp = requests.get(ASSESSOR_ENDPOINT, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())

    if df.empty:
        raise RuntimeError(
            f"No rows returned for fiscal year {year}. Check that "
            f"{FIELD_YEAR} and {FIELD_NEIGHBORHOOD} match the current schema."
        )

    df["total_assessed_value"] = pd.to_numeric(df["total_assessed_value"])
    df["parcel_count"] = pd.to_numeric(df["parcel_count"])
    return df


def fetch_neighborhood_acreage(app_token: str | None = None) -> pd.DataFrame:
    """
    Pulls neighborhood polygons and computes acreage from the shape area
    field. The Analysis Neighborhoods dataset returns multipolygon geometry;
    Socrata exposes area via $select with within_circle/area functions is
    inconsistent across datasets, so the safe approach is to pull the GeoJSON
    and compute area locally with shapely/pyproj rather than trust a
    pre-computed column that may not exist.
    """
    params = {"$limit": 100}
    headers = {"X-App-Token": app_token} if app_token else {}

    resp = requests.get(NEIGHBORHOODS_ENDPOINT, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    records = resp.json()

    try:
        import pyproj
        from shapely.geometry import shape
        from shapely.ops import transform
    except ImportError as e:
        raise ImportError(
            "Acreage computation needs shapely + pyproj: "
            "pip install shapely pyproj --break-system-packages"
        ) from e

    # SF is in UTM zone 10N (EPSG:32610) — good for accurate small-area
    # calculations within the city; avoid using unprojected lat/lon degrees.
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True).transform

    rows = []
    for rec in records:
        geom_field = rec.get("the_geom") or rec.get("shape")
        name = rec.get("nhood") or rec.get("neighborhood") or rec.get("name")
        if not geom_field or not name:
            continue
        geom = shape(geom_field)
        geom_m = transform(project, geom)
        acres = geom_m.area / 4046.8564224  # sq meters -> acres
        rows.append({"neighborhood": name, "acres": acres})

    return pd.DataFrame(rows)


def compute_value_per_acre(year: int, app_token: str | None = None) -> pd.DataFrame:
    values = fetch_assessed_value_by_neighborhood(year, app_token)
    acreage = fetch_neighborhood_acreage(app_token)

    merged = values.merge(acreage, on="neighborhood", how="inner")
    merged["value_per_acre"] = merged["total_assessed_value"] / merged["acres"]
    merged = merged.sort_values("value_per_acre", ascending=False).reset_index(drop=True)
    return merged[["neighborhood", "total_assessed_value", "acres", "value_per_acre", "parcel_count"]]


def print_report(df: pd.DataFrame, top_n: int) -> None:
    display = df.head(top_n).copy()
    display["total_assessed_value"] = display["total_assessed_value"].map(lambda v: f"${v:,.0f}")
    display["acres"] = display["acres"].map(lambda v: f"{v:,.1f}")
    display["value_per_acre"] = display["value_per_acre"].map(lambda v: f"${v:,.0f}")
    print(display.to_string(index=False))


def plot_report(df: pd.DataFrame, top_n: int) -> None:
    import matplotlib.pyplot as plt

    display = df.head(top_n).iloc[::-1]  # reverse for horizontal bar chart top-to-bottom
    fig, ax = plt.subplots(figsize=(9, 0.4 * top_n + 1.5))
    ax.barh(display["neighborhood"], display["value_per_acre"], color="#2b6cb0")
    ax.set_xlabel("Assessed value per acre ($)")
    ax.set_title(f"SF Assessed Value per Acre by Neighborhood (top {top_n})")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x/1e6:,.0f}M")
    plt.tight_layout()
    out_path = "/mnt/user-data/outputs/sf_value_per_acre.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="Closed roll fiscal year, e.g. 2023")
    parser.add_argument("--top", type=int, default=15, help="Number of neighborhoods to show")
    parser.add_argument("--plot", action="store_true", help="Save a horizontal bar chart")
    parser.add_argument("--app-token", type=str, default=None, help="Socrata app token (recommended for repeated use)")
    args = parser.parse_args()

    try:
        df = compute_value_per_acre(args.year, args.app_token)
    except requests.HTTPError as e:
        print(f"API request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print_report(df, args.top)
    if args.plot:
        plot_report(df, args.top)


if __name__ == "__main__":
    main()

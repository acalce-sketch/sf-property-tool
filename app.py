import streamlit as st
import pandas as pd
import requests
import re
import math

st.set_page_config(page_title="SF Property Tax & Redevelopment Explorer", layout="wide")
st.title("SF Property Tax & Redevelopment Explorer")
st.caption(
    "Pulls real, live data from the SF Assessor's Historical Secured Property Tax Roll "
    "(DataSF). Shows the Prop 13 assessment gap between parcels and estimates the tax "
    "impact of redeveloping a given parcel."
)

DATASET_URL = "https://data.sfgov.org/resource/wv5m-vpq2.json"

TYPOLOGIES = {
    "Townhomes":          {"units_per_acre": 15,  "avg_unit_sqft": 1400, "value_per_sqft": 1000, "retail_pct": 0.0},
    "5-over-1 podium":    {"units_per_acre": 90,  "avg_unit_sqft": 850,  "value_per_sqft": 950,  "retail_pct": 0.0},
    "High-rise":          {"units_per_acre": 220, "avg_unit_sqft": 900,  "value_per_sqft": 1300, "retail_pct": 0.0},
    "Courtyard mixed-use":{"units_per_acre": 65,  "avg_unit_sqft": 900,  "value_per_sqft": 1050, "retail_pct": 0.15},
}
RETAIL_VALUE_PER_SQFT = 550  # commercial ground-floor space values differently than residential
# value_per_sqft figures are market sale-price assumptions (sales-comparison approach), not construction
# cost \u2014 grounded in SF's spring 2026 citywide condo average of ~$1,170/sqft, with high-rise set above that

ZONING_DESCRIPTIONS = {
    "RH1": "Single-family homes only \u2014 one dwelling unit per lot (though SB 9 can allow up to 4 units via a lot split, regardless of this label).",
    "RH2": "House-scale, up to 2 units per lot.",
    "RH3": "House-scale, up to 3 units per lot.",
    "RM1": "Low-density multi-family residential.",
    "RM2": "Moderate-density multi-family residential.",
    "RM3": "Medium-high density multi-family residential; taller buildings allowed.",
    "RM4": "High-density multi-family residential \u2014 the densest purely-residential district.",
    "RC3": "Combined residential-commercial, medium-high density, ground-floor commercial allowed.",
    "RC4": "Combined residential-commercial, high density, ground-floor commercial allowed.",
    "NC1": "Small neighborhood commercial corridor \u2014 local shops with housing above, low height.",
    "NC2": "Neighborhood commercial corridor \u2014 broader range of shops with housing above.",
    "NC3": "Moderate-scale neighborhood commercial corridor, taller mixed-use buildings allowed.",
    "NCT": "Neighborhood Commercial Transit district \u2014 higher-density mixed-use near transit, reduced/no parking required.",
    "C2": "Community business district \u2014 broader commercial uses, moderate height.",
    "C3": "Downtown commercial core \u2014 high-rise office and residential towers allowed.",
    "M1": "Light industrial \u2014 limited residential use allowed.",
    "M2": "Heavy industrial \u2014 residential use generally restricted.",
    "PDR": "Production, Distribution & Repair \u2014 industrial/maker space, residential generally not allowed.",
    "P": "Public use \u2014 government, institutional, or open space land.",
}

# Approximate base height limits by SF zoning district prefix, in stories (~12 ft/story). This is a
# simplification of SF's actual planning code \u2014 real limits vary by lot, overlay zones, and specific
# height/bulk districts appended to the code. Matched by prefix since real zoning_code values often carry
# suffixes (e.g. "RH1(D)"). Keys deliberately have no hyphens \u2014 confirmed against real Assessor data,
# which stores these as "RH3", "C2", etc., not the hyphenated "RH-3"/"C-2" form used on public zoning maps.
ZONING_BASE_STORIES = {
    "RH1": 3, "RH2": 3, "RH3": 3,
    "RM1": 3, "RM2": 4, "RM3": 6, "RM4": 9,
    "RC3": 6, "RC4": 9,
    "NC1": 3, "NC2": 3, "NC3": 5, "NCT": 6,
    "C3": 20, "C2": 8,
    "M1": 3, "M2": 3, "PDR": 3, "P": 2,
}


def zoning_prefix(zoning_code):
    if not isinstance(zoning_code, str):
        return None
    for key in sorted(ZONING_BASE_STORIES, key=len, reverse=True):
        if zoning_code.upper().startswith(key):
            return key
    return None


ZONING_GROUP_COLORS = {
    "RH (house-scale)": (86, 156, 214),
    "RM (multi-family)": (78, 178, 122),
    "RC (res-commercial)": (201, 162, 39),
    "NC/NCT (commercial corridor)": (201, 100, 59),
    "C-2/C-3 (business/downtown)": (180, 70, 160),
    "Industrial/PDR/Public": (120, 120, 120),
    "Unrecognized": (200, 200, 200),
}


def zoning_group(prefix):
    if prefix is None or pd.isna(prefix):
        return "Unrecognized"
    if prefix.startswith("RH"):
        return "RH (house-scale)"
    if prefix.startswith("RM"):
        return "RM (multi-family)"
    if prefix.startswith("RC"):
        return "RC (res-commercial)"
    if prefix.startswith("NC"):
        return "NC/NCT (commercial corridor)"
    if prefix.startswith("C"):
        return "C-2/C-3 (business/downtown)"
    if prefix in ("M1", "M2", "PDR", "P"):
        return "Industrial/PDR/Public"
    return "Unrecognized"


def render_zoning_grid(source_df, grid_size=10):
    """Static (non-interactive) grid image, colored by the dominant zoning group per cell.
    Uses matplotlib only \u2014 deliberately avoids any map-rendering library given past instability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = source_df.dropna(subset=["lat", "lon", "zoning_code"]).copy()
    if d.empty:
        return None
    d["prefix"] = d["zoning_code"].apply(zoning_prefix)
    d["group"] = d["prefix"].apply(zoning_group)

    lat_min, lat_max = d["lat"].min(), d["lat"].max()
    lon_min, lon_max = d["lon"].min(), d["lon"].max()
    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)
    d["lat_bin"] = np.clip(np.digitize(d["lat"], lat_edges) - 1, 0, grid_size - 1)
    d["lon_bin"] = np.clip(np.digitize(d["lon"], lon_edges) - 1, 0, grid_size - 1)

    grid_rgb = np.full((grid_size, grid_size, 3), 255, dtype=int)
    for (i, j), cell in d.groupby(["lat_bin", "lon_bin"]):
        dominant = cell["group"].value_counts().idxmax()
        grid_rgb[grid_size - 1 - i, j] = ZONING_GROUP_COLORS[dominant]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_rgb.astype("uint8"), extent=[lon_min, lon_max, lat_min, lat_max], aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Dominant zoning group by grid cell (static, not interactive)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in ZONING_GROUP_COLORS.values()]
    ax.legend(handles, list(ZONING_GROUP_COLORS.keys()), loc="upper center",
              bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def check_zoning_legality(zoning_code, stories, units):
    """Approximate legality check against base SF zoning plus current CA state overrides
    (State Density Bonus Law per AB 2345/AB 1287, and SB 9), as of July 2026. Does not check
    CEQA, discretionary review, historic status, or SB 79 transit-overlay zones (no transit
    proximity data available) \u2014 by design, per user request."""
    if not isinstance(zoning_code, str) or not zoning_code:
        return "unknown", "Zoning code not available for this parcel \u2014 no legality check applied.", None
    prefix = None
    for key in sorted(ZONING_BASE_STORIES, key=len, reverse=True):
        if zoning_code.upper().startswith(key):
            prefix = key
            break
    is_single_family_zone = zoning_code.upper().startswith("RH")
    sb9_note = (
        " This zoning is single-family (RH) \u2014 under SB 9, up to 4 units can qualify for ministerial "
        "approval on a lot split regardless of the base 1-unit designation, independent of the height check below."
        if is_single_family_zone else ""
    )
    if prefix is None:
        return "unknown", f"Zoning code '{zoning_code}' not in our simplified lookup \u2014 no height check applied.{sb9_note}", None
    base_max = ZONING_BASE_STORIES[prefix]
    bonus_max = base_max * 2  # rough proxy for CA State Density Bonus Law's stackable up-to-100% bonus
    if stories <= base_max:
        return "ok", f"Within the approximate base zoning height limit for {prefix} (~{base_max} stories).{sb9_note}", base_max
    elif stories <= bonus_max:
        return "bonus", (
            f"Exceeds the approximate base {prefix} limit (~{base_max} stories), but could plausibly be "
            f"achievable under California's State Density Bonus Law (up to a stacked 100% bonus with sufficient "
            f"affordable set-asides, per AB 2345/AB 1287) \u2014 not guaranteed by-right.{sb9_note}"
        ), base_max
    else:
        return "exceeds", (
            f"Exceeds even the approximate density-bonus ceiling for {prefix} (~{bonus_max} stories). Likely needs "
            f"a variance, rezoning, or doesn't have a clear as-of-right legal path under current law.{sb9_note}"
        ), base_max
# to reflect new-tower pricing (181 Fremont, One Steuart Lane) and podium/townhome set nearer entry-level
# new-construction pricing (e.g. RENOU in SoMa). This matches how Prop 13 actually assesses for-sale
# product \u2014 on realized/comparable sale price, not what it cost to build.
EFFECTIVE_TAX_RATE = 0.012  # base 1% plus typical SF voter-approved add-ons


@st.cache_data(ttl=86400)
def fetch_latest_roll_year() -> str:
    params = {"$select": "max(closed_roll_year) as latest"}
    r = requests.get(DATASET_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return str(data[0]["latest"]) if data and data[0].get("latest") else "2024"


LATEST_ROLL_YEAR = fetch_latest_roll_year()


@st.cache_data(ttl=86400)
def fetch_neighborhood_list() -> list:
    params = {
        "$select": "distinct analysis_neighborhood",
        "$where": f"closed_roll_year='{LATEST_ROLL_YEAR}' AND analysis_neighborhood IS NOT NULL",
        "$order": "analysis_neighborhood",
        "$limit": 100,
    }
    r = requests.get(DATASET_URL, params=params, timeout=30)
    r.raise_for_status()
    names = [row["analysis_neighborhood"] for row in r.json() if row.get("analysis_neighborhood")]
    return sorted(names)


@st.cache_data(ttl=3600)
def fetch_parcels(neighborhood: str, limit: int = 500) -> pd.DataFrame:
    params = {
        "$where": f"analysis_neighborhood='{neighborhood}' AND closed_roll_year='{LATEST_ROLL_YEAR}'",
        "$limit": limit,
    }
    r = requests.get(DATASET_URL, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df

    for col in ["lot_area", "assessed_land_value", "assessed_improvement_value", "number_of_units",
                "property_area", "number_of_stories", "year_property_built", "lot_frontage", "lot_depth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def parse_point(geom):
        if not isinstance(geom, dict):
            return None, None
        coords = geom.get("coordinates")
        if not coords:
            return None, None
        return coords[1], coords[0]  # lat, lon

    if "the_geom" in df.columns:
        latlon = df["the_geom"].apply(parse_point)
        df["lat"] = latlon.apply(lambda t: t[0])
        df["lon"] = latlon.apply(lambda t: t[1])

    df = df[df["lot_area"] > 0].copy()
    df["total_assessed"] = df["assessed_land_value"].fillna(0) + df["assessed_improvement_value"].fillna(0)
    df["land_per_sqft"] = df["assessed_land_value"] / df["lot_area"]
    df["current_annual_tax"] = df["total_assessed"] * EFFECTIVE_TAX_RATE
    if "current_sales_date" in df.columns:
        df["current_sales_date"] = pd.to_datetime(df["current_sales_date"], errors="coerce")
    df["improvement_value_per_sqft"] = df["assessed_improvement_value"] / df["property_area"].replace(0, pd.NA)
    return df


with st.sidebar:
    st.header("1. Choose an area")
    st.caption(f"Using the latest available roll: FY{LATEST_ROLL_YEAR}")
    try:
        neighborhood_options = fetch_neighborhood_list()
    except Exception as e:
        st.error(f"Couldn't load neighborhood list: {e}")
        neighborhood_options = ["Russian Hill"]
    default_index = neighborhood_options.index("Russian Hill") if "Russian Hill" in neighborhood_options else 0
    neighborhood = st.selectbox("SF neighborhood", neighborhood_options, index=default_index)
    limit = st.slider("Max parcels to pull", 50, 2000, 500, step=50)
    fetch = st.button("Fetch parcels", type="primary")

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "last_neighborhood" not in st.session_state:
    st.session_state.last_neighborhood = None

neighborhood_changed = neighborhood != st.session_state.last_neighborhood
should_fetch = fetch or neighborhood_changed or st.session_state.df.empty

if should_fetch and neighborhood:
    with st.spinner(f"Pulling live data for {neighborhood} from DataSF..."):
        try:
            st.session_state.df = fetch_parcels(neighborhood, limit)
            st.session_state.last_neighborhood = neighborhood
        except Exception as e:
            st.error(f"Fetch failed: {e}")

df = st.session_state.df

if df.empty:
    st.info("Enter a neighborhood and click Fetch parcels to load real assessor data.")
    st.stop()

st.success(f"Loaded {len(df)} parcels in {neighborhood}.")

col1, col2 = st.columns([2, 1])

if "selected_parcel" not in st.session_state:
    st.session_state.selected_parcel = None

with col1:
    st.subheader("Assessed land value per sqft")
    st.caption("Same zoning, wildly different $/sqft \u2014 that spread is almost entirely Prop 13 turnover timing, not real land value differences. Use the parcel dropdown below to select one.")
    map_df = df.dropna(subset=["lat", "lon"]).copy()
    if not map_df.empty:
        lo, hi = map_df["land_per_sqft"].min(), map_df["land_per_sqft"].max()
        span = max(hi - lo, 1)
        map_df["norm"] = ((map_df["land_per_sqft"] - lo) / span).clip(0, 1).fillna(0)
        map_df["color"] = map_df["norm"].apply(lambda n: [220, int(200 * (1 - n)), 40])
        map_df["size"] = 40
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), color="color", size="size")

    st.subheader("Zoning")
    zoning_fig = render_zoning_grid(df)
    if zoning_fig is not None:
        st.pyplot(zoning_fig)
        st.caption("Static snapshot, not interactive \u2014 each cell shows whichever zoning group has the most parcels in that area of the loaded sample. Not overlaid on the map above (that map is interactive; this grid can't be, given past instability from map-overlay rendering).")
    else:
        st.caption("Not enough located parcels with zoning data to build a grid.")

with col2:
    st.subheader("Peer group stats")
    st.metric("Median $/sqft (this sample)", f"${df['land_per_sqft'].median():,.0f}")
    st.metric("Max $/sqft", f"${df['land_per_sqft'].max():,.0f}")
    st.metric("Min $/sqft", f"${df['land_per_sqft'].min():,.0f}")
    gap = df["land_per_sqft"].max() / max(df["land_per_sqft"].min(), 1)
    st.metric("Widest gap (same neighborhood)", f"{gap:.1f}x")
    with st.expander(f"Zoning mix in {neighborhood} ({len(df)} parcels)"):
        zoning_counts = df["zoning_code"].value_counts().head(15).rename("Parcel count").reset_index()
        zoning_counts.columns = ["Zoning code", "Parcel count"]
        zoning_counts["Meaning"] = zoning_counts["Zoning code"].apply(
            lambda z: ZONING_DESCRIPTIONS.get(zoning_prefix(z), "\u2014")
        )
        st.dataframe(zoning_counts, width="stretch", hide_index=True)

st.divider()
st.subheader("2. Pick a parcel and a redevelopment scenario")

df_display = df[["parcel_number", "use_definition", "zoning_code", "lot_area",
                  "year_property_built", "land_per_sqft", "total_assessed", "current_annual_tax"]].sort_values("land_per_sqft")

parcel_list = list(df_display["parcel_number"])
if st.session_state.selected_parcel in parcel_list:
    default_parcel_index = parcel_list.index(st.session_state.selected_parcel)
else:
    default_parcel_index = 0

parcel_choice = st.selectbox(
    "Parcel (sorted lowest to highest $/sqft \u2014 the cheapest ones are your best redevelopment candidates). "
    "Clicking a dot on the map above updates this automatically.",
    parcel_list,
    index=default_parcel_index,
)
st.session_state.selected_parcel = parcel_choice

st.markdown("**Redevelopment scenario**")
s1, s2 = st.columns(2)
typology_choice = s1.selectbox("Typology (sets base price/retail assumptions)", list(TYPOLOGIES.keys()))
price_tier = s2.selectbox("Price tier", ["Entry-level", "Mid-market", "Luxury"], index=1)
s3, s4, s5 = st.columns(3)
stories = s3.slider("Stories", 1, 40, 6)
lot_coverage = s4.slider("Lot coverage (% of lot the building footprint covers)", 30, 95, 70) / 100
avg_unit_sqft = s5.slider("Avg unit size (sqft)", 400, 2500, 900, step=50)
st.caption("Typology sets the starting price/retail assumptions; the sliders above are fully independent \u2014 adjust any of them to model a specific massing.")

row = df[df.parcel_number == parcel_choice].iloc[0]

st.markdown(f"**Parcel {row.parcel_number}** \u2014 {row.property_location if 'property_location' in row and pd.notna(row.property_location) else 'address not in this sample'}")

info1, info2, info3, info4, info5, info6, info7 = st.columns(7)
info1.metric("Current use", row.use_definition if pd.notna(row.use_definition) else "\u2014")
info2.metric("Zoning", row.zoning_code if pd.notna(row.zoning_code) else "\u2014")
info3.metric("Year built", f"{int(row.year_property_built)}" if pd.notna(row.year_property_built) and row.year_property_built > 0 else "\u2014")
info4.metric("Existing units", f"{int(row.number_of_units)}" if pd.notna(row.number_of_units) else "\u2014")
info5.metric("Building sqft", f"{int(row.property_area):,}" if pd.notna(row.property_area) and row.property_area > 0 else "\u2014")
info6.metric("Last sale", row.current_sales_date[:10] if "current_sales_date" in row and pd.notna(row.current_sales_date) else "No sale on record")
ownership_pct = pd.to_numeric(row.percent_of_ownership, errors="coerce") if "percent_of_ownership" in row else None
info7.metric("Ownership share", f"{ownership_pct*100:.0f}%" if pd.notna(ownership_pct) else "\u2014")
if pd.notna(ownership_pct) and ownership_pct < 1.0:
    st.caption("This parcel is a fractional interest in a larger building (e.g. a condo unit) \u2014 its land value reflects only this share, not the whole property.")

actual_zoning = row.zoning_code if pd.notna(row.zoning_code) else "Unknown"
zoning_options = [f"Current zoning ({actual_zoning})"] + sorted(ZONING_BASE_STORIES.keys())
zoning_scenario = st.selectbox(
    "Zoning to test \u2014 change this to see the legality check against a hypothetical rezoning instead of what's actually on the books",
    zoning_options,
)
test_zoning_code = actual_zoning if zoning_scenario.startswith("Current zoning") else zoning_scenario
if test_zoning_code != actual_zoning:
    st.caption(f"Testing a hypothetical rezoning to {test_zoning_code} \u2014 actual current zoning is {actual_zoning}. Land/building value estimates elsewhere on this page still use real comps for this parcel's actual zoning, not the hypothetical one, since relabeling zoning doesn't change what's actually sold nearby.")

test_prefix = zoning_prefix(test_zoning_code)
if test_prefix and test_prefix in ZONING_DESCRIPTIONS:
    st.caption(f"**{test_prefix}**: {ZONING_DESCRIPTIONS[test_prefix]}")
else:
    st.caption(f"No plain-language description available for '{test_zoning_code}' \u2014 not in the simplified lookup.")

zoning_status, zoning_message, zoning_base_max = check_zoning_legality(test_zoning_code, stories, None)
if zoning_status == "ok":
    st.success(f"\u2705 {zoning_message}")
elif zoning_status == "bonus":
    st.warning(f"\u26a0\ufe0f {zoning_message}")
elif zoning_status == "exceeds":
    st.error(f"\U0001F6AB {zoning_message}")
else:
    st.info(f"\u2139\ufe0f {zoning_message}")
st.caption("Reflects CA's State Density Bonus Law and SB 9 as of July 2026. Does not check SB 79 transit-overlay zones (no transit-proximity data), CEQA, discretionary review, or litigation risk.")

RECENT_CUTOFF = pd.Timestamp.now() - pd.DateOffset(years=5)

same_zone = df[df.zoning_code == row.zoning_code]
if "current_sales_date" in same_zone.columns:
    recent_same_zone = same_zone[same_zone["current_sales_date"] >= RECENT_CUTOFF]
else:
    recent_same_zone = pd.DataFrame()
land_recency_filtered = len(recent_same_zone) >= 3
land_comp_pool = recent_same_zone if land_recency_filtered else same_zone
market_rate = land_comp_pool["land_per_sqft"].quantile(0.75)  # top-quartile as a market-rate proxy
market_rate = market_rate if pd.notna(market_rate) and market_rate > 0 else df["land_per_sqft"].median()

if "current_sales_date" in df.columns:
    recent_building_comps = df[
        (df["current_sales_date"] >= RECENT_CUTOFF)
        & (df["property_area"] > 0)
        & (df["improvement_value_per_sqft"] > 0)
    ]
else:
    recent_building_comps = pd.DataFrame()
building_value_from_comps = len(recent_building_comps) >= 3
if building_value_from_comps:
    observed_value_per_sqft = recent_building_comps["improvement_value_per_sqft"].median()

PRICE_TIER_MULTIPLIER = {"Entry-level": 0.80, "Mid-market": 1.00, "Luxury": 1.40}


def construction_cost_per_sqft(n_stories: int) -> int:
    # step function reflecting real code-driven cost jumps: elevator/fire pump at 4 stories,
    # elevator speed + structural upgrades at 8, steel/concrete high-rise premium at 15+.
    # Base rebased to SF-specific 2026 figures: RSMeans regional cost index for SF runs
    # ~1.35-1.42x the national average, and multifamily-specific benchmarks for SF put hard
    # construction costs at $450+/sqft \u2014 well above a generic national base. Elevated Bay Area
    # labor rates are partly attributed to competition from concurrent AI-driven commercial/office
    # construction for the same skilled trades, per 2026 Bay Area builder cost data.
    base = 450
    if n_stories >= 4:
        base += 65
    if n_stories >= 8:
        base += 105
    if n_stories >= 15:
        base += 130
    return base


t = TYPOLOGIES[typology_choice]
acres = row.lot_area / 43560
efficiency = 0.85  # circulation/common-area loss

building_footprint = row.lot_area * lot_coverage
retail_sqft = building_footprint * t["retail_pct"] * efficiency  # ground floor only
total_building_sqft = (building_footprint * stories * efficiency) 
residential_sqft = max(total_building_sqft - retail_sqft, 0)
units = round(residential_sqft / avg_unit_sqft) if avg_unit_sqft > 0 else 0

base_value_per_sqft = observed_value_per_sqft if building_value_from_comps else t["value_per_sqft"]
value_per_sqft_adj = base_value_per_sqft * PRICE_TIER_MULTIPLIER[price_tier]
improvement_value = (residential_sqft * value_per_sqft_adj) + (retail_sqft * RETAIL_VALUE_PER_SQFT)

st.caption(
    (f"Land rate: top-quartile of {len(land_comp_pool)} same-zoning comps"
     + (", sold in the last 5 years. " if land_recency_filtered else " (no recent sales in this zone, using full sample). "))
    + (f"Building value: ${observed_value_per_sqft:,.0f}/sqft, median of {len(recent_building_comps)} actual local sales in the last 5 years."
       if building_value_from_comps else
       f"Building value: citywide typology assumption (${t['value_per_sqft']}/sqft) \u2014 fewer than 3 recent local sales found to derive a real comp.")
)

cost_per_sqft = construction_cost_per_sqft(stories)
construction_cost = (residential_sqft + retail_sqft) * cost_per_sqft
est_margin = improvement_value - construction_cost

new_land_value = row.lot_area * market_rate
new_assessed = new_land_value + improvement_value
new_tax = new_assessed * EFFECTIVE_TAX_RATE
current_tax = row.current_annual_tax

st.markdown(f"#### Today \u2192 Proposed: {typology_choice}")
before_col, after_col = st.columns(2)

with before_col:
    st.markdown("**Today**")
    b1, b2 = st.columns(2)
    b1.metric("Units", f"{int(row.number_of_units)}" if pd.notna(row.number_of_units) else "\u2014")
    b2.metric("Building sqft", f"{int(row.property_area):,}" if pd.notna(row.property_area) and row.property_area > 0 else "\u2014")
    b3, b4 = st.columns(2)
    b3.metric("Assessed value", f"${row.total_assessed:,.0f}")
    b4.metric("Annual tax", f"${current_tax:,.0f}")

with after_col:
    st.markdown(f"**Proposed ({typology_choice}, {stories} stories, {price_tier})**")
    a1, a2 = st.columns(2)
    a1.metric("Units", f"{units}", f"{units - (int(row.number_of_units) if pd.notna(row.number_of_units) else 0):+d}")
    a2.metric("Building sqft", f"{int(residential_sqft + retail_sqft):,}")
    a3, a4 = st.columns(2)
    a3.metric("Assessed value", f"${new_assessed:,.0f}", f"+${new_assessed - row.total_assessed:,.0f}")
    a4.metric("Annual tax", f"${new_tax:,.0f}", f"+${new_tax - current_tax:,.0f}")

st.metric("Annual tax revenue delta", f"+${new_tax - current_tax:,.0f}/yr", f"{(new_tax / max(current_tax,1)):.1f}x current")

st.markdown("**Feasibility check \u2014 does the market value clear the cost of building it?**")
f1, f2, f3, f4 = st.columns(4)
f1.metric("Construction cost/sqft", f"${cost_per_sqft}", help="Rises with stories \u2014 elevator/fire pump at 4 stories, elevator speed and structural upgrades at 8, steel/concrete premium at 15+.")
f2.metric("Total construction cost", f"${construction_cost:,.0f}")
f3.metric("Market value of building", f"${improvement_value:,.0f}")
f4.metric("Estimated margin", f"${est_margin:,.0f}", delta_color="normal" if est_margin >= 0 else "inverse")
if est_margin < 0:
    st.warning("At this stories/price-tier combination, estimated construction cost exceeds market value \u2014 this scenario likely wouldn't get built, whatever the tax revenue looks like.")

st.caption(
    "Land rate and building value basis are shown above, next to the redevelopment scenario controls. "
    "When fewer than 3 real local comps are available, figures fall back to citywide planning-level "
    "assumptions, noted explicitly where that happens. Construction cost is a separate, independent "
    "estimate used only for the feasibility check above, not for the tax calculation. These are all "
    "decision-support estimates, not a substitute for an actual appraisal or pro forma."
)

with st.expander("See all parcels in this sample"):
    st.dataframe(df_display, width="stretch")

st.divider()
st.subheader("3. Citywide density & fiscal productivity")
st.caption(
    "Total assessed value per acre by neighborhood, citywide \u2014 the Urban3/Minicozzi "
    "metric for how much tax base a neighborhood generates per unit of land, independent "
    "of parcel count or population. Denominator is summed parcel lot area from this same "
    "assessor dataset (not external GIS boundaries), so it reflects developed footprint "
    "rather than total neighborhood acreage including streets/parks."
)


@st.cache_data(ttl=86400)
def fetch_citywide_density() -> pd.DataFrame:
    params = {
        "$select": (
            "analysis_neighborhood as neighborhood, "
            "sum(assessed_land_value + assessed_improvement_value) as total_value, "
            "sum(lot_area) as total_lot_area_sqft, "
            "count(*) as parcel_count"
        ),
        "$where": f"closed_roll_year='{LATEST_ROLL_YEAR}' AND analysis_neighborhood IS NOT NULL AND lot_area > 0",
        "$group": "neighborhood",
        "$limit": 100,
    }
    r = requests.get(DATASET_URL, params=params, timeout=30)
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    if d.empty:
        return d
    for col in ["total_value", "total_lot_area_sqft", "parcel_count"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[d["total_lot_area_sqft"] > 0].copy()
    d["acres"] = d["total_lot_area_sqft"] / 43560
    d["value_per_acre"] = d["total_value"] / d["acres"]
    return d.sort_values("value_per_acre", ascending=False).reset_index(drop=True)


try:
    density_df = fetch_citywide_density()
except Exception as e:
    density_df = pd.DataFrame()
    st.error(f"Couldn't load citywide density data: {e}")

if not density_df.empty:
    top_n = st.slider("Show top N neighborhoods", 5, min(41, len(density_df)), 15)
    st.bar_chart(density_df.head(top_n).set_index("neighborhood")["value_per_acre"])

    if neighborhood in density_df["neighborhood"].values:
        rank_pos = density_df[density_df["neighborhood"] == neighborhood].index[0] + 1
        st.caption(f"**{neighborhood}** ranks #{rank_pos} of {len(density_df)} SF neighborhoods by assessed value per acre.")

    with st.expander("See full citywide table"):
        display_density = density_df.copy()
        display_density["total_value"] = display_density["total_value"].map(lambda v: f"${v:,.0f}")
        display_density["acres"] = display_density["acres"].map(lambda v: f"{v:,.1f}")
        display_density["value_per_acre"] = display_density["value_per_acre"].map(lambda v: f"${v:,.0f}")
        st.dataframe(
            display_density[["neighborhood", "total_value", "acres", "value_per_acre", "parcel_count"]],
            width="stretch", hide_index=True,
        )
    st.caption(
        "This is a fiscal-productivity ranking, not a causal claim \u2014 high value/acre reflects some mix of "
        "density, location premium, and building quality. Use alongside the per-parcel assessment gap above "
        "to see whether a neighborhood's low-value parcels are dragging down its overall productivity."
    )

st.divider()
st.subheader("4. 3D value-per-acre skyline")
st.caption(
    "Each bar is one parcel in the currently loaded sample; height = that parcel's own assessed "
    "value per acre (land + improvements), not the neighborhood average. This is the Urban3-style "
    "visualization \u2014 dense, high-value parcels read as skyscrapers of tax revenue, low-value or "
    "underbuilt parcels read as flat ground."
)


def render_value_per_acre_3d(source_df, neighborhood_label):
    """Static (non-interactive) 3D bar chart, matplotlib only \u2014 deliberately avoids any
    interactive map-rendering library given past instability, matching render_zoning_grid above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 \u2014 registers the 3d projection
    import numpy as np

    d = source_df.dropna(subset=["lat", "lon"]).copy()
    d = d[d["lot_area"] > 0]
    if d.empty:
        return None

    d["value_per_acre"] = d["total_assessed"] / (d["lot_area"] / 43560)
    # Cap at the 98th percentile so a handful of outliers (e.g. downtown towers) don't
    # flatten every other bar into invisibility on the same z-scale.
    cap = d["value_per_acre"].quantile(0.98)
    d["height_capped"] = d["value_per_acre"].clip(upper=cap)

    lon_span = max(d["lon"].max() - d["lon"].min(), 1e-6)
    lat_span = max(d["lat"].max() - d["lat"].min(), 1e-6)
    bar_w = lon_span / 60
    bar_d = lat_span / 60

    norm = plt.Normalize(d["height_capped"].min(), d["height_capped"].max())
    colors = plt.cm.inferno(norm(d["height_capped"]))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(d["lon"], d["lat"], np.zeros(len(d)), bar_w, bar_d, d["height_capped"],
              color=colors, shade=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Assessed value per acre ($)")
    ax.set_title(f"Value per acre \u2014 {neighborhood_label} (top 2% capped for scale)", fontsize=11)
    ax.view_init(elev=35, azim=-60)
    fig.tight_layout()
    return fig


value_3d_fig = render_value_per_acre_3d(df, neighborhood)
if value_3d_fig is not None:
    st.pyplot(value_3d_fig)
    st.caption(
        "Static image, not interactive/rotatable \u2014 same rendering approach as the zoning grid "
        "above. Bar footprint size is a fixed visual unit for legibility, not the parcel's actual "
        "lot dimensions."
    )
else:
    st.caption("Not enough located parcels with valid lot area to build a 3D chart.")

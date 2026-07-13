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

# Approximate base height limits by SF zoning district prefix, in stories (~12 ft/story). This is a
# simplification of SF's actual planning code \u2014 real limits vary by lot, overlay zones, and specific
# height/bulk districts appended to the code. Matched by prefix since real zoning_code values often carry
# suffixes (e.g. "RH-1(D)").
ZONING_BASE_STORIES = {
    "RH-1": 3, "RH-2": 3, "RH-3": 3,
    "RM-1": 3, "RM-2": 4, "RM-3": 6, "RM-4": 9,
    "RC-3": 6, "RC-4": 9,
    "NC-1": 3, "NC-2": 3, "NC-3": 5, "NCT": 6,
    "C-3": 20, "C-2": 8,
    "M-1": 3, "M-2": 3, "PDR": 3, "P": 2,
}


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

with col2:
    st.subheader("Peer group stats")
    st.metric("Median $/sqft (this sample)", f"${df['land_per_sqft'].median():,.0f}")
    st.metric("Max $/sqft", f"${df['land_per_sqft'].max():,.0f}")
    st.metric("Min $/sqft", f"${df['land_per_sqft'].min():,.0f}")
    gap = df["land_per_sqft"].max() / max(df["land_per_sqft"].min(), 1)
    st.metric("Widest gap (same neighborhood)", f"{gap:.1f}x")
    with st.expander(f"Zoning mix in {neighborhood} ({len(df)} parcels)"):
        zoning_counts = df["zoning_code"].value_counts().head(15)
        st.dataframe(zoning_counts.rename("Parcel count"), width="stretch")

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
    # elevator speed + structural upgrades at 8, steel/concrete high-rise premium at 15+
    base = 350
    if n_stories >= 4:
        base += 50
    if n_stories >= 8:
        base += 80
    if n_stories >= 15:
        base += 100
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

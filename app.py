import streamlit as st
import pandas as pd
import requests
import re
import pydeck as pdk

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
# to reflect new-tower pricing (181 Fremont, One Steuart Lane) and podium/townhome set nearer entry-level
# new-construction pricing (e.g. RENOU in SoMa). This matches how Prop 13 actually assesses for-sale
# product \u2014 on realized/comparable sale price, not what it cost to build.
EFFECTIVE_TAX_RATE = 0.012  # base 1% plus typical SF voter-approved add-ons


@st.cache_data(ttl=86400)
def fetch_neighborhood_list() -> list:
    params = {
        "$select": "distinct analysis_neighborhood",
        "$where": "closed_roll_year='2024' AND analysis_neighborhood IS NOT NULL",
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
        "$where": f"analysis_neighborhood='{neighborhood}' AND closed_roll_year='2024'",
        "$limit": limit,
    }
    r = requests.get(DATASET_URL, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df

    for col in ["lot_area", "assessed_land_value", "assessed_improvement_value", "number_of_units",
                "property_area", "number_of_stories", "year_property_built"]:
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
    return df


with st.sidebar:
    st.header("1. Choose an area")
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
    st.subheader("Assessed land value per sqft \u2014 click a dot to select that parcel")
    st.caption("Same zoning, wildly different $/sqft \u2014 that spread is almost entirely Prop 13 turnover timing, not real land value differences.")
    show_3d = st.toggle("Show building heights in 3D", value=True, help="Extrudes each parcel by its actual number of stories. Turn off to fall back to the flat 2D view.")
    map_df = df.dropna(subset=["lat", "lon"]).copy()
    if not map_df.empty:
        lo, hi = map_df["land_per_sqft"].min(), map_df["land_per_sqft"].max()
        span = max(hi - lo, 1)
        map_df["norm"] = ((map_df["land_per_sqft"] - lo) / span).clip(0, 1).fillna(0)
        map_df["r"] = 220
        map_df["g"] = (200 * (1 - map_df["norm"])).fillna(100).astype(int)
        map_df["b"] = 40
        map_df["lat"] = map_df["lat"].astype(float)
        map_df["lon"] = map_df["lon"].astype(float)
        map_df["land_per_sqft"] = map_df["land_per_sqft"].fillna(0).round(0).astype(int)
        map_df["parcel_number"] = map_df["parcel_number"].astype(str)
        map_df["stories"] = pd.to_numeric(map_df.get("number_of_stories"), errors="coerce").fillna(1).clip(lower=1)
        map_df["elevation"] = (map_df["stories"] * 12).astype(float)  # ~12 ft per floor
        map_df["stories"] = map_df["stories"].astype(int)
        map_df = map_df[["lat", "lon", "r", "g", "b", "land_per_sqft", "parcel_number", "stories", "elevation"]].reset_index(drop=True)

        if show_3d:
            layer = pdk.Layer(
                "ColumnLayer",
                id="parcels",
                data=map_df,
                get_position=["lon", "lat"],
                get_elevation="elevation",
                elevation_scale=1,
                radius=15,
                get_fill_color=["r", "g", "b", 180],
                pickable=True,
                auto_highlight=True,
                extruded=True,
            )
            pitch = 45
        else:
            layer = pdk.Layer(
                "ScatterplotLayer",
                id="parcels",
                data=map_df,
                get_position=["lon", "lat"],
                get_fill_color=["r", "g", "b", 180],
                get_radius=18,
                pickable=True,
                auto_highlight=True,
            )
            pitch = 0
        view_state = pdk.ViewState(
            latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=15, pitch=pitch
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=None,
            tooltip={"text": "Parcel {parcel_number}\n${land_per_sqft} /sqft land value\n{stories} stories today"},
        )
        event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key="parcel_map")

        clicked = None
        if event and event.selection and event.selection.get("objects", {}).get("parcels"):
            clicked = event.selection["objects"]["parcels"][0].get("parcel_number")
        if clicked:
            st.session_state.selected_parcel = clicked

with col2:
    st.subheader("Peer group stats")
    st.metric("Median $/sqft (this sample)", f"${df['land_per_sqft'].median():,.0f}")
    st.metric("Max $/sqft", f"${df['land_per_sqft'].max():,.0f}")
    st.metric("Min $/sqft", f"${df['land_per_sqft'].min():,.0f}")
    gap = df["land_per_sqft"].max() / max(df["land_per_sqft"].min(), 1)
    st.metric("Widest gap (same neighborhood)", f"{gap:.1f}x")

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

same_zone = df[df.zoning_code == row.zoning_code]
market_rate = same_zone["land_per_sqft"].quantile(0.75)  # top-quartile as a market-rate proxy
market_rate = market_rate if pd.notna(market_rate) and market_rate > 0 else df["land_per_sqft"].median()

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

value_per_sqft_adj = t["value_per_sqft"] * PRICE_TIER_MULTIPLIER[price_tier]
improvement_value = (residential_sqft * value_per_sqft_adj) + (retail_sqft * RETAIL_VALUE_PER_SQFT)

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
    "Market land rate is estimated as the 75th percentile $/sqft among same-zoning parcels in this "
    "sample. Building value uses per-sqft sale price assumptions grounded in SF's 2026 condo market "
    "(citywide average ~$1,170/sqft), scaled by the price tier selected above \u2014 this matches how Prop "
    "13 actually assesses for-sale product. Construction cost is a separate, independent estimate used "
    "only for the feasibility check, not for the tax calculation. These are all planning-level estimates, "
    "not a substitute for an actual appraisal or pro forma."
)

with st.expander("See all parcels in this sample"):
    st.dataframe(df_display, width="stretch")

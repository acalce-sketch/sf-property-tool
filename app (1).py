import streamlit as st
import pandas as pd
import requests
import re

st.set_page_config(page_title="SF Property Tax & Redevelopment Explorer", layout="wide")
st.title("SF Property Tax & Redevelopment Explorer")
st.caption(
    "Pulls real, live data from the SF Assessor's Historical Secured Property Tax Roll "
    "(DataSF). Shows the Prop 13 assessment gap between parcels and estimates the tax "
    "impact of redeveloping a given parcel."
)

DATASET_URL = "https://data.sfgov.org/resource/wv5m-vpq2.json"

TYPOLOGIES = {
    "Townhomes":          {"units_per_acre": 15,  "avg_unit_sqft": 1400, "cost_per_sqft": 320, "retail_pct": 0.0},
    "5-over-1 podium":    {"units_per_acre": 90,  "avg_unit_sqft": 850,  "cost_per_sqft": 290, "retail_pct": 0.0},
    "High-rise":          {"units_per_acre": 220, "avg_unit_sqft": 900,  "cost_per_sqft": 420, "retail_pct": 0.0},
    "Courtyard mixed-use":{"units_per_acre": 65,  "avg_unit_sqft": 900,  "cost_per_sqft": 300, "retail_pct": 0.15},
}
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

    for col in ["lot_area", "assessed_land_value", "assessed_improvement_value", "number_of_units"]:
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

if fetch or (neighborhood and st.session_state.df.empty):
    with st.spinner(f"Pulling live data for {neighborhood} from DataSF..."):
        try:
            st.session_state.df = fetch_parcels(neighborhood, limit)
        except Exception as e:
            st.error(f"Fetch failed: {e}")

df = st.session_state.df

if df.empty:
    st.info("Enter a neighborhood and click Fetch parcels to load real assessor data.")
    st.stop()

st.success(f"Loaded {len(df)} parcels in {neighborhood}.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Assessed land value per sqft")
    st.caption("Same zoning, wildly different $/sqft \u2014 that spread is almost entirely Prop 13 turnover timing, not real land value differences.")
    map_df = df.dropna(subset=["lat", "lon"])
    if not map_df.empty:
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=40)

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

parcel_choice = st.selectbox(
    "Parcel (sorted lowest to highest $/sqft \u2014 the cheapest ones are your best redevelopment candidates)",
    df_display["parcel_number"],
)
typology_choice = st.selectbox("Redevelopment typology", list(TYPOLOGIES.keys()))

row = df[df.parcel_number == parcel_choice].iloc[0]
same_zone = df[df.zoning_code == row.zoning_code]
market_rate = same_zone["land_per_sqft"].quantile(0.75)  # top-quartile as a market-rate proxy
market_rate = market_rate if pd.notna(market_rate) and market_rate > 0 else df["land_per_sqft"].median()

t = TYPOLOGIES[typology_choice]
acres = row.lot_area / 43560
units = round(t["units_per_acre"] * acres)
res_sqft = t["units_per_acre"] * acres * t["avg_unit_sqft"]
retail_sqft = res_sqft * t["retail_pct"]
improvement_value = (res_sqft + retail_sqft) * t["cost_per_sqft"]
new_land_value = row.lot_area * market_rate
new_assessed = new_land_value + improvement_value
new_tax = new_assessed * EFFECTIVE_TAX_RATE
current_tax = row.current_annual_tax

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current annual tax", f"${current_tax:,.0f}")
c2.metric("Projected new assessed value", f"${new_assessed:,.0f}")
c3.metric("Projected new annual tax", f"${new_tax:,.0f}")
c4.metric("Annual delta", f"+${new_tax - current_tax:,.0f}", f"{(new_tax / max(current_tax,1)):.1f}x")

st.caption(
    "Market land rate is estimated as the 75th percentile $/sqft among same-zoning parcels in this "
    "sample \u2014 a rough stand-in for 'recently reassessed' comps. Construction cost assumptions are "
    "planning-level, not a substitute for an actual pro forma. This is a decision-support estimate, not "
    "an appraisal."
)

with st.expander("See all parcels in this sample"):
    st.dataframe(df_display, use_container_width=True)

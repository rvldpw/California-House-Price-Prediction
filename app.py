import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, MousePosition
from streamlit_folium import st_folium

st.set_page_config(
    page_title="CA Price Estimator",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"     # sidebar open by default
)

# Custom CSS (same as original, kept for reference)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: #f9f9f8;
  color: #1a1a1a;
}
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid #ebebea;
}
[data-testid="stSidebar"] > div { padding: 1.75rem 1.5rem; }
[data-testid="stSidebar"] * { color: #1a1a1a !important; }
[data-testid="stSidebar"] label {
  font-size: 0.68rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: #aaa !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
  background: #f5f5f3 !important;
  border: 1px solid #e8e8e6 !important;
  border-radius: 7px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] div {
  font-size: 0.85rem !important;
  color: #1a1a1a !important;
}
[data-testid="stSidebar"] .stButton > button {
  background: #1a1a1a !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 7px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  padding: 0.6rem 1rem !important;
  width: 100% !important;
  transition: opacity 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.65 !important; }
[data-testid="stSidebar"] hr { border-color: #ebebea !important; margin: 1rem 0 !important; }

.main .block-container { padding: 2.25rem 2.75rem; max-width: 1080px; }

iframe { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Load model and data (cached)
@st.cache_resource
def load_model():
    with open("ca_house_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data_ca_geocoded.csv")
    df["county"] = df["county"].fillna("Unknown")
    df["city"]   = df["city"].fillna("Unknown")
    return df

model = load_model()
df    = load_data()

# Helper functions (unchanged)
def engineer_features(row):
    d = row.copy()
    d["rooms_per_household"]      = d["total_rooms"]    / max(d["households"], 1)
    d["bedrooms_per_room"]        = d["total_bedrooms"] / max(d["total_rooms"], 1)
    d["population_per_household"] = d["population"]     / max(d["households"], 1)
    d["income_per_room"]          = d["median_income"]  / (d["total_rooms"] + 1)
    d["is_island"]  = 1 if d["ocean_proximity"] == "ISLAND" else 0
    d["is_capped"]  = 0
    return pd.DataFrame([d])

def predict(row):
    X   = engineer_features(row)
    lp  = model.predict(X)[0]
    p   = np.expm1(lp)
    return p, p * 0.88, p * 1.12

def fmt(p):
    if p >= 1_000_000: return f"${p/1_000_000:.2f}M"
    return f"${p/1_000:.0f}K"

def fmt_full(p): return f"${p:,.0f}"

county_list = sorted(df["county"].unique())
ocean_map = {
    "<1H OCEAN": "< 1 Hour from Ocean",
    "INLAND":    "Inland",
    "NEAR BAY":  "Near Bay",
    "NEAR OCEAN":"Near Ocean",
    "ISLAND":    "Island"
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**⬡ HomeValueIQ**")
    st.caption("California House Price Estimator")
    st.markdown("---")

    with st.expander("📍 Location", expanded=True):
        selected_county = st.selectbox(
            "County", county_list,
            index=county_list.index("Los Angeles County") if "Los Angeles County" in county_list else 0,
            label_visibility="collapsed"
        )

        city_list = sorted(df[df["county"] == selected_county]["city"].unique())
        selected_city = st.selectbox("City", city_list, label_visibility="collapsed")

    city_rows = df[(df["county"] == selected_county) & (df["city"] == selected_city)]
    city_lat  = float(city_rows["latitude"].mean())
    city_lon  = float(city_rows["longitude"].mean())

    with st.expander("🌊 Ocean Proximity"):
        ocean_sel = st.selectbox(
            "Ocean", list(ocean_map.keys()),
            format_func=lambda x: ocean_map[x],
            label_visibility="collapsed"
        )

    with st.expander("🏠 Property"):
        housing_age    = st.slider("House Age (yrs)", 1, 52, 20)
        total_rooms    = st.slider("Total Rooms", 2, 20, 8)
        total_bedrooms = st.slider("Bedrooms", 1, 10, 3)
        total_bedrooms = min(total_bedrooms, total_rooms)

    with st.expander("📊 Demographics"):
        households    = st.slider("Households", 10, 400, 100)
        population    = st.slider("Population", 30, 2000, 400)
        median_income = st.slider("Income (×$10K)", 0.5, 15.0, 4.0, 0.1)

    st.button("Estimate Price", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION (scale per‑house rooms to block level)
# ─────────────────────────────────────────────────────────────────────────────
block_total_rooms    = total_rooms    * households
block_total_bedrooms = total_bedrooms * households

input_row = dict(
    longitude=city_lon, latitude=city_lat,
    housing_median_age=housing_age,
    total_rooms=block_total_rooms,
    total_bedrooms=block_total_bedrooms,
    population=population, households=households,
    median_income=median_income, ocean_proximity=ocean_sel
)
pred, low, high = predict(input_row)

# County & city stats for comparison
county_data = df[df["county"] == selected_county]
city_data   = df[(df["county"] == selected_county) & (df["city"] == selected_city)]
county_med  = county_data["median_house_value"].median()
city_med    = city_data["median_house_value"].median() if len(city_data) > 0 else county_med
vs_pct      = (pred - county_med) / county_med * 100

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT (left = data, right = map)
# ─────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([6, 1])
with c1:
    st.markdown(
        "<p style='font-size:0.65rem;font-weight:600;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#bbb;margin-bottom:1.5rem;'>"
        "California · House Price Estimator</p>",
        unsafe_allow_html=True
    )

left, right = st.columns([5, 4], gap="large")

# ─── LEFT COLUMN (statistics, drivers) ───────────────────────────────────────
with left:
    arrow  = "↑" if vs_pct > 0 else ("↓" if vs_pct < 0 else "≈")
    a_col  = "#c0392b" if vs_pct > 0 else ("#27ae60" if vs_pct < 0 else "#aaa")
    vs_str = f"{arrow} {abs(vs_pct):.1f}% vs county median"

    st.markdown(
        "<p style='font-size:0.65rem;font-weight:600;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#bbb;margin-bottom:0.3rem;'>Estimated Value</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-family:\"DM Mono\",monospace;font-size:3.6rem;"
        f"font-weight:500;letter-spacing:-0.02em;color:#1a1a1a;"
        f"line-height:1;margin:0;'>{fmt_full(pred)}</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-family:\"DM Mono\",monospace;font-size:0.75rem;"
        f"color:#bbb;margin-top:0.35rem;margin-bottom:1.5rem;'>"
        f"{fmt(low)} &nbsp;—&nbsp; {fmt(high)} &nbsp;·&nbsp; "
        f"<span style='color:{a_col};'>{vs_str}</span></p>",
        unsafe_allow_html=True
    )

    # Chips
    chips_html = (
        f"<div style='display:flex;flex-wrap:wrap;gap:0.35rem;margin-bottom:1.5rem;'>"
        f"<span style='font-size:0.65rem;font-weight:600;letter-spacing:0.06em;"
        f"background:#1a1a1a;color:#f9f9f8;padding:0.18rem 0.55rem;"
        f"border-radius:3px;'>{selected_city}</span>"
    )
    for label in [ocean_map[ocean_sel], f"{housing_age}yr", f"{total_rooms} rooms",
                  f"{total_bedrooms}bd", fmt_full(median_income*10_000)]:
        chips_html += (
            f"<span style='font-size:0.65rem;font-weight:500;letter-spacing:0.04em;"
            f"background:#ebebea;color:#555;padding:0.18rem 0.55rem;"
            f"border-radius:3px;'>{label}</span>"
        )
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    # Area statistics
    st.markdown(
        "<p style='font-size:0.65rem;font-weight:600;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#bbb;margin-bottom:0;'>Area Statistics</p>",
        unsafe_allow_html=True
    )
    stats = [
        ("County median",     fmt_full(county_med)),
        ("City median",       fmt_full(city_med)),
        ("County range",      f"{fmt(county_data['median_house_value'].min())} – {fmt(county_data['median_house_value'].max())}"),
        ("Rooms / household", f"{total_rooms:.0f}"),
        ("People / household",f"{population/max(households,1):.1f}"),
        ("Block groups (city)",f"{len(city_data):,}"),
    ]
    for name, val in stats:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:baseline;padding:0.65rem 0;"
            f"border-top:1px solid #ebebea;'>"
            f"<span style='font-size:0.78rem;color:#888;'>{name}</span>"
            f"<span style='font-family:\"DM Mono\",monospace;font-size:0.82rem;"
            f"color:#1a1a1a;font-weight:500;'>{val}</span></div>",
            unsafe_allow_html=True
        )
    st.markdown("<div style='border-top:1px solid #ebebea;'></div>", unsafe_allow_html=True)

    # Key drivers
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.65rem;font-weight:600;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#bbb;margin-bottom:0;'>Key Drivers</p>",
        unsafe_allow_html=True
    )
    drivers = []
    if median_income >= 7:
        drivers.append(("High income area", "+strong", "#27ae60"))
    elif median_income >= 4.5:
        drivers.append(("Above-avg income", "+moderate", "#27ae60"))
    elif median_income <= 2:
        drivers.append(("Low income area", "−suppressing", "#c0392b"))
    else:
        drivers.append(("Average income", "neutral", "#aaa"))

    if ocean_sel in ("NEAR OCEAN", "NEAR BAY"):
        drivers.append(("Water proximity", "+premium", "#27ae60"))
    elif ocean_sel == "ISLAND":
        drivers.append(("Island location", "+rare", "#27ae60"))
    elif ocean_sel == "INLAND":
        drivers.append(("Inland location", "−discount", "#c0392b"))
    else:
        drivers.append(("< 1hr ocean", "+slight", "#888"))

    if housing_age <= 12:
        drivers.append(("New construction", "+premium", "#27ae60"))
    elif housing_age >= 40:
        drivers.append(("Older housing", "−discount", "#c0392b"))
    else:
        drivers.append(("Mid-age stock", "neutral", "#aaa"))

    if total_rooms >= 6:
        drivers.append(("Spacious home", "+premium", "#27ae60"))
    elif total_rooms <= 3:
        drivers.append(("Compact home", "−slight", "#c0392b"))
    else:
        drivers.append(("Average size", "neutral", "#aaa"))

    for dname, deff, dcol in drivers:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:baseline;padding:0.6rem 0;border-bottom:1px solid #f2f2f0;'>"
            f"<span style='font-size:0.8rem;color:#1a1a1a;'>{dname}</span>"
            f"<span style='font-family:\"DM Mono\",monospace;font-size:0.72rem;"
            f"color:{dcol};'>{deff}</span></div>",
            unsafe_allow_html=True
        )

# ─── RIGHT COLUMN (map) ──────────────────────────────────────────────────────
with right:
    st.markdown(
        "<p style='font-size:0.65rem;font-weight:600;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#bbb;margin-bottom:0.5rem;'>Location</p>",
        unsafe_allow_html=True
    )

    # Build map only for the selected county (much faster)
    @st.cache_data
    def build_map_for_county(county_name, lat, lon):
        county_df = df[df["county"] == county_name].copy()
        if county_df.empty:
            # fallback
            m = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB positron")
            return m
        m = folium.Map(
            location=[lat, lon],
            zoom_start=10,               # narrower view (makes map smaller in practice)
            tiles="CartoDB positron",
            zoom_control=True,
            control_scale=True
        )
        Fullscreen().add_to(m)
        MiniMap(toggle_display=True).add_to(m)
        MousePosition(position="bottomright", separator=" | ", prefix="Coordinates:").add_to(m)

        # Marker cluster for all properties in the county
        marker_cluster = MarkerCluster().add_to(m)

        for row in county_df.itertuples(index=False):
            price = row.median_house_value
            if price < 150_000:
                color = "green"
            elif price < 300_000:
                color = "orange"
            else:
                color = "red"
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Price: ${price:,.0f}"
            ).add_to(marker_cluster)

        # Highlight the selected city
        folium.CircleMarker(
            [lat, lon], radius=14,
            color="#1a1a1a", fill=True, fill_color="#1a1a1a",
            fill_opacity=0.1, weight=0
        ).add_to(m)
        folium.CircleMarker(
            [lat, lon], radius=6,
            color="#1a1a1a", fill=True, fill_color="#1a1a1a",
            fill_opacity=1, weight=0,
            tooltip=f"{selected_city} · {fmt_full(pred)}"
        ).add_to(m)
        return m

    # Generate and display the map (smaller height)
    m = build_map_for_county(selected_county, city_lat, city_lon)
    st_folium(m, height=280, use_container_width=True)   # reduced height

    # County comparison chart (unchanged, still useful)
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.65rem;font-weight:600;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#bbb;margin-bottom:0.75rem;'>County Comparison</p>",
        unsafe_allow_html=True
    )
    top_co = (
        df.groupby("county")["median_house_value"]
        .median().sort_values(ascending=False).head(10).reset_index()
    )
    max_v = top_co["median_house_value"].max()
    for _, row in top_co.iterrows():
        pct   = row["median_house_value"] / max_v * 100
        is_me = row["county"] == selected_county
        lbl   = row["county"].replace(" County", "")
        fw    = "600" if is_me else "400"
        fc    = "#1a1a1a" if is_me else "#aaa"
        bar_c = "#1a1a1a" if is_me else "#e0e0de"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem;'>"
            f"<span style='font-size:0.72rem;font-weight:{fw};color:{fc};"
            f"width:110px;flex-shrink:0;white-space:nowrap;"
            f"overflow:hidden;text-overflow:ellipsis;'>{lbl}</span>"
            f"<div style='flex:1;height:2px;background:#ebebea;border-radius:99px;'>"
            f"<div style='width:{pct}%;height:2px;background:{bar_c};border-radius:99px;'></div></div>"
            f"<span style='font-family:\"DM Mono\",monospace;font-size:0.68rem;"
            f"color:{fc};width:52px;text-align:right;flex-shrink:0;'>{fmt(row['median_house_value'])}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:0.65rem;color:#ccc;text-align:center;letter-spacing:0.04em;'>"
    "1990 U.S. Census · LightGBM · For educational use only</p>",
    unsafe_allow_html=True
)

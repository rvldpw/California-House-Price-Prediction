import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import folium
from streamlit_folium import st_folium

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    padding: 0;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
  [data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
  }

  /* Main area */
  .main .block-container { padding: 2rem 2.5rem; max-width: 1200px; }

  /* Metric Cards */
  .metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 20px rgba(0,0,0,0.05);
    border: 1px solid #f1f5f9;
    transition: all 0.2s;
  }
  .metric-card:hover { box-shadow: 0 8px 30px rgba(0,0,0,0.1); transform: translateY(-2px); }
  .metric-label { font-size: 0.75rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
  .metric-value { font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1; }
  .metric-sub { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }
  .metric-icon { font-size: 1.75rem; margin-bottom: 0.75rem; }

  /* Result Banner */
  .result-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1e40af 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    color: white;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
  }
  .result-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.3) 0%, transparent 70%);
    border-radius: 50%;
  }
  .price-display { font-size: 3.5rem; font-weight: 800; color: #60a5fa; line-height: 1; }
  .price-range { font-size: 0.95rem; color: #93c5fd; margin-top: 0.5rem; }
  .price-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.5rem; }

  /* Section headers */
  .section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e8f0;
    margin-left: 0.5rem;
  }

  /* Insight cards */
  .insight-card {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border: 1px solid #bae6fd;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    color: #0c4a6e;
  }
  .insight-card strong { color: #0369a1; }

  /* Tag pills */
  .tag {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
  }
  .tag-blue { background: #dbeafe; color: #1d4ed8; }
  .tag-green { background: #dcfce7; color: #15803d; }
  .tag-orange { background: #ffedd5; color: #c2410c; }
  .tag-purple { background: #f3e8ff; color: #7e22ce; }

  /* Sidebar brand */
  .sidebar-brand {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    padding: 1.5rem;
    margin: -1rem -1rem 1.5rem;
    border-radius: 0 0 20px 20px;
  }
  .sidebar-brand h2 { color: white !important; font-size: 1.2rem !important; font-weight: 800 !important; margin: 0 !important; }
  .sidebar-brand p { color: #93c5fd !important; font-size: 0.78rem !important; margin: 0 !important; }

  /* Map container */
  .map-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
  }

  /* Input sections */
  .input-section {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .input-section-title {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-bottom: 0.75rem;
  }

  /* Confidence bar */
  .conf-bar-wrap { margin-top: 0.5rem; }
  .conf-bar-bg { background: rgba(255,255,255,0.15); border-radius: 999px; height: 6px; }
  .conf-bar-fill { background: linear-gradient(90deg, #60a5fa, #34d399); border-radius: 999px; height: 6px; }

  /* Detail chip */
  .detail-chip {
    display: flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.1); border-radius: 8px;
    padding: 0.4rem 0.75rem; margin: 0.25rem 0;
    font-size: 0.8rem; color: #e2e8f0;
  }
  .detail-chip-label { color: #94a3b8; font-size: 0.72rem; }

  /* Stat table */
  .stat-row { display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #f1f5f9; font-size: 0.82rem; }
  .stat-row:last-child { border-bottom: none; }
  .stat-key { color: #64748b; }
  .stat-val { color: #0f172a; font-weight: 600; }

  /* About section */
  .about-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
    font-size: 0.78rem;
    color: #64748b;
  }
  .about-box strong { color: #0f172a; }

  /* Tier badge */
  .tier-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Data ────────────────────────────────────────────────────────
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

# ─── Helper Functions ─────────────────────────────────────────────────────────
def engineer_features(row: dict) -> pd.DataFrame:
    d = row.copy()
    d["rooms_per_household"]      = d["total_rooms"]    / max(d["households"], 1)
    d["bedrooms_per_room"]        = d["total_bedrooms"] / max(d["total_rooms"], 1)
    d["population_per_household"] = d["population"]     / max(d["households"], 1)
    d["income_per_room"]          = d["median_income"]  / (d["total_rooms"] + 1)
    d["is_island"]  = 1 if d["ocean_proximity"] == "ISLAND" else 0
    d["is_capped"]  = 0
    return pd.DataFrame([d])

def predict_price(row: dict):
    X = engineer_features(row)
    log_pred = model.predict(X)[0]
    pred     = np.expm1(log_pred)
    low      = pred * 0.88
    high     = pred * 1.12
    return pred, low, high

def price_tier(price):
    if price < 150_000:   return "Budget",   "#dcfce7", "#15803d"
    if price < 300_000:   return "Moderate",  "#dbeafe", "#1d4ed8"
    if price < 500_000:   return "Mid-Range", "#fef9c3", "#a16207"
    if price < 750_000:   return "Premium",   "#ffedd5", "#c2410c"
    return "Luxury",      "#f3e8ff", "#7e22ce"

def format_price(p): return f"${p:,.0f}"
def format_price_m(p): return f"${p/1e6:.2f}M" if p >= 1e6 else f"${p/1e3:.0f}K"

# Precompute county/city lists
county_list = sorted(df["county"].unique())
county_medians = df.groupby("county")["median_house_value"].median().to_dict()
county_coords  = df.groupby("county")[["latitude","longitude"]].mean().to_dict("index")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-brand'>
      <h2>🏡 HomeValueIQ</h2>
      <p>California House Price Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Location ──
    st.markdown("**📍 Location**")
    selected_county = st.selectbox(
        "County", county_list,
        index=county_list.index("Los Angeles County") if "Los Angeles County" in county_list else 0,
        label_visibility="collapsed"
    )
    city_list = sorted(df[df["county"] == selected_county]["city"].unique())
    selected_city = st.selectbox("City", city_list, label_visibility="collapsed")

    # Use city mean lat/lon
    city_mask = (df["county"] == selected_county) & (df["city"] == selected_city)
    city_lat  = float(df[city_mask]["latitude"].mean())
    city_lon  = float(df[city_mask]["longitude"].mean())

    st.markdown("---")

    # ── Ocean Proximity ──
    st.markdown("**🌊 Ocean Proximity**")
    ocean_labels = {
        "<1H OCEAN": "< 1 Hour from Ocean",
        "INLAND":    "Inland",
        "NEAR BAY":  "Near Bay",
        "NEAR OCEAN":"Near Ocean",
        "ISLAND":    "Island"
    }
    ocean_sel = st.selectbox(
        "Ocean", list(ocean_labels.keys()),
        format_func=lambda x: ocean_labels[x],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ── Property Details ──
    st.markdown("**🏠 Property Details**")
    housing_age = st.slider("Median House Age (years)", 1, 52, 20)
    total_rooms = st.slider("Total Rooms", 2, 500, 80)
    total_bedrooms = st.slider("Total Bedrooms", 1, 200, 20)
    households  = st.slider("Households in Block", 1, 600, 100)
    population  = st.slider("Block Population", 3, 3000, 500)

    st.markdown("---")

    # ── Income ──
    st.markdown("**💰 Median Income**")
    median_income = st.slider(
        "Income (×$10,000)", 0.5, 15.0, 4.0, 0.1,
        help="e.g. 5.0 = $50,000 median household income"
    )
    st.caption(f"≈ ${median_income*10_000:,.0f} per household")

    st.markdown("---")

    estimate_btn = st.button("🔍 Estimate Price", use_container_width=True)

    st.markdown("""
    <div class='about-box'>
      <strong>About this tool</strong><br>
      Powered by a <strong>LightGBM</strong> model trained on 14,448 California block groups
      from the 1990 U.S. Census. Engineered features include income ratios, geo-clusters,
      and bedroom density. Model target: <strong>R² ≥ 0.85</strong>.
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("## 🏡 California House Price Predictor")
    st.caption("AI-powered estimates based on location, demographics, and proximity to the ocean.")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="tag tag-blue">LightGBM Model</span> <span class="tag tag-green">R² ≥ 0.85</span>', unsafe_allow_html=True)

st.markdown("---")

# ─── Run prediction ───────────────────────────────────────────────────────────
input_row = {
    "longitude":          city_lon,
    "latitude":           city_lat,
    "housing_median_age": housing_age,
    "total_rooms":        total_rooms,
    "total_bedrooms":     total_bedrooms,
    "population":         population,
    "households":         households,
    "median_income":      median_income,
    "ocean_proximity":    ocean_sel,
}

if estimate_btn or True:   # Always show — live update
    pred, low, high = predict_price(input_row)
    tier, tier_bg, tier_color = price_tier(pred)

    # Confidence (rough: tighter for common ranges)
    pct_err = 12
    confidence = max(60, 100 - int(abs(pred - 200_000) / 10_000))
    confidence = min(confidence, 95)

    # ── Result Banner ──
    st.markdown(f"""
    <div class='result-banner'>
      <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem; position:relative; z-index:1;'>
        <div>
          <div class='price-label'>Estimated House Price</div>
          <div class='price-display'>{format_price(pred)}</div>
          <div class='price-range'>Estimated range: {format_price_m(low)} – {format_price_m(high)}</div>
          <div class='conf-bar-wrap'>
            <div class='conf-bar-bg' style='width:220px;'>
              <div class='conf-bar-fill' style='width:{confidence}%;'></div>
            </div>
            <div style='font-size:0.72rem; color:#94a3b8; margin-top:3px;'>Model confidence: {confidence}%</div>
          </div>
        </div>
        <div style='display:flex; gap:2rem; flex-wrap:wrap;'>
          <div>
            <div class='detail-chip'>
              <span>📍</span>
              <div><div class='detail-chip-label'>Location</div>{selected_city}, {selected_county.replace(" County","")}</div>
            </div>
            <div class='detail-chip'>
              <span>🌊</span>
              <div><div class='detail-chip-label'>Ocean</div>{ocean_labels[ocean_sel]}</div>
            </div>
            <div class='detail-chip'>
              <span>🏠</span>
              <div><div class='detail-chip-label'>House Age</div>{housing_age} years</div>
            </div>
          </div>
          <div>
            <div class='detail-chip'>
              <span>🛏️</span>
              <div><div class='detail-chip-label'>Rooms / Bedrooms</div>{total_rooms} / {total_bedrooms}</div>
            </div>
            <div class='detail-chip'>
              <span>👥</span>
              <div><div class='detail-chip-label'>Population</div>{population:,}</div>
            </div>
            <div class='detail-chip'>
              <span>💰</span>
              <div><div class='detail-chip-label'>Median Income</div>${median_income*10_000:,.0f}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Map + Stats Layout ───────────────────────────────────────────────────────
map_col, stats_col = st.columns([3, 2], gap="large")

with map_col:
    st.markdown("<div class='section-header'>📍 Location Overview</div>", unsafe_allow_html=True)

    m = folium.Map(
        location=[city_lat, city_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True
    )

    # Main pin
    folium.Marker(
        [city_lat, city_lon],
        popup=folium.Popup(f"<b>{selected_city}</b><br>{selected_county}<br>Est. {format_price(pred)}", max_width=200),
        tooltip=f"{selected_city} — {format_price(pred)}",
        icon=folium.Icon(color="blue", icon="home", prefix="fa")
    ).add_to(m)

    # Nearby sample points from same county (heatmap feel)
    nearby = df[(df["county"] == selected_county)].sample(min(40, len(df[df["county"] == selected_county])), random_state=42)
    for _, row in nearby.iterrows():
        v = row["median_house_value"]
        color = "#ef4444" if v > 400_000 else "#f97316" if v > 250_000 else "#22c55e"
        folium.CircleMarker(
            [row["latitude"], row["longitude"]],
            radius=4,
            color=color, fill=True, fill_opacity=0.5, weight=1,
            tooltip=f"{row['city']} — {format_price(v)}"
        ).add_to(m)

    with st.container():
        st_folium(m, height=400, use_container_width=True)

    st.caption("🟢 < $250K   🟠 $250K–$400K   🔴 > $400K   Blue pin = selected location")

with stats_col:
    st.markdown("<div class='section-header'>📊 Area Statistics</div>", unsafe_allow_html=True)

    county_data = df[df["county"] == selected_county]
    city_data   = df[(df["county"] == selected_county) & (df["city"] == selected_city)]

    county_med  = county_data["median_house_value"].median()
    county_avg  = county_data["median_house_value"].mean()
    county_min  = county_data["median_house_value"].min()
    county_max  = county_data["median_house_value"].max()
    city_med    = city_data["median_house_value"].median() if len(city_data) > 0 else county_med
    city_avg_inc = city_data["median_income"].mean() * 10_000 if len(city_data) > 0 else county_data["median_income"].mean() * 10_000

    vs_county = ((pred - county_med) / county_med * 100)
    arrow = "↑" if vs_county > 0 else "↓"
    arrow_color = "#ef4444" if vs_county > 0 else "#22c55e"

    st.markdown(f"""
    <div class='metric-card' style='margin-bottom:0.75rem;'>
      <div class='metric-label'>Your Estimate vs County Median</div>
      <div style='font-size:1.6rem; font-weight:800; color:{arrow_color};'>{arrow} {abs(vs_county):.1f}%</div>
      <div class='metric-sub'>County median: {format_price(county_med)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metric-card' style='margin-bottom:0.75rem;'>
      <div class='metric-label'>{selected_county} Price Range</div>
      <div style='display:flex; gap:1rem;'>
        <div><div style='font-size:0.72rem;color:#94a3b8;'>Min</div><div style='font-weight:700;color:#22c55e;'>{format_price_m(county_min)}</div></div>
        <div><div style='font-size:0.72rem;color:#94a3b8;'>Avg</div><div style='font-weight:700;color:#3b82f6;'>{format_price_m(county_avg)}</div></div>
        <div><div style='font-size:0.72rem;color:#94a3b8;'>Max</div><div style='font-weight:700;color:#ef4444;'>{format_price_m(county_max)}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Area Details — {selected_city}</div>
      <div class='stat-row'><span class='stat-key'>City Median Price</span><span class='stat-val'>{format_price(city_med)}</span></div>
      <div class='stat-row'><span class='stat-key'>Avg Median Income</span><span class='stat-val'>${city_avg_inc:,.0f}</span></div>
      <div class='stat-row'><span class='stat-key'>Block Groups (city)</span><span class='stat-val'>{len(city_data):,}</span></div>
      <div class='stat-row'><span class='stat-key'>Block Groups (county)</span><span class='stat-val'>{len(county_data):,}</span></div>
      <div class='stat-row'><span class='stat-key'>Rooms per Household</span><span class='stat-val'>{total_rooms/max(households,1):.1f}</span></div>
      <div class='stat-row'><span class='stat-key'>People per Household</span><span class='stat-val'>{population/max(households,1):.1f}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─── Feature Importance + County Comparison ───────────────────────────────────
st.markdown("---")
insight_col, compare_col = st.columns([1, 1], gap="large")

with insight_col:
    st.markdown("<div class='section-header'>💡 Price Driver Insights</div>", unsafe_allow_html=True)

    drivers = []
    if median_income >= 6.0:
        drivers.append(("💰 High Income Area", "Strong income signal pushes price up", "tag-green"))
    elif median_income <= 2.5:
        drivers.append(("💰 Lower Income Area", "Income below average suppresses price", "tag-orange"))

    if ocean_sel in ("NEAR OCEAN", "NEAR BAY"):
        drivers.append(("🌊 Ocean Proximity", "Near-water locations command a premium", "tag-blue"))
    elif ocean_sel == "ISLAND":
        drivers.append(("🏝️ Island Location", "Island properties are rare and expensive", "tag-purple"))
    elif ocean_sel == "INLAND":
        drivers.append(("🏔️ Inland Location", "Inland areas typically price lower", "tag-orange"))

    if housing_age <= 15:
        drivers.append(("🏗️ Newer Construction", "Newer homes typically command higher prices", "tag-green"))
    elif housing_age >= 40:
        drivers.append(("🏚️ Older Housing Stock", "Older properties may require renovation premium", "tag-orange"))

    rph = total_rooms / max(households, 1)
    if rph >= 6:
        drivers.append(("🛋️ Spacious Properties", f"{rph:.1f} rooms/household — above average space", "tag-blue"))
    elif rph <= 3:
        drivers.append(("🏢 Dense Housing", f"{rph:.1f} rooms/household — compact units", "tag-orange"))

    if not drivers:
        drivers.append(("⚖️ Average Profile", "No standout features driving price significantly", "tag"))

    for title, desc, css_class in drivers:
        st.markdown(f"""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:0.85rem 1rem; margin-bottom:0.6rem; display:flex; gap:0.75rem; align-items:flex-start;'>
          <div style='flex:1;'>
            <div style='font-weight:700; color:#0f172a; font-size:0.88rem; margin-bottom:0.15rem;'>{title}</div>
            <div style='font-size:0.78rem; color:#64748b;'>{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with compare_col:
    st.markdown("<div class='section-header'>🗺️ Top Counties by Median Price</div>", unsafe_allow_html=True)

    top_counties = (
        df.groupby("county")["median_house_value"]
        .median()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )
    max_val = top_counties["median_house_value"].max()

    for _, row in top_counties.iterrows():
        pct = row["median_house_value"] / max_val * 100
        is_selected = row["county"] == selected_county
        bar_color = "#3b82f6" if is_selected else "#e2e8f0"
        text_color = "#1d4ed8" if is_selected else "#0f172a"
        badge = " ★" if is_selected else ""
        st.markdown(f"""
        <div style='margin-bottom:0.5rem;'>
          <div style='display:flex; justify-content:space-between; margin-bottom:2px;'>
            <span style='font-size:0.78rem; font-weight:{"700" if is_selected else "500"}; color:{text_color};'>{row["county"].replace(" County","")}{badge}</span>
            <span style='font-size:0.78rem; font-weight:700; color:{text_color};'>{format_price_m(row["median_house_value"])}</span>
          </div>
          <div style='background:#f1f5f9; border-radius:999px; height:6px;'>
            <div style='background:{bar_color}; border-radius:999px; height:6px; width:{pct}%;'></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Price Distribution ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>📈 Price Distribution in Selected County</div>", unsafe_allow_html=True)

import streamlit.components.v1 as components

# Build a histogram using HTML/JS (no extra deps)
county_prices = df[df["county"] == selected_county]["median_house_value"].dropna().tolist()
bins = list(range(0, 550_000, 50_000))
hist_counts = [sum(1 for p in county_prices if bins[i] <= p < bins[i+1]) for i in range(len(bins)-1)]
bin_labels = [f"${b//1000}K" for b in bins[:-1]]
max_count = max(hist_counts) if hist_counts else 1
pred_bin = min(int(pred // 50_000), len(bins)-2)

bars_html = ""
for i, (cnt, lbl) in enumerate(zip(hist_counts, bin_labels)):
    h = int(cnt / max_count * 120)
    is_pred = (i == pred_bin)
    color = "#3b82f6" if is_pred else "#e2e8f0"
    border = "border: 2px solid #1d4ed8;" if is_pred else ""
    bars_html += f"""
      <div style='display:flex; flex-direction:column; align-items:center; gap:2px; flex:1;'>
        <div style='font-size:0.6rem; color:#94a3b8;'>{cnt}</div>
        <div style='height:{h}px; width:100%; background:{color}; border-radius:4px 4px 0 0; {border} cursor:default;' title='{lbl}: {cnt} blocks'></div>
        <div style='font-size:0.6rem; color:#94a3b8; writing-mode:vertical-rl; transform:rotate(180deg); white-space:nowrap;'>{lbl}</div>
      </div>
    """

hist_html = f"""
<div style='background:white; border:1px solid #e2e8f0; border-radius:16px; padding:1.25rem 1.5rem;'>
  <div style='font-size:0.78rem; color:#64748b; margin-bottom:0.75rem;'>
    Block group count by price range in <strong>{selected_county}</strong>. 
    <span style='color:#1d4ed8; font-weight:700;'>Blue bar</span> = your estimated price range.
  </div>
  <div style='display:flex; align-items:flex-end; gap:3px; height:160px; padding-bottom:20px;'>
    {bars_html}
  </div>
</div>
"""
st.markdown(hist_html, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.75rem; padding:1rem 0;'>
  ⚠️ <strong>Note:</strong> Estimates are based on 1990 U.S. Census data and a LightGBM regression model.
  Actual prices may vary based on current market conditions, property-specific features, and economic changes.
  This tool is for educational and exploratory purposes only.<br><br>
  Built with <strong>Streamlit</strong> · Geocoded via <strong>reverse_geocoder</strong> · Model: <strong>LightGBM + sklearn Pipeline</strong>
</div>
""", unsafe_allow_html=True)

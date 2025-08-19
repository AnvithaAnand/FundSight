import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="FundSight", page_icon="📊", layout="wide")
st.title("FundSight — Startup Funding Intelligence (India)")

# ---- Data loader
@st.cache_data
def load_data():
    df = None
    for path in ["FundSight_clean.csv", "FundSight_clean_sample.csv"]:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue

    if df is None:
        st.error("No data file found. Please add FundSight_clean.csv to the repo.")
        st.stop()

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['final_amount'] = pd.to_numeric(df['final_amount'], errors='coerce')
    for c in ['startup_name', 'sector', 'city_clean', 'investors_clean', 'stage', 'type']:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip().str.title()

    df = df[df['year'].between(2011, 2021, inclusive='both')]
    df = df[df['final_amount'].notna() & (df['final_amount'] > 0)]
    return df

df = load_data()

# ---- Sidebar filters
years = sorted(df['year'].unique().tolist())
min_y, max_y = st.sidebar.select_slider("Year range", options=years, value=(min(years), max(years)))
sector_list = ["All"] + sorted([s for s in df['sector'].unique() if s]) if 'sector' in df.columns else ["All"]
city_list   = ["All"] + sorted([c for c in df['city_clean'].unique() if c]) if 'city_clean' in df.columns else ["All"]

sel_sector  = st.sidebar.selectbox("Sector", sector_list, index=0)
sel_city    = st.sidebar.selectbox("City",   city_list,   index=0)
TOPN        = st.sidebar.slider("Top N (rank tables)", 5, 25, 10)

f = df[(df['year'] >= min_y) & (df['year'] <= max_y)].copy()
if sel_sector != "All" and 'sector' in f.columns:
    f = f[f['sector'].str.strip().str.title() == sel_sector]

if sel_city != "All" and 'city_clean' in f.columns:
    f = f[f['city_clean'].str.strip().str.title() == sel_city]


# ---- KPIs
total_usd = f['final_amount'].sum()
num_deals = len(f)
avg_deal  = f['final_amount'].mean() if num_deals else 0.0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Funding (USD)", f"${total_usd:,.0f}")
k2.metric("Deals", f"{num_deals:,}")
k3.metric("Avg Deal", f"${avg_deal:,.0f}")
k4.metric("Years", f"{f['year'].nunique()}")

# ---- Yearly trend
st.subheader("Yearly Total Funding")
yearly = f.groupby('year', as_index=False)['final_amount'].sum().sort_values('year')
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(yearly['year'], yearly['final_amount'])
for x, y in zip(yearly['year'], yearly['final_amount']):
    ax.text(x, y * 1.01, f"{y / 1e9:.2f}B", ha='center', va='bottom', fontsize=8)
ax.set_xlabel("Year")
ax.set_ylabel("Funding (USD)")
ax.set_title("Total Funding by Year")
st.pyplot(fig)

# ---- Ranked tables
st.subheader("Ranked Views")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top Startups**")
    top_startups = (f.groupby('startup_name', as_index=False)['final_amount']
                      .sum().sort_values('final_amount', ascending=False).head(TOPN))
    st.dataframe(top_startups.rename(columns={'startup_name': 'Startup', 'final_amount': 'Funding (USD)'}), use_container_width=True)

    st.markdown("**Top Cities**")
    if 'city_clean' in f.columns:
        top_cities = (f.groupby('city_clean', as_index=False)['final_amount']
                        .sum().sort_values('final_amount', ascending=False).head(TOPN))
        st.dataframe(top_cities.rename(columns={'city_clean': 'City', 'final_amount': 'Funding (USD)'}), use_container_width=True)
    else:
        st.info("City column not available in this dataset.")

with c2:
    st.markdown("**Top Sectors**")
    if 'sector' in f.columns:
        top_sectors = (f.groupby('sector', as_index=False)['final_amount']
                         .sum().sort_values('final_amount', ascending=False).head(TOPN))
        st.dataframe(top_sectors.rename(columns={'sector': 'Sector', 'final_amount': 'Funding (USD)'}), use_container_width=True)
    else:
        st.info("Sector column not available in this dataset.")

    st.markdown("**Top Investors**")
    inv_cols_present = all(col in f.columns for col in ['investors_clean', 'final_amount'])
    if inv_cols_present and not f[['investors_clean']].dropna().empty:
        inv = f[['investors_clean', 'final_amount']].dropna()
        inv = inv.assign(investor=inv['investors_clean'].str.split(',')).explode('investor')
        inv['investor'] = inv['investor'].str.strip().str.title()
        top_investors = (inv.groupby('investor', as_index=False)['final_amount']
                           .sum().sort_values('final_amount', ascending=False).head(TOPN))
        st.dataframe(top_investors.rename(columns={'investor': 'Investor', 'final_amount': 'Funding (USD)'}), use_container_width=True)
    else:
        st.info("No investor data in current filter.")

# ---- Risk heuristics
st.subheader("Risk Heuristics (HHI concentration & monthly volatility)")
tmp = f.copy()
tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
tmp['month'] = tmp['date'].dt.to_period('M')

def hhi(x):
    return ((x / x.sum()) ** 2).sum()

conc = (tmp.groupby(['year', 'startup_name'])['final_amount'].sum()
          .groupby('year').apply(hhi).reset_index(name='HHI'))

vol = (tmp.dropna(subset=['month'])
         .groupby('month')['final_amount'].sum().reset_index()
         .assign(year=lambda d: d['month'].dt.year)
         .groupby('year')['final_amount'].std()
         .reset_index(name='Volatility'))

by_year_f = (tmp.groupby('year')
               .agg(Total=('final_amount', 'sum'),
                    Deals=('final_amount', 'size'),
                    AvgDeal=('final_amount', 'mean'))
               .reset_index())

risk = by_year_f.merge(conc, on='year', how='left').merge(vol, on='year', how='left')

# ✅ FIXED BLOCK
risk_clean = risk.dropna(subset=['HHI', 'Volatility']).copy()
if not risk_clean.empty:
    scaler = MinMaxScaler()
    risk_clean[['HHI_n', 'Vol_n']] = scaler.fit_transform(risk_clean[['HHI', 'Volatility']])
    risk = risk.merge(risk_clean[['year', 'HHI_n', 'Vol_n']], on='year', how='left')
    risk['RiskScore'] = 0.6 * risk['HHI_n'].fillna(0) + 0.4 * risk['Vol_n'].fillna(0)
else:
    risk['HHI_n'] = 0
    risk['Vol_n'] = 0
    risk['RiskScore'] = 0

st.dataframe(risk.sort_values('year'), use_container_width=True)

# Risk score line
fig, ax = plt.subplots(figsize=(9, 4))
r = risk.sort_values('year')
ax.plot(r['year'], r['RiskScore'], marker='o')
ax.set_title('Composite Risk Score (0–1)')
ax.set_xlabel('Year')
ax.set_ylabel('Risk Score')
ax.grid(True)
st.pyplot(fig)

# Download filtered data
st.download_button("Download filtered dataset (CSV)", f.to_csv(index=False), "FundSight_filtered.csv", "text/csv")

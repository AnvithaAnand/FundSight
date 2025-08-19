import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('FundSight_clean.csv')
risk_summary_df = pd.read_csv('FundSight_risk_summary.csv')

# City alias correction
city_aliases = {
    'Bangalore': 'Bengaluru',
    'Bengaluru': 'Bengaluru',
    'Delhi NCR': 'Delhi NCR',
    'Mumbai': 'Mumbai',
    'Pune': 'Pune',
    'Hyderabad': 'Hyderabad',
    'Chennai': 'Chennai',
    'Jaipur': 'Jaipur',
    'Kolkata': 'Kolkata',
    'Ahmedabad': 'Ahmedabad',
    'Palo Alto': 'Palo Alto',
    'Singapore': 'Singapore'
}

# Clean year column
df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
df.dropna(subset=['Year'], inplace=True)
df['Year'] = df['Year'].astype(int)

# UI settings
st.set_page_config(layout="wide", page_title="FundSight")

# Sidebar filters
st.sidebar.header("Filters")
years = df['Year'].dropna().astype(int)
min_year = years.min()
max_year = years.max()

selected_years = st.sidebar.slider("Year range", int(min_year), int(max_year), (int(min_year), int(max_year)))

sectors = ["All"] + sorted(df['Sector'].dropna().unique())
selected_sector = st.sidebar.selectbox("Sector", sectors)

cities = ["All"] + sorted(city_aliases.keys())
selected_city_input = st.sidebar.selectbox("City", cities)

top_n = st.sidebar.slider("Top N (rank tables)", 5, 25, 10)

# Map alias to actual city in dataset
selected_city = city_aliases.get(selected_city_input, selected_city_input)

# Filtering
filtered_df = df[
    (df['Year'] >= selected_years[0]) & 
    (df['Year'] <= selected_years[1])
]

if selected_sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]

if selected_city != "All":
    filtered_df = filtered_df[filtered_df['City'] == selected_city]

# Main title
st.title("FundSight — Startup Funding Intelligence (India)")

# Metrics
total_funding = filtered_df['AmountInUSD'].sum()
total_deals = filtered_df.shape[0]
avg_deal = filtered_df['AmountInUSD'].mean() if total_deals > 0 else 0
total_years = filtered_df['Year'].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Funding (USD)", f"${total_funding:,.0f}")
col2.metric("Deals", total_deals)
col3.metric("Avg Deal", f"${avg_deal:,.0f}")
col4.metric("Years", total_years)

# Visualization: Total funding by year
st.subheader("Yearly Total Funding")
if not filtered_df.empty:
    funding_by_year = filtered_df.groupby('Year')['AmountInUSD'].sum().reset_index()
    fig1 = px.bar(funding_by_year, x='Year', y='AmountInUSD', labels={'AmountInUSD': 'Total Funding (USD)'})
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")

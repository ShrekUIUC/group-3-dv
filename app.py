import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import math
# Set page config
st.set_page_config(page_title="Nuisance Complaints Dashboard", layout="wide")

# Title and introduction
st.title("Nuisance Complaints Analysis Dashboard")
st.markdown("""
**Team Members:**
* Lu Chang (luchang2@illinois.edu)
* Qiming Li (qimingl4@illinois.edu)
* Ruchita Alate (ralate2@illinois.edu)
* Shreyas Kulkarni (ssk16@illinois.edu)
* Vishal Devulapalli (nsd3@illinois.edu)
""")
st.write("This dashboard analyzes nuisance complaints data from the City of Urbana.")

# Load and clean data
@st.cache_data
def load_and_clean_data():
    try:
        # Load data
        data = pd.read_csv('Nuisance_Complaints.csv')

        # Drop rows with missing 'File Number'
        data = data.dropna(subset=['File Number'])

        # Convert dates and handle date-related columns
        data['Date Reported'] = pd.to_datetime(data['Date Reported'])
        data['Date Notice Mailed or Given'] = pd.to_datetime(data['Date Notice Mailed or Given'])
        data['File Close Date'] = pd.to_datetime(data['File Close Date'], errors='coerce')

        # Handle 'Date Notice Mailed or Given'
        median_delay = (data['Date Notice Mailed or Given'] - data['Date Reported']).dt.days.median()
        data.loc[data['Date Notice Mailed or Given'].isna(), 'Date Notice Mailed or Given'] = \
            data.loc[data['Date Notice Mailed or Given'].isna(), 'Date Reported'] + pd.Timedelta(days=median_delay)

        # Handle 'Type of Complaint'
        data['Type of Complaint'] = data['Type of Complaint'].fillna('Unknown')

        # Handle 'Disposition'
        most_common_disposition = data.groupby('Type of Complaint')['Disposition'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Pending'
        )
        data['Disposition'] = data.apply(
            lambda row: most_common_disposition[row['Type of Complaint']]
            if pd.isna(row['Disposition']) else row['Disposition'],
            axis=1
        )

        # Calculate processing time for resolved cases
        data['Processing Time'] = np.where(
            data['File Close Date'].notna(),
            (data['File Close Date'] - data['Date Reported']).dt.days,
            np.nan
        )

        # Handle 'Method Submitted'
        data.loc[
            (data['Submitted Online?']) & (data['Method Submitted'].isna()),
            'Method Submitted'
        ] = 'Online'
        data['Method Submitted'] = data['Method Submitted'].fillna(data['Method Submitted'].mode()[0])

        # Drop rows with missing critical values
        data = data.dropna(subset=['Submitted Online?', 'Mapped Location'])

        # Extract and clean location data
        data['Latitude'] = data['Mapped Location'].str.extract(r'\(([^,]+),')[0].astype(float)
        data['Longitude'] = data['Mapped Location'].str.extract(r', ([^,]+)\)')[0].astype(float)

        # Ensure Year Reported is integer
        data['Year Reported'] = data['Year Reported'].astype(int)

        return data

    except Exception as e:
        st.error(f"Error in data preprocessing: {str(e)}")
        raise e

# Load the data
try:
    data = load_and_clean_data()
    st.success("Data successfully loaded and cleaned!")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Create sidebar

st.sidebar.header("Dashboard Controls")

# Get unique years and convert to list for selectbox
year_list = sorted(data['Year Reported'].unique().tolist())
year_options = ['All Time'] + [int(year) for year in year_list]  # Convert years to integers

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=year_options,
)

# Filter data based on selected year
if selected_year == 'All Time':
    filtered_data = data  # Use complete dataset when 'All Time' is selected
else:
    filtered_data = data[data['Year Reported'] == selected_year]

# Update header text
if selected_year == 'All Time':
    st.header("Analysis for All Time")
else:
    st.header(f"Analysis for Year {selected_year}")
# Add visualization type selector
viz_type = st.sidebar.selectbox(
    "Select Visualization",
    ["Complaint Types", "Geographic Distribution", "Resolution Status",
     "Submission Methods", "Processing Time"]
)

# Filter data based on selected year
filtered_data = data[data['Year Reported'] == selected_year]

# Main content
st.header(f"Analysis for Year {selected_year}")

# Create metrics
# Create metrics
# Create metrics
# Create metrics
# Create metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Complaints", len(filtered_data))
with col2:
    avg_time = filtered_data['Processing Time'].mean()
    st.metric("Average Processing Time", f"{avg_time:.1f} days" if pd.notna(avg_time) else "N/A")
with col3:
    if not filtered_data.empty:
        most_common = filtered_data['Type of Complaint'].value_counts().index[0]
        st.metric("Most Common Type", most_common)
    else:
        st.metric("Most Common Type", "N/A")
if viz_type == "Complaint Types":
    # Interactive Pie Chart
    st.subheader("Interactive Complaint Types Pie Chart")
    complaint_counts = filtered_data['Type of Complaint'].value_counts().reset_index()
    complaint_counts.columns = ['Complaint Type', 'Count']

    fig = px.pie(
        complaint_counts,
        names='Complaint Type',
        values='Count',
        title=f'Complaint Types Distribution in {selected_year}',
        hole=0.4  # Donut style
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Geographic Distribution":
    # Clustered Heatmap
    st.subheader("Clustered Heatmap of Complaints")
    map_center = [filtered_data['Latitude'].dropna().mean(), filtered_data['Longitude'].dropna().mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    heat_data = filtered_data[['Latitude', 'Longitude']].dropna().values.tolist()
    HeatMap(heat_data).add_to(m)

    st_data = st_folium(m, width=700, height=500)


elif viz_type == "Resolution Status":
    st.subheader("Complaint Resolution Status")
    fig, ax = plt.subplots(figsize=(10, 6))
    resolution_counts = filtered_data['Disposition'].value_counts()
    sns.barplot(x=resolution_counts.values, y=resolution_counts.index)
    plt.title(f'Resolution Status Distribution in {selected_year}')
    st.pyplot(fig)

elif viz_type == "Submission Methods":
    st.subheader("Submission Methods Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    submission_counts = filtered_data['Method Submitted'].value_counts()
    sns.barplot(x=submission_counts.values, y=submission_counts.index)
    plt.title(f'Submission Methods in {selected_year}')
    st.pyplot(fig)


elif viz_type == "Processing Time":
    st.subheader("Processing Time Analysis")
    # Filter for resolved cases only
    resolved_data = filtered_data[filtered_data['File Close Date'].notna()]
    if len(resolved_data) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=resolved_data, x='Processing Time', bins=30)
        plt.title(f'Distribution of Processing Times in {selected_year}')
        plt.xlabel('Processing Time (Days)')
        st.pyplot(fig)
    else:
        st.write("No resolved cases in this period")

# Additional insights
st.header("Key Insights")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 3 Complaint Types")
    top_complaints = filtered_data['Type of Complaint'].value_counts().head(3)
    st.write(top_complaints)

with col2:
    st.subheader("Resolution Efficiency")
    resolution_rate = (filtered_data['Disposition'].value_counts() /
                      len(filtered_data) * 100).round(2)
    st.write(resolution_rate)

# Footer
st.markdown("---")
st.markdown("Dataset provided by the City of Urbana Open Data Portal")
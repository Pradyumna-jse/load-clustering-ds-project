import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cluster import KMeans

# Add tools to path
sys.path.append("tools/")
from make_df import make_df
from plot_funcs import plot_range

st.set_page_config(page_title="Loads Clustering Dashboard", layout="wide")

st.title("📊 Commercial Loads Clustering Dashboard")

# Function to find all relevant CSVs in the root
def find_csvs(root_dir):
    csv_files = []
    # Relevant filenames to look for based on README
    targets = ["all_commercial.csv", "all_residential.csv", "buildings_feats.csv"]
    for file in os.listdir(root_dir):
        if file.endswith(".csv"):
            csv_files.append(file)
    return sorted(csv_files)

csv_options = find_csvs(".")

if not csv_options:
    st.error("No CSV files found in the current directory.")
else:
    selected_csv = st.sidebar.selectbox("Select a dataset file", csv_options)

    if selected_csv:
        st.write(f"### Analyzing: `{selected_csv}`")
        
        # Load data
        with st.spinner("Loading data..."):
            try:
                # Aggregate datasets logic
                if "all_" in selected_csv or "buildings_feats" in selected_csv:
                    df = pd.read_csv(selected_csv, index_col=0)
                    is_aggregate = True
                else:
                    df = make_df(selected_csv)
                    is_aggregate = False
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Data Preview")
            st.dataframe(df.head())

        with col2:
            st.write("#### Statistics")
            st.write(df.describe())

        st.write("---")
        
        if is_aggregate:
            st.write("#### Building Profiles Visualization")
            # For aggregate files, rows are buildings, cols 0-23 are hours
            hour_cols = [str(i) for i in range(24)]
            if all(col in df.columns for col in hour_cols):
                fig, ax = plt.subplots(figsize=(15, 5))
                # Show first 50 buildings
                for i in range(min(50, len(df))):
                    plt.plot(range(24), df.iloc[i][hour_cols].values, color=(0,0,0,0.1))
                plt.title("Sample Load Profiles (First 50 Buildings)")
                plt.xlabel("Hour of the day")
                plt.ylabel("Normalized Load")
                st.pyplot(fig)
            else:
                st.warning("Selected CSV does not contain standard hourly columns (0-23). Showing raw data visualization instead.")
        else:
            st.write("#### Daily Load Profiles")
            fig, ax = plt.subplots(figsize=(15, 5))
            days = [day.strftime("%Y-%m-%d") for day in df.index.date]
            unique_days = sorted(list(set(days)))
            for day in unique_days[:14]: # Show first 2 weeks
                plt.plot(df.loc[day].index.hour, 
                         df.loc[day]['Electricity:Facility [kW](Hourly)'], 
                         color=(0,0,0,0.1))
            plt.title("Sample Daily Load Profiles (First 2 Weeks)")
            plt.xlabel("Hour of the day")
            plt.ylabel("Load profile [kW]")
            st.pyplot(fig)

        st.write("---")
        st.write("#### Clustering Analysis")
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 6)
        
        if st.button("Run KMeans Clustering"):
            with st.spinner("Clustering..."):
                if is_aggregate:
                    hour_cols = [str(i) for i in range(24)]
                    if all(col in df.columns for col in hour_cols):
                        data_to_cluster = df[hour_cols].values
                    else:
                        st.error("Cannot cluster: hourly data columns not found.")
                        st.stop()
                else:
                    # Prepare data from single building time series
                    daily_data = []
                    days = [day.strftime("%Y-%m-%d") for day in df.index.date]
                    unique_days = sorted(list(set(days)))
                    for day in unique_days:
                        day_load = df.loc[day]['Electricity:Facility [kW](Hourly)'].values
                        if len(day_load) == 24:
                            daily_data.append(day_load)
                    data_to_cluster = np.array(daily_data)
                
                if len(data_to_cluster) < n_clusters:
                    st.error(f"Not enough data points ({len(data_to_cluster)}) for {n_clusters} clusters.")
                else:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(data_to_cluster)
                    
                    st.write(f"Clustering complete. Found {n_clusters} clusters across {len(data_to_cluster)} samples.")
                    
                    # Plot clusters
                    fig_clusters, axes = plt.subplots(1, n_clusters, figsize=(20, 4), sharey=True)
                    if n_clusters == 1: axes = [axes]
                    for i in range(n_clusters):
                        cluster_samples = data_to_cluster[clusters == i]
                        for sample in cluster_samples:
                            axes[i].plot(range(24), sample, color=(0,0,0,0.05))
                        axes[i].plot(range(24), kmeans.cluster_centers_[i], color='red', linewidth=2)
                        axes[i].set_title(f"Cluster {i}\n({len(cluster_samples)} samples)")
                        axes[i].set_xlabel("Hour")
                    
                    st.pyplot(fig_clusters)

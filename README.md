# load-clustering-ds-project
This project analyzes hourly electricity consumption of commercial buildings and groups similar usage patterns using K-Means clustering. It converts data into daily profiles, extracts key features like peaks and averages, and visualizes results to identify behavior patterns for energy optimization.
# 📊 Commercial Load Clustering using Machine Learning

## 🔍 Overview

This project analyzes electricity consumption patterns of commercial buildings and groups them into similar categories using **K-Means clustering**.
The goal is to identify patterns in energy usage and understand how different types of buildings consume electricity.

---

## 🎯 Objectives

* Analyze hourly electricity consumption data
* Convert time-series data into daily load profiles
* Extract meaningful features (peaks, average, variation)
* Apply clustering to identify similar usage patterns
* Visualize and interpret clusters

---

## 📁 Dataset

The project uses:

* **all_commercial.csv** → Main dataset containing hourly electricity usage of commercial buildings

Additional processed datasets:

* buildings_feats.csv
* buildings_feats_scaled.csv
* buildings_feats_hours.csv
* buildings_feats_hours_summer.csv

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Clean and format time-series data
* Handle missing values and time inconsistencies

### 2. Load Profile Creation

* Convert raw data into **24-hour daily profiles**

### 3. Feature Engineering

* Number of peaks
* Peak timings
* Average consumption
* Maximum consumption
* Standard deviation

### 4. Clustering

* Algorithm used: **K-Means**
* Groups similar load patterns together

### 5. Visualization

* Daily load curves
* Cluster-wise average patterns
* Interactive dashboard using Streamlit

---

## 📊 Results

The model groups buildings into clusters such as:

* Office-type usage (daytime peaks)
* Restaurant/hospital (multiple peaks)
* Hotel/apartment (continuous usage)
* Warehouse (flat load)

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

2. Run the dashboard:

```bash
streamlit run dashboard.py
```

---

## 📌 Key Insight

Different buildings exhibit distinct electricity usage patterns, and clustering helps in automatically identifying these patterns for better energy management.

---

## 🔮 Future Work

* Compare commercial vs residential usage
* Apply advanced clustering techniques (DBSCAN, Hierarchical)
* Real-time energy prediction

---

## 👨‍💻 Author

Pradyumna Senapati

---

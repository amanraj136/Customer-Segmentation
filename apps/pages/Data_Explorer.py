import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Data Explorer", layout="wide")

st.title("📂 Data Explorer")
st.markdown("### Explore, filter, and understand your dataset.")
st.markdown("---")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("apps/data/clustered_data.csv") 

st.subheader("📁 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", df.shape[1])
col3.metric("Clusters", df["Clusters"].nunique())

st.markdown("---")

# -----------------------------
# Data Preview
# -----------------------------
st.subheader("🔎 Preview Data")

num_rows = st.slider("Number of rows to display", 5, 50, 10)
st.dataframe(df.head(num_rows), use_container_width=True)

# -----------------------------
# Select Columns to Display
# -----------------------------
st.subheader("🧭 Select Columns")

all_columns = df.columns.tolist()
selected_cols = st.multiselect("Choose columns:", all_columns, default=all_columns)

if selected_cols:
    st.dataframe(df[selected_cols], use_container_width=True)

st.markdown("---")

# -----------------------------
# Dynamic Filters
# -----------------------------
st.subheader("🎛️ Filter Data")

filter_cols = st.multiselect("Select columns to filter:", all_columns)

filtered_df = df.copy()

for col in filter_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        min_val, max_val = st.slider(
            f"Range for {col}", 
            float(df[col].min()), 
            float(df[col].max()), 
            (float(df[col].min()), float(df[col].max()))
        )
        filtered_df = filtered_df[df[col].between(min_val, max_val)]
    else:
        vals = st.multiselect(f"Values for {col}", df[col].unique())
        if vals:
            filtered_df = filtered_df[df[col].isin(vals)]

st.subheader("📄 Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

st.success(f"Showing {len(filtered_df):,} rows after filtering.")

st.markdown("---")

# -----------------------------
# Summary Statistics
# -----------------------------
st.subheader("📊 Summary Statistics")

with st.expander("Show Summary Statistics"):
    st.dataframe(df.describe(), use_container_width=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("🔥 Correlation Heatmap")

with st.expander("Show Heatmap"):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    st.pyplot(fig)

# -----------------------------
# Missing Values Heatmap
# -----------------------------
st.subheader("⚠️ Missing Values")

with st.expander("Show Missing Values Heatmap"):
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(df.isnull(), cbar=False)
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Download Filtered Data
# -----------------------------
st.subheader("⬇️ Download Data")

csv_buffer = io.StringIO()
filtered_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv_buffer.getvalue(),
    file_name="filtered_data.csv",
    mime="text/csv"
)

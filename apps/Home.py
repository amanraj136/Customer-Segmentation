import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🧩 Customer Segmentation Dashboard")
st.markdown("### Understand your customers. Personalize your strategy. Grow your business.")
st.markdown("---")

# -----------------------------
# Load Data (Adjust Path)
# -----------------------------
df = pd.read_csv("apps/data/clustered_data.csv")    # Your clustered DF
num_clusters = df["Clusters"].nunique()
silhouette_score = 0.35424776915703293   # replace with your own variable later

# -----------------------------
# KPI Cards
# -----------------------------
total_customers = len(df)
highest_spending_cluster = (
    df.groupby("Clusters")["Spent"].mean().sort_values(ascending=False).index[0]
)
highest_spending_value = df.groupby("Clusters")["Spent"].mean().max()
total_revenue = df["Spent"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Number of Clusters", num_clusters)
col3.metric("Silhouette Score", round(silhouette_score, 3))
col4.metric(f"Top Spending Cluster", f"Cluster {highest_spending_cluster} (${int(highest_spending_value)})")

st.markdown("---")

# -----------------------------
# Cluster Distribution Chart
# -----------------------------
st.subheader("📊 Cluster Distribution")

cluster_counts = df["Clusters"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

fig = px.pie(
    cluster_counts,
    names="Cluster",
    values="Count",
    hole=0.45,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig.update_layout(
    title="Customer Distribution by Cluster",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# Cluster Personas (Placeholder)
# -----------------------------
st.subheader("🧑‍🤝‍🧑 Customer Personas by Cluster")

persona_container = st.container()

with persona_container:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        ### 🔵 Cluster 0 – Mid-Income Family Spenders  
        - Balanced spenders  
        - Regular buyers  
        - Moderate income  
        - Prefer wine + meat  
        """)

    with c2:
        st.markdown("""
        ### 🟠 Cluster 1 – Low-Income Minimal Shoppers  
        - Low spending  
        - Rarely engage  
        - Promotion sensitive  
        - Mostly essentials  
        """)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
        ### 🟣 Cluster 2 – Premium High-Value Customers  
        - Highest income  
        - Highest overall spending  
        - Loyal, stable  
        - Strong wine + meat preference  
        """)

    with c4:
        st.markdown("""
        ### 🟢 Cluster 3 – Young Budget Families  
        - Low–mid income  
        - Kids & teens  
        - Occasional buyers  
        - Good target for promos  
        """)

st.markdown("---")

# -----------------------------
# Executive Summary
# -----------------------------
st.subheader("📘 Executive Summary")

st.info("""
Overall, the segmentation reveals four distinct customer groups with 
different spending habits, demographics, and product preferences.  
Cluster 2 represents premium high-value customers, while Cluster 1 requires 
personalized promotions to increase engagement.  
Family-based clusters (0 & 3) respond well to category-specific offers.  
""")

st.markdown("---")

# -----------------------------
# PCA Preview
# -----------------------------

# Automatically find project root
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

ROOT = get_project_root()
IMG_PATH = os.path.join(ROOT, "figures", "3D_PCA_Clusters.png")

# st.write("Debug path:", IMG_PATH)  # optional
img = Image.open(IMG_PATH)
img = img.resize((900, 900))

st.subheader("🔍 Cluster Separation (PCA Preview)")

if os.path.exists(IMG_PATH):
    st.image(img, caption="3D PCA Cluster Projection")
else:
    st.error(f"Image not found at: {IMG_PATH}")


st.markdown("---")

# -----------------------------
# Navigation Buttons
# -----------------------------
cA, cB, cC = st.columns(3)

with cA:
    st.page_link("pages/Data_Explorer.py", label="📂 Explore Data", icon="📄")

with cB:
    st.page_link("pages/Cluster_Insights.py", label="📈 Cluster Insights", icon="📈")

with cC:
    st.page_link("pages/Predict_Segment.py", label="🎯 Predict Customer Segment", icon="🎯")

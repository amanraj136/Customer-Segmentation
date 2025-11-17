import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Cluster Insights", layout="wide")

st.title("📈 Cluster Insights & Segmentation Analysis")
st.markdown("### Deep-dive into clusters, spending patterns, and customer behavior.")
st.markdown("---")

# -----------------------------
# Load clustered data
# -----------------------------
df = pd.read_csv("apps/data/clustered_data.csv")

# Helpful variables
clusters = sorted(df["Clusters"].unique())
category_cols = ["Wines", "Fruits", "Meat", "Fish", "Sweets"]  # You can expand this list

# -----------------------------
# Cluster Selector
# -----------------------------
st.subheader("🎯 Select Cluster for Analysis")

selected_cluster = st.selectbox("Choose a cluster:", clusters)

cluster_df = df[df["Clusters"] == selected_cluster]

st.markdown("---")

# -----------------------------
# KPI Section
# -----------------------------
st.subheader(f"📊 Cluster {selected_cluster} Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", f"{len(cluster_df):,}")
col2.metric("Avg Income", f"${int(cluster_df['Income'].mean())}")
col3.metric("Avg Spend", f"${round(cluster_df['Spent'].mean(), 2)}")
col4.metric("Avg Customer Age", int(cluster_df["Age"].mean()))

st.markdown("---")

# -----------------------------
# Spending Comparison Across Clusters
# -----------------------------
st.subheader("💸 Average Spending by Cluster")

spend_df = df.groupby("Clusters")["Spent"].mean().reset_index()

fig_spend = px.bar(
    spend_df,
    x="Clusters",
    y="Spent",
    text_auto=True,
    color="Clusters",
    title="Cluster-wise Average Spending",
)

st.plotly_chart(fig_spend, use_container_width=True)

st.markdown("---")

# -----------------------------
# Radar Chart (Category Preferences)
# -----------------------------
st.subheader(f"🍽️ Product Category Preferences (Cluster {selected_cluster})")

radar_source = cluster_df[category_cols].mean()

# Normalize for radar
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[category_cols])
scaled_cluster = np.mean(scaled[df["Clusters"] == selected_cluster], axis=0)

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=scaled_cluster,
    theta=category_cols,
    fill='toself',
    name=f'Cluster {selected_cluster}'
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title=f"Category Preferences for Cluster {selected_cluster}"
)

st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# -----------------------------
# Interactive 3D PCA Plot
# -----------------------------
st.subheader("🧭 3D PCA Cluster Visualization")

st.info("This shows cluster separation in 3 dimensions using PCA reduction.")

pca_df = pd.read_csv("apps/data/pca_3d.csv")  # You will generate this file

fig_pca = px.scatter_3d(
    pca_df,
    x="col1",
    y="col2",
    z="col3",
    color="Clusters",
    title="3D PCA Cluster Projection",
    opacity=1,
    symbol="Clusters",
)

fig_pca.update_layout(
    scene=dict(
        xaxis=dict(gridcolor="white"),
        yaxis=dict(gridcolor="white"),
        zaxis=dict(gridcolor="white"),
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    width = 1000,
    height = 700,
    font_color="white",
    title_font=dict(size=22, color="white", family="Times new Roman Black"),
    legend=dict(
        x=0.9,
        y=0.7,
        xanchor="left",
        yanchor="top",
        font=dict(
            color="white",
            size=14,
            # family="Arial"
        )
    )
)


fig_pca.update_traces(marker=dict(size=5))

st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("---")

# -----------------------------
# Cluster Comparison
# -----------------------------
st.subheader("⚖️ Compare Two Clusters")

c1, c2 = st.columns(2)

with c1:
    comp_a = st.selectbox("Select First Cluster", clusters, key="A")

with c2:
    comp_b = st.selectbox("Select Second Cluster", clusters, key="B")

if comp_a != comp_b:
    comp_df = df[df["Clusters"].isin([comp_a, comp_b])]
    fig_compare = px.bar(
        comp_df.groupby("Clusters")[category_cols].mean().T,
        barmode="group",
        title=f"Category Comparison: Cluster {comp_a} vs Cluster {comp_b}"
    )
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.warning("Select two different clusters to compare.")

st.markdown("---")

# -----------------------------
# Business Recommendations Section
# -----------------------------
st.subheader("💡 Business Recommendations")

recommendations = {
    0: """
### 🔵 Cluster 0 – Mid-Income Family Spenders
- Launch **value combo packs** for Wines + Meat.
- Send **family-oriented festive campaigns**.
- Upsell premium categories slowly.
""",
    1: """
### 🟠 Cluster 1 – Low-Income Minimal Shoppers
- Use **heavy discounts and promotions**.
- Push **essentials** category.
- Offer **loyalty cashback**.
""",
    2: """
### 🟣 Cluster 2 – Premium High-Value Customers
- Push **premium luxury items**.
- Offer **VIP membership programs**.
- Personalized recommendations via email.
""",
    3: """
### 🟢 Cluster 3 – Young Budget Families
- Promote **kids & family bundles**.
- Use **email/SMS reminders**.
- Cross-sell mid-range product lines.
"""
}

st.markdown(recommendations[selected_cluster])

st.markdown("---")

# -----------------------------
# Export Insights
# -----------------------------
st.subheader("⬇️ Export Insights")

export_text = f"""
Cluster {selected_cluster} Insights\n
Customers: {len(cluster_df)}  
Average Spend: {cluster_df['Spent'].mean():.2f}  
Average Income: {cluster_df['Income'].mean():.2f}  

Category Preferences:
{cluster_df[category_cols].mean().to_string()}

Recommendations:
{recommendations[selected_cluster]}
"""

st.download_button(
    "Download Insights as TXT",
    export_text,
    file_name=f"cluster_{selected_cluster}_insights.txt"
)

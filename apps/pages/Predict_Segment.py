import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.metrics import pairwise_distances
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Predict Segment", layout="wide")
st.title("🎯 Predict Customer Segment")
st.markdown("Upload new customers or enter a single customer's details to predict their cluster and get recommendations.")
st.markdown("---")

# -----------------------------
# Load helper data & models
# -----------------------------
def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

ROOT = get_project_root()

# Paths (adjust if your structure is different)
SCALER_PATH = os.path.join(ROOT, "models", "Scaler.pkl")
PCA_PATH = os.path.join(ROOT, "models", "PCA.pkl")
KMEANS_PATH = os.path.join(ROOT, "models", "KMeans.pkl")
# FEATURES_PATH = "model/feature_columns.pkl"  # optional saved list of feature columns

# Load clustered dataframe (to get defaults / columns)
try:
    df_clustered = pd.read_csv("apps/data/clustered_data.csv")
except Exception:
    df_clustered = None

# Load models (graceful)
scaler = pca = kmeans = None
try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.warning(f"Scaler not found at {SCALER_PATH}. Single/batch predictions will not work until saved scaler is placed. ({e})")

try:
    pca = joblib.load(PCA_PATH)
except Exception as e:
    st.warning(f"PCA not found at {PCA_PATH}. PCA-transform based prediction will not work. ({e})")

try:
    kmeans = joblib.load(KMEANS_PATH)
except Exception as e:
    st.warning(f"KMeans not found at {KMEANS_PATH}. Prediction will not work. ({e})")

# Get feature columns expected by the saved scaler
feature_columns = None
if scaler is not None:
    # try to load feature list
    try:
        feature_columns = joblib.load(FEATURES_PATH)
    except Exception:
        if df_clustered is not None:
            # Reconstruct expected columns by removing promotion/target cols if present
            cols_del = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5', 'Complain', 'Response']
            # Use the DF columns if available
            feature_columns = [c for c in df_clustered.columns if c not in cols_del + ["Clusters", "Total_Promos"]]
        else:
            feature_columns = None

# Helpful message
if feature_columns is None:
    st.error("Feature column names could not be determined. Please create 'model/feature_columns.pkl' or ensure 'data/processed/clustered_data.csv' exists.")
    st.stop()

# -----------------------------
# Personas & Recommendations (reuse from insights)
# -----------------------------
PERSONAS = {
    0: {
        "title": "🔵 Cluster 0 – Mid-Income Family Spenders",
        "summary": "Balanced family buyers: moderate income, regular spending, prefer wine & meat.",
        "reco": ["Value combo packs (Wine + Meat)", "Family-oriented festive campaigns", "Slow upsell to premium"]
    },
    1: {
        "title": "🟠 Cluster 1 – Low-Income Minimal Shoppers",
        "summary": "Price-sensitive households: low spend, promotion responsive for essentials.",
        "reco": ["Heavy discounts and promotions", "Essentials-focused bundles", "Loyalty cashback offers"]
    },
    2: {
        "title": "🟣 Cluster 2 – Premium High-Value Customers",
        "summary": "Affluent individuals: very high spending, luxury preference, highly responsive.",
        "reco": ["Premium product lines", "VIP memberships", "Personalized high-touch recommendations"]
    },
    3: {
        "title": "🟢 Cluster 3 – Young Budget Families",
        "summary": "Young families with kids/teens: moderate spend, good target for promos and bundles.",
        "reco": ["Kids & family bundles", "Email/SMS reminders", "Cross-sell mid-range lines"]
    }
}

# -----------------------------
# Utility functions
# -----------------------------
def preprocess_dataframe(raw_df):
    """
    Accepts a raw dataframe (with same column names as feature_columns or a superset)
    Returns transformed array ready for PCA/KMeans prediction.
    """
    # Ensure all expected features exist
    df = raw_df.copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # default missing numeric columns to 0

    X = df[feature_columns].astype(float)
    # scale
    if scaler is None:
        raise RuntimeError("Scaler not loaded.")
    Xs = scaler.transform(X)
    # pca
    if pca is None:
        raise RuntimeError("PCA not loaded.")
    Xp = pca.transform(Xs)
    return X, Xs, Xp  # return original numeric df, scaled, and pca-transformed

def predict_clusters_from_pca(Xp):
    """Return cluster labels and distances to cluster centers (Euclidean)."""
    if kmeans is None:
        raise RuntimeError("KMeans not loaded.")
    labels = kmeans.predict(Xp)
    centers = kmeans.cluster_centers_
    dists = pairwise_distances(Xp, centers, metric="euclidean")  # shape (n_samples, n_clusters)
    min_dist = np.min(dists, axis=1)
    dist_to_assigned = dists[np.arange(len(labels)), labels]
    return labels, dist_to_assigned, dists

def single_prediction(input_dict):
    """
    Input: dict mapping feature_columns -> value (as numeric)
    Returns: dict with prediction details
    """
    single_df = pd.DataFrame([input_dict], columns=feature_columns)
    _, _, Xp = preprocess_dataframe(single_df)
    labels, dist_to_assigned, all_dists = predict_clusters_from_pca(Xp)
    assigned = int(labels[0])
    return {
        "cluster": assigned,
        "distance": float(dist_to_assigned[0]),
        "all_distances": all_dists[0]
    }

# -----------------------------
# Sidebar: quick tips and model info
# -----------------------------
st.sidebar.header("Model Info & Tools")
st.sidebar.write(f"Loaded features: {len(feature_columns)}")
st.sidebar.write(f"Scaler: {'✅' if scaler is not None else '❌'}")
st.sidebar.write(f"PCA: {'✅' if pca is not None else '❌'}")
st.sidebar.write(f"KMeans: {'✅' if kmeans is not None else '❌'}")

st.sidebar.markdown("---")
st.sidebar.info("Upload a CSV with columns matching the expected features (or a superset). Missing numeric columns will be filled with 0.")

# -----------------------------
# SECTION 1: Single Customer Prediction (What-if)
# -----------------------------
st.header("Predict for a Single Customer (What-If)")
st.markdown("Use sliders and inputs to craft a customer profile and predict which cluster they belong to.")

# Prepare sensible defaults: use dataset medians if available
defaults = {}
if df_clustered is not None:
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df_clustered[col]):
            defaults[col] = float(df_clustered[col].median())
        else:
            defaults[col] = 0.0
else:
    defaults = {c: 0.0 for c in feature_columns}

# Show a compact form with important features exposed; hide the rest in "advanced"
important_features = ["Income", "Age", "Recency", "Wines", "Meat", "Fish", "Fruits", "Sweets", "Kidhome", "Teenhome"]
visible = [f for f in important_features if f in feature_columns]
hidden = [f for f in feature_columns if f not in visible]

cols = st.columns(2)
input_vals = {}
with cols[0]:
    for f in visible[:len(visible)//2 + 1]:
        v = st.number_input(f"{f}", value=float(defaults.get(f, 0.0)), format="%.2f")
        input_vals[f] = v
with cols[1]:
    for f in visible[len(visible)//2 + 1:]:
        v = st.number_input(f"{f}", value=float(defaults.get(f, 0.0)), format="%.2f")
        input_vals[f] = v

with st.expander("Advanced features (you can leave defaults)"):
    for f in hidden:
        v = st.number_input(f"{f}", value=float(defaults.get(f, 0.0)), format="%.2f")
        input_vals[f] = v

if st.button("Predict Single Customer"):
    try:
        pred = single_prediction(input_vals)
        cluster_id = pred["cluster"]
        dist = pred["distance"]
        persona = PERSONAS.get(cluster_id, {})
        st.success(f"Assigned to Cluster {cluster_id}")
        st.write(f"**Persona:** {persona.get('title','')}")
        st.write(persona.get("summary",""))
        st.write("**Top recommendations:**")
        for r in persona.get("reco", []):
            st.write(f"- {r}")
        st.write(f"Distance to cluster center: {dist:.4f}")
        # Offer download of the single-customer summary
        summary_txt = f"Cluster: {cluster_id}\nPersona: {persona.get('title','')}\nSummary: {persona.get('summary','')}\nDistance: {dist:.4f}\nRecommendations:\n" + "\n".join(persona.get("reco", []))
        st.download_button("Download Prediction Summary (TXT)", summary_txt, file_name=f"prediction_cluster_{cluster_id}.txt")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# -----------------------------
# SECTION 2: Batch Prediction (CSV Upload)
# -----------------------------
st.header("Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV file with customer rows", type=["csv"])
if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(raw.head(), use_container_width=True)

        # Preprocess and predict
        try:
            X_raw, X_scaled, X_pca = preprocess_dataframe(raw)
            labels, dist_to_assigned, all_dists = predict_clusters_from_pca(X_pca)
            raw_out = raw.copy()
            raw_out["Predicted_Cluster"] = labels
            raw_out["Distance_to_Assigned"] = dist_to_assigned
            # Attach persona summary
            raw_out["Persona_Title"] = raw_out["Predicted_Cluster"].map(lambda x: PERSONAS.get(int(x), {}).get("title", ""))
            raw_out["Persona_Summary"] = raw_out["Predicted_Cluster"].map(lambda x: PERSONAS.get(int(x), {}).get("summary", ""))
            st.success(f"Predicted clusters for {len(raw_out)} rows.")
            st.dataframe(raw_out.head(), use_container_width=True)

            # Download labeled CSV
            csv_buf = io.StringIO()
            raw_out.to_csv(csv_buf, index=False)
            st.download_button(
                "Download Labeled CSV",
                csv_buf.getvalue(),
                file_name="labeled_customers.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed to preprocess/predict uploaded file: {e}")

    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")

st.markdown("---")

# -----------------------------
# SECTION 3: Quick Utilities
# -----------------------------
st.header("Utilities")

if st.button("Show cluster centers (PCA space)"):
    if kmeans is None:
        st.error("KMeans not loaded.")
    else:
        centers = kmeans.cluster_centers_
        st.write("Cluster centers (in PCA space):")
        st.write(pd.DataFrame(centers, columns=[f"PC{i+1}" for i in range(centers.shape[1])]))

if st.button("Show feature columns used by the model"):
    st.write(feature_columns)

st.markdown("---")
st.info("Note: this page expects the saved scaler, PCA and KMeans to match the columns in your processed clustered data. If predictions look odd, please verify feature order and preprocessing pipeline.")

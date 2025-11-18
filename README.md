# Customer Segmentation — Streamlit App + Machine Learning


🔗 **Live Demo:**  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-segmentation-kzd5ejdp5lv2epdtjfrgv6.streamlit.app/)

---


## Overview

This project performs **customer segmentation** on a real-world marketing dataset to uncover meaningful customer groups, spending patterns, and behavioural personas.  

Using a combination of **feature engineering**, **Principal Component Analysis (PCA)**, and **KMeans clustering**, the system identifies four distinct customer segments that enable personalized marketing strategies.

The Streamlit application provides:
- A **home dashboard** with KPIs & cluster summaries  
- An interactive **Data Explorer**  
- **Cluster Insights** including radar charts & PCA 3D visualization  
- A **Prediction module** for single or batch customer inputs using saved ML models  

This end-to-end system represents a complete data science workflow — from raw data to an interactive, business-ready segmentation tool.

---

## Dataset Summary

- **Source:** marketing_campaign.csv  
- **Shape:** ~2,240 rows × multiple demographic & spending features  
- **Contains:**  
	- Demographics: `Age`, `Marital_Status`, `Education`, `Income`  
	- Family indicators: `Kidhome`, `Teenhome`, `Family_Size`  
	- Purchase behavior: `Wines`, `Fruits`, `Meat`, `Fish`, `Sweets`, `Spent`  
	- Campaign responses & recency  
- **Objective:** Identify meaningful customer segments for strategic marketing  

---

## Methodology

### **Preprocessing**
- Handling missing values  
- Encoding categorical variables  
- Standardization  
- Outlier treatment  
- Feature aggregation (e.g., total spend, family size)


### **Dimensionality Reduction**
- **PCA (3 Components)**  
		Used for cluster visualization and model training pipeline  


### **Clustering Algorithms Evaluated**
- **KMeans (final model)**  
- Agglomerative Clustering (hierarchical validation)


### **Evaluation Metrics**
- **Silhouette Score:** ~0.35  
- Intra-cluster cohesion & inter-cluster separation  
- Visual validation using 3D PCA plots  


### **Persona & Insights Generation**
Each cluster is assigned a business-friendly persona based on:
- Spending patterns  
- Income range  
- Product category preferences  
- Engagement level (recency, purchase variety)


---


## Key Results

- **Final Clusters:** 4  
- **Cluster Personas:**  
 - **Cluster 0 — Mid-Income Family Spenders**  
 - **Cluster 1 — Low-Income Minimal Shoppers**  
 - **Cluster 2 — Premium High-Value Customers**  
 - **Cluster 3 — Young Budget Families**


- **Important Figures (in `/figures`):**  
 - `3D_PCA_Clusters.png`  
 - Category preference radar charts  
 - Spending distribution comparisons  
 - Heatmaps & correlation visualizations  

These insights help businesses tailor campaigns, promotions, and product recommendations more effectively.

---

## How to Run Locally

1. **Python 3.10+**
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ---
3. Ensure the trained model files are available in:
	models/Scaler.pkl
	models/PCA.pkl
	models/KMeans.pkl
4. Start the Streamlit app:
	Option A: Double-Click `app.bat` to launch the app in your browser.
	Option B: `streamlit run apps/Home.py` in the terminal.

---

## Files
- **`apps/`**
	- *`data/`*
		- clustered_data.csv
		- pca_3d.csv
- **`pages/`**
	- `Home.py` — main dashboard
	- `Data_Explorer.py` — dataset exploration
	- `Cluster_Insights.py` — analysis & visualizations
	- `Predict_Segment.py` — prediction interface
- **`models/`**
 	- `Scaler.pkl`
 	- `PCA.pkl`
 	- `KMeans.pkl`
 	- `Agglomerative.pkl`
	- `Label_Encoder.pkl`

- **`figures/`** - Contains the figures of PCA Visualization, Cluster Plots, Category Preference charts, etc.
- **`dataset/`** — use `narketing_campaign.csv` inside the dataset folder.
- **`notebook/`**
 	- `Customer-Segmentation.ipynb` — full EDA + clustering workflow
	- `Customer-Segmentation.pdf` — exported analysis

---

## Why This Project Matters

**Customer segmentation is widely used in:**
- Personalized marketing & recommendations
- Customer lifetime value optimization
- Campaign targeting for better ROI
- Building loyalty programs and personas

---

## Future Enhancements

- Add DB storage for predictions  
- Train advanced models (Gaussian Mixture Models, DBSCAN)  
- Build API endpoint using FastAPI  
- Add user authentication on Streamlit  
- Deploy a Dockerized version  

---

## Notes
- The project is modular and production-friendly.
- New visualizations, segments, and deployment enhancements may be added.
- Ideal for marketing analytics, targeted campaigns, and customer profiling.

---

## Changelog

**v1.0.0 — Initial Release**
- Added KMeans segmentation pipeline
- Added full Streamlit multi-page app
- Added PCA visualization and personas
- Added prediction module

---

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software for both personal and commercial purposes.

See the full license in the \[LICENSE](LICENSE) file.

---




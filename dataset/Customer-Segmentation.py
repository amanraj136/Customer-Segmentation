#!/usr/bin/env python
# coding: utf-8

# ## 1. Importing Libraries

# In[571]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt 
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score
from sklearn import metrics
import warnings
import sys
import joblib


# In[572]:


if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)


# ## 2. Load Dataset

# In[573]:


df = pd.read_csv("dataset/marketing_campaign.csv", sep = "\t")


# ## 3. Explore Dataset

# In[574]:


print("Rows, Columns")
df.shape


# In[575]:


df.head()


# In[576]:


df.describe()


# In[577]:


df.info()


# In[578]:


df.isnull().sum()


# ## 4. Dropping NaN Rows

# In[579]:


df = df.dropna()
print("Total Number of data-points(Rows) after removing rows with NaN values:", len(df))


# ## 5. Data Cleaning

# #### 5.1 Data-Time Encoding

# In[580]:


df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst = True, errors = 'coerce')
dates = []
for i in df["Dt_Customer"]:
    i = i.date()
    dates.append(i)

print("Newsest Customer's enrolment date in records:", max(dates))
print("Oldest Customer's enrolment date in records:", min(dates))


# #### 5.2. Feature Engineering

# In[581]:


# Creating a feature ""Customer_For
days = []
d1 = max(dates)

for i in dates:
    delta = d1 - i
    days.append(delta)

df["Customer_For"] = days
df["Customer_For"] = pd.to_numeric(df["Customer_For"], errors = "coerce")


# In[582]:


print("Total categories in the feature Marital_Status:\n", df["Marital_Status"].value_counts(), "\n")


# In[583]:


print("Total categories in the feature Education:\n", df["Education"].value_counts())


# In[584]:


# Age of Customer today
df["Age"] = 2025-df["Year_Birth"]


# In[585]:


# Total spendings on various items
df["Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]


# In[586]:


# Deriving living situation by marital status "Alone"
df["Living_With"] = df["Marital_Status"].replace(
    {
        "Married": "Partner",
        "Together": "Partner",
         "Absurd": "Alone",
         "Widow": "Alone",
         "YOLO": "Alone",
         "Divorced": "Alone",
         "Single": "Alone"
    }
)


# In[587]:


# Feature indicating total children living in the household
df["Children"] = df["Kidhome"] + df["Teenhome"]


# In[588]:


# Feature for total memeber in the household
df["Family_Size"] = df["Living_With"].replace(
    {
        "Alone": 1,
        "Partner": 2
    }
) + df["Children"]


# In[589]:


# Feature pertaining parenthood
df["Is_Parent"] = np.where(df.Children > 0, 1, 0)


# In[590]:


# Segmenting education levels in three groups
df["Education"] = df["Education"].replace(
    {
        "Basic": "Undergraduate",
        "2n Cycle": "Undegraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate",
        "Phd": "Postgraduate"
    }
)


# In[591]:


# For clarity
df = df.rename(columns = 
               {
                   "MntWines": "Wines",
                   "MntFruits": "Fruits",
                   "MntMeatProducts": "Meat",
                   "MntFishProducts": "Fish",
                   "MntSweetProducts": "Sweets",
                   "MntGoldProd": "Gold",
               }
              )


# In[592]:


# Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
df = df.drop(to_drop, axis = 1)


# In[593]:


df.describe()


# #### 5.3 Plotting Selected Features

# In[594]:


# Setting up colors prefrences
sns.set(rc = 
       {
           "axes.facecolor": "#FFF9ED",
           "figure.facecolor": "#FFF9ED",
       })
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])


# In[595]:


# Plotting Features
To_plot = ["Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
print("Reletive Plot of Selected Features: A Data Subnet")
plt.figure()
sns.pairplot(df[To_plot], hue = "Is_Parent", palette = (["#682F2F","#F3AB60"]))
plt.savefig("figures/Reletive_Plot_of_Selected_Feature.png", dpi = 300, bbox_inches = "tight")
plt.show()


# #### 5.4 Removing Outliers

# In[596]:


# Dropping outlieres by selecting a cap on Age and Income
df = df[(df["Age"] < 90)]
df = df[(df["Income"] < 600000)]
print("Total Number of data-points after removing the outliers are:", len(df))


# #### 5.5 Correlation Matrix

# In[597]:


corrmat = df.select_dtypes(include = ['number']).corr()
plt.figure(figsize = (20, 20))
sns.heatmap(corrmat, annot = True, cmap = cmap, center = 0)
plt.savefig("figures/Correlation_Matrix.png", dpi = 300, bbox_inches = "tight")
plt.show()


# ## 6. Data Preprocessing

# In[598]:


# Listing the Categorical variables
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical Variables in dataset:", object_cols)


# #### 6.1 Label Encoding

# In[600]:


# from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in object_cols:
    LE = LabelEncoder()
    df[col] = LE.fit_transform(df[col])
    label_encoders[col] = LE    # store encoder for deployment

print("All features are now numerical Data Type:")
df.info()


# In[601]:


# # Label Encoding the object types
# LE = LabelEncoder()
# for i in object_cols:
#     df[i] = df[[i]].apply(LE.fit_transform)

# print("All features are now numerical Data Type:")
# df.info()


# In[602]:


# Creating a copy of data
ds = df.copy()
# Creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5', 'Complain', 'Response']
ds = ds.drop(cols_del, axis = 1)


# #### 6.2 Scaling the Data 

# In[603]:


# Scaling Data
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds), columns = ds.columns)
scaler


# In[604]:


# Scaled data to be used for reducing the Dimesionality
print("DataFrame to beused for further modelling:")
scaled_ds.head()


# ## 7. Dimensionality Reduction

# #### 7.1 Dimenionality Reduction with PCA

# In[605]:


# Initiating PCA to reduce dimentions aka fetures to 3
pca = PCA(n_components = 3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns = (["col1", "col2", "col3"]))
PCA_ds.describe()
pca


# #### 7.2 Plotting the reduced dataframe

# In[606]:


# 3D Projection of Data
x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]

fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(x, y, z, c = "maroon", marker = "o")
ax.set_title("3D Projection of Data In The Reduced Dimension")
plt.savefig("figures/3D_Projection_of_Data_In_The_Reduced_Dimension.png", dpi = 300, bbox_inches = "tight")
plt.show()


# ## 8. Clustering 

# #### 8.1 Elbow Method

# In[607]:


# Quick Examination of Elbow-Method to find the number of clusters to make.
print("Elbow-Method to determine the number of clusters to be formed:")
Elbow_M = KElbowVisualizer(KMeans(), k = 10)
Elbow_M.fit(PCA_ds)
plt.savefig("figures/Elbow-Method_to_determine_the_number_of_clusters_to_be_formed.png", dpi = 300, bbox_inches = "tight")
Elbow_M.show()


# #### 8.2 Agglomerative Clustering

# In[608]:


# Initiating the Agglomerative Clustering model
AC = AgglomerativeClustering(n_clusters = 4)
AC


# In[609]:


# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC


# In[610]:


# Add the Clusters feature to the original DataFrame
df["Clusters"] = yhat_AC


# #### 8.3 Plotting the Clusters

# In[611]:


fig = plt.figure(figsize = (10, 8))
ax = plt.subplot(111, projection = '3d', label = "bla")
ax.scatter(x, y, z, s = 40, c = PCA_ds["Clusters"], marker = 'o', cmap = cmap)
ax.set_title("The Plot of Clusters")
plt.savefig("figures/The_Plot_of_Clusters.png", dpi = 300, bbox_inches = "tight")
plt.show()


# ## 9. Evaluating Models 

# In[612]:


# Plotting countplot of Clusters
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x = df["Clusters"], palette = pal)
pl.set_title("Distribution Of The Clusters")
plt.savefig("figures/Distribution_Of_The_Clusters.png", dpi = 300, bbox_inches = "tight")
plt.show()


# In[613]:


pl = sns.scatterplot(data = df, x = df["Spent"], y = df["Income"], hue = df["Clusters"], palette = pal)
pl.set_title("Cluster's Profile Based on Income and Spending")
plt.savefig("figures/Cluster's_Profile_Based_on_Income_and_Spending.png", dpi = 300, bbox_inches = "tight")
plt.legend()
plt.show()


# In[614]:


plt.figure()
pl = sns.swarmplot(x = df["Clusters"], y = df["Spent"], color = "#CBEDDD", alpha = 0.5)
pl = sns.boxenplot(x = df["Clusters"], y = df["Spent"], palette = pal)
pl.set_title("Box Plot of Cluster based on Spent")
plt.savefig("figures/Box_Plot_of_Cluster_based_on_Spent.png", dpi = 300, bbox_inches = "tight")
plt.show()


# In[615]:


# Creating a feature to get a sum of accepted promotions
df["Total_Promos"] = df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] + df["AcceptedCmp4"] + df["AcceptedCmp5"] 


# In[616]:


# Plotting count of total campaign accepted
plt.figure()
pl = sns.countplot(x = df["Total_Promos"], hue = df["Clusters"], palette = pal)
pl.set_title("Count of Promotion Accepted")
pl.set_xlabel("Number of Total Accepted Promotions")
plt.savefig("figures/Count_of_Promotion_Accepted", dpi = 300, bbox_inches = "tight")
plt.show()


# In[617]:


# Plotting the number of deals purchase
plt.figure()
pl = sns.boxenplot(y = df["NumDealsPurchases"], x = df["Clusters"], palette = pal)
pl.set_title("Number of Deal Purchased")
plt.savefig("figures/Number_of_Deal_Purchased", dpi = 300, bbox_inches = "tight")
plt.show()


# ### 10. Profiling

# In[618]:


Personal = ["Kidhome","Teenhome","Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]
title = ["Spent vs KidHome", "Spent vs TeenHome", "Spent vs Customer_For", "Spent vs Age", "Spent vs Children", "Spent vs Family_Size", "Spent vs Is_Parent",
         "Spent vs Education", "Spent vs Living_With"]
for i, t in zip(Personal, title):
    plt.figure(figsize = (15, 8))
    sns.jointplot(x = df[i], y = df["Spent"], hue = df["Clusters"], kind = "kde", palette = pal)
    plt.savefig(f"figures/{t}", dpi = 300, bbox_inches = "tight")
    plt.show()


# ## 11. Silhouette Score

# In[619]:


# Mean of the Clusters
df.groupby('Clusters').mean()


# #### 11.1 Caculating Silhouette Score

# In[620]:


# Silhouette Score for Current Clustering
sil_score = silhouette_score(PCA_ds[["col1", "col2", "col3"]], df["Clusters"])
print("Silhouette Score:", sil_score)


# #### 11.2 Comparing Silhouete Scores for K=2 to k=10

# In[621]:


silhouette_scores = []
for k in range(2, 11):
    kmeans_test = KMeans(n_clusters = k, random_state = 42)
    kmeans_test.fit(PCA_ds[["col1", "col2", "col3"]])
    score = silhouette_score(PCA_ds[["col1", "col2", "col3"]], kmeans_test.labels_)
    silhouette_scores.append(score)


# In[622]:


print("\nSilhouette Scores for K=2 to K=10:")
for k, score in enumerate(silhouette_scores, start = 2):
    print(f"K={k}:{score}")


# #### 11.3 Plotting Silhouette Curve

# In[623]:


plt.figure(figsize = (10, 6))
plt.plot(range(2, 11), silhouette_scores, marker = 'o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.grid(True)
plt.savefig("figures/Silhouette_Score_vs_K.png", dpi=300, bbox_inches="tight")
plt.show()


# #### 11.4  3D PCA Cluster Separation - Visualizatioin

# In[624]:


fig = plt.figure(figsize = (10, 11))
ax = fig.add_subplot(111, projection = '3d')
scatter = ax.scatter(PCA_ds["col1"], PCA_ds["col2"], PCA_ds["col3"], c = df["Clusters"], cmap = cmap, s = 40, alpha = 0.8)
ax.set_title("3D PCA Visualization of Customer Clusters", fontsize = 14)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
legend = ax.legend(*scatter.legend_elements(), title = "Clusters", loc = "upper right")
plt.tight_layout()
plt.savefig("figures/3D_PCA_Clusters.png", dpi=300, bbox_inches="tight")
plt.show()


# #### 11.5 Average Spending Per Cluster Plot

# In[625]:


cluster_spending = df.groupby("Clusters")["Spent"].mean()
print(f"Cluster Spending:\n{cluster_spending}")


# In[626]:


plt.figure(figsize = (8, 6))
plt.bar(cluster_spending.index, cluster_spending.values, color = pal)
plt.xlabel("Cluster")
plt.ylabel("Average Spending")
plt.title("Average Spending by Cluster")
plt.savefig("figures/Average_Spending_by_Cluster.png", dpi=300, bbox_inches="tight")
plt.show()


# #### 11.6 Final Kmeans model

# In[627]:


kmeans_final = KMeans(n_clusters=4, random_state=42)
kmeans_final.fit(PCA_ds[["col1", "col2", "col3"]])
kmeans_final


# ## 12. Saving Models 

# In[645]:


models = {
    'Scaler' : scaler,
    'PCA': pca,
    'KMeans' : kmeans_final,
    'Agglomerative' : AC,
    'Label_Encoder' : label_encoders
}


# In[647]:


for path, model in models.items():
    joblib.dump(model, f"models/{path}.pkl")
    print(f"{path} Model Save successfully at - models/{path}.pkl")







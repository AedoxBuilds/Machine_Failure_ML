import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==========================================
# 1. Load Preprocessed Data
# ==========================================

data_file = "preprocessed_data.csv"
X = pd.read_csv(data_file)
print("Preprocessed data shape:", X.shape)


# ==========================================
# 2. Create KMeans Model (2 clusters)
# ==========================================

model = KMeans(n_clusters=2, random_state=42, n_init=10)


# ==========================================
# 3. Train (fit) the Model
# ==========================================

model.fit(X)


# ==========================================
# 4. Evaluate
# ==========================================

labels = model.labels_
print("\nCluster sizes:")
print(pd.Series(labels).value_counts().sort_index())

print("\nInertia:", round(model.inertia_, 2))
print("Silhouette score:", round(silhouette_score(X, labels), 4))
# Silhouette ranges from -1 to 1; closer to 1 means better-separated clusters


# ==========================================
# 5. Save Model
# ==========================================

joblib.dump(model, "kmeans_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
print("\nModel saved to kmeans_model.pkl")
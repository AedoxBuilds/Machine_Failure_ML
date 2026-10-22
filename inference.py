import joblib
import pandas as pd


# ==========================================
# 1. Load Saved Model, Scaler and Column Info
# ==========================================

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
numeric_columns = joblib.load("numeric_columns.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ==========================================
# 2. Create New Data
# ==========================================

new_data = pd.DataFrame({
    "Temperature":       [45.0, 70.5],
    "Pressure":          [150.0, 420.0],
    "Vibration_Level":   [4.2, 8.1],
    "Humidity":          [55.0, 80.0],
    "Power_Consumption": [40.0, 90.0],
})


# ==========================================
# 3. Preprocess New Data (same steps as training)
# ==========================================

new_data = new_data.reindex(columns=feature_columns)       # same columns, same order
new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])


# ==========================================
# 4. Predict Cluster
# ==========================================

clusters = model.predict(new_data)

result = pd.DataFrame({"Predicted_Cluster": clusters})
print(result)
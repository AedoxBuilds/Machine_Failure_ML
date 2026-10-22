import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


# ==========================================
# 1. Load Saved Model, Scaler and Column Info (once, at startup)
# ==========================================

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
numeric_columns = joblib.load("numeric_columns.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ==========================================
# 2. Home Page + Prediction
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            # Read the form values the user typed
            values = {col: request.form[col] for col in numeric_columns}
            new_data = pd.DataFrame([{col: float(val) for col, val in values.items()}])

            # Preprocess exactly like training
            new_data = new_data.reindex(columns=feature_columns)
            new_data[numeric_columns] = scaler.transform(new_data[numeric_columns])

            # Predict cluster
            result = int(model.predict(new_data)[0])

        except ValueError:
            error = "Please enter a valid number in every field."

    return render_template(
        "index.html",
        columns=numeric_columns,
        values=values,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
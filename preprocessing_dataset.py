import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ==========================================
# 0. Settings (only part you edit)
# ==========================================

file_name = "machine_failure.csv"            # <-- your CSV file name/path
output_file = "preprocessed_data.csv"        # <-- where the cleaned data is saved
columns_to_drop = ["Machine_ID", "Timestamp", "Failure_Status"]   # <-- IDs / labels you don't want to cluster on


# ==========================================
# 1. Load ANY CSV file
# ==========================================

df = pd.read_csv(file_name)
print("Original Data:")
print(df.head())
print("Shape:", df.shape)

print("########################################")


# ==========================================
# 2. Remove Duplicate Rows
# ==========================================

before_rows = df.shape[0]
df = df.drop_duplicates().reset_index(drop=True)
after_rows = df.shape[0]

print(f"\nDuplicates removed: {before_rows - after_rows}")
# Duplicate rows can bias the model, so we drop exact duplicate records


# ==========================================
# 3. Missing Values Before Cleaning
# ==========================================

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())


# ==========================================
# 4. Create Feature Set (no target in clustering)
# ==========================================

# Clustering is unsupervised, so there is no label column to separate.
# errors="ignore" keeps the script working even if a listed column is absent.
X = df.drop(columns=columns_to_drop, errors="ignore").copy()


# ==========================================
# 5. Automatically Find Column Types
# ==========================================

numeric_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(exclude="number").columns

print("\nNumeric Columns:")
print(list(numeric_columns))

print("\nCategorical Columns:")
print(list(categorical_columns))


# ==========================================
# 6. Handle Missing Values
# ==========================================

# Numeric columns -> fill with mean
for column in numeric_columns:
    X[column] = X[column].fillna(X[column].mean())

# Categorical columns -> fill with most frequent value (mode)
for column in categorical_columns:
    if X[column].mode().empty:               # column is entirely empty
        X[column] = X[column].fillna("Unknown")
    else:
        X[column] = X[column].fillna(X[column].mode()[0])

print("\nMissing values AFTER cleaning:")
print(X.isnull().sum())


# ==========================================
# 7. Encode Categorical Columns
# ==========================================

# KMeans only understands numbers, so categories become 0/1 columns
if len(categorical_columns) > 0:
    X = pd.get_dummies(X, columns=list(categorical_columns), dtype=int)

print("\nShape after encoding:", X.shape)


# ==========================================
# 8. Scale Numeric Columns
# ==========================================

# KMeans is distance-based, so features must be on the same scale
scaler = StandardScaler()
if len(numeric_columns) > 0:
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])


# ==========================================
# 9. Save Preprocessed Data
# ==========================================

X.to_csv(output_file, index=False)
print(f"\nPreprocessed data saved to {output_file}")
print("Final shape:", X.shape)


# ==========================================
# 10. Save Scaler (needed to preprocess new data later)
# ==========================================

joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(numeric_columns), "numeric_columns.pkl")
print("Scaler saved to scaler.pkl")
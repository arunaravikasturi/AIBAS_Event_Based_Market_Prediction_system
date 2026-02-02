import pandas as pd
import tensorflow as tf
import joblib
import statsmodels.api as sm
import os

# =========================
# Paths
# =========================
ACTIVATION_PATH = "/tmp/activation/activation_data.csv"

ANN_MODEL_PATH  = "/tmp/knowledge/models/currentAiSolution.h5"
OLS_MODEL_PATH  = "/tmp/knowledge/models/currentOlsSolution.pkl"
OLS_FEATURES_PATH = "/tmp/knowledge/models/ols_features.pkl"
SCALER_PATH     = "/tmp/knowledge/models/scaler.pkl"

OUTPUT_PATH     = "/tmp/output/predictions.csv"

# =========================
# Load activation data
# =========================
df = pd.read_csv(ACTIVATION_PATH)

# =========================
# Feature engineering (MATCH TRAINING)
# =========================
df["Date"] = pd.to_datetime(df["Date"])
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"]   = df["Date"].dt.day
df.drop(columns=["Date", "Close_Price"], inplace=True, errors="ignore")

# =========================
# Load models & artifacts
# =========================
scaler       = joblib.load(SCALER_PATH)
ann          = tf.keras.models.load_model(ANN_MODEL_PATH)
ols          = joblib.load(OLS_MODEL_PATH)
ols_features = joblib.load(OLS_FEATURES_PATH)

# =========================
# ANN prediction
# =========================
ann_features = list(scaler.feature_names_in_)
X_ann = df[ann_features]

X_ann_scaled = scaler.transform(X_ann)
X_ann_scaled = X_ann_scaled[:, :ann.input_shape[1]]

ann_pred = ann.predict(X_ann_scaled).flatten()[0]

# =========================
# OLS prediction (STRICT ALIGNMENT)
# =========================
ols_feature_cols = [f for f in ols_features if f != "const"]

X_ols = df.reindex(columns=ols_feature_cols, fill_value=0)
X_ols = sm.add_constant(X_ols, has_constant="add")
X_ols = X_ols[ols_features].astype(float)

ols_pred = ols.predict(X_ols).values.flatten()[0]

# =========================
# Save output
# =========================
os.makedirs("/tmp/output", exist_ok=True)

out = pd.DataFrame({
    "ANN_Prediction": [ann_pred],
    "OLS_Prediction": [ols_pred]
})

out.to_csv(OUTPUT_PATH, index=False)

print("=== Event-Based Market Prediction ===")
print(out)

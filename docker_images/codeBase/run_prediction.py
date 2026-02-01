import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
import joblib
import os

# =========================
# Paths inside containers
# =========================
ACTIVATION_PATH = "/tmp/activation/activation_data.csv"
ANN_MODEL_PATH  = "/tmp/knowledge/models/currentAiSolution.h5"
OLS_MODEL_PATH  = "/tmp/knowledge/models/currentOlsSolution.pkl"
SCALER_PATH     = "/tmp/knowledge/models/scaler.pkl"
OUTPUT_PATH     = "/tmp/output/predictions.csv"

# =========================
# Load activation data
# =========================
activation_df = pd.read_csv(ACTIVATION_PATH)

# =========================
# Load scaler and models
# =========================
scaler = joblib.load(SCALER_PATH)
ann = tf.keras.models.load_model(ANN_MODEL_PATH)
ols = joblib.load(OLS_MODEL_PATH)

# =========================
# Ensure required time features exist
# (used during training)
# =========================
if "Date" in activation_df.columns:
    activation_df["Date"] = pd.to_datetime(activation_df["Date"])

    if "Year" not in activation_df.columns:
        activation_df["Year"] = activation_df["Date"].dt.year
    if "Month" not in activation_df.columns:
        activation_df["Month"] = activation_df["Date"].dt.month
    if "Day" not in activation_df.columns:
        activation_df["Day"] = activation_df["Date"].dt.day

# =========================
# Select EXACT training features
# =========================
FEATURES = list(scaler.feature_names_in_)
X = activation_df[FEATURES]

# =========================
# Scale features
# =========================
X_scaled_full = scaler.transform(X)

# =========================
# Align ANN input dimension
# (ANN trained on fewer features than scaler)
# =========================
ann_input_dim = ann.input_shape[1]
X_scaled_ann = X_scaled_full[:, :ann_input_dim]

# =========================
# ANN prediction
# =========================
ann_pred = ann.predict(X_scaled_ann).flatten()[0]

# =========================
# OLS prediction
# NOTE: OLS model already includes constant
# =========================
ols_pred = ols.predict(X)[0]

# =========================
# Save output
# =========================
os.makedirs("/tmp/output", exist_ok=True)

output_df = pd.DataFrame({
    "ANN_Prediction": [ann_pred],
    "OLS_Prediction": [ols_pred]
})

output_df.to_csv(OUTPUT_PATH, index=False)

print("=== Event-Based Market Prediction ===")
print(output_df)

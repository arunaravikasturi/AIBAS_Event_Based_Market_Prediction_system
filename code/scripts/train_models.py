import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# Load Data
# =========================
train_df = pd.read_csv("/app/data/train/training_data.csv")
test_df  = pd.read_csv("/app/data/validation/test_data.csv")

# =========================
# Prepare Features
# =========================
TARGET = "Close_Price_Normalized"
DROP_COLS = ["Date", "Close_Price"]

X_train = train_df.drop(columns=DROP_COLS + [TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLS + [TARGET])
y_test = test_df[TARGET]

# =========================
# Feature Engineering
# =========================
for df_ in [X_train, X_test]:
    df_["Price_Range"] = df_["High_Price"] - df_["Low_Price"]
    df_["Return_Volatility"] = df_["Daily_Return_Pct"] * df_["Volatility_Range"]
    df_["Volume_Impact"] = df_["Volume"] * df_["Daily_Return_Pct"]

# =========================
# Scale for ANN
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =========================
# ---- Train ANN ----
# =========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(X_train_scaled, y_train,
          validation_data=(X_test_scaled, y_test),
          epochs=100,
          batch_size=32,
          callbacks=[early_stop],
          verbose=1)

model.save("/app/models/currentAiSolution.h5")
joblib.dump(scaler, "/app/models/scaler.pkl")

# =========================
# ---- Train OLS ----
# =========================
X_train_ols = sm.add_constant(X_train)
X_test_ols  = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train_ols).fit()
ols_model.save("/app/models/currentOlsSolution.pkl")

# =========================
# ---- Evaluation ----
# =========================
ann_pred = model.predict(X_test_scaled).flatten()
ols_pred = ols_model.predict(X_test_ols)

print("\nANN RMSE:", np.sqrt(mean_squared_error(y_test, ann_pred)))
print("ANN R2:", r2_score(y_test, ann_pred))

print("\nOLS RMSE:", np.sqrt(mean_squared_error(y_test, ols_pred)))
print("OLS R2:", r2_score(y_test, ols_pred))

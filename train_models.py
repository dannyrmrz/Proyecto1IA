# train_models.py

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

# =========================
# 1. CARGAR DATOS
# =========================

X = pd.read_csv("data/training_features.csv")
y_df = pd.read_csv("data/training_target.csv")

# Guardamos versión original para métricas interpretables
y_original = y_df.values.ravel()

# Transformación log1p para entrenamiento
y = np.log1p(y_original)

print("Dimensiones X:", X.shape)
print("Dimensiones y:", y.shape)

# =========================
# 2. IDENTIFICAR VARIABLES
# =========================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

print("Numéricas:", len(numeric_features))
print("Categóricas:", len(categorical_features))

# =========================
# 3. PREPROCESAMIENTO
# =========================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Importante: no reutilizar el mismo objeto `preprocessor` en 2 pipelines,
# porque se comparte estado al hacer `.fit()`.
preprocessor_rf = clone(preprocessor)
preprocessor_xgb = clone(preprocessor)

# =========================
# 4. SPLIT TRAIN / VALIDATION
# =========================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. RANDOM FOREST
# =========================

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_rf),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_preds_log = rf_pipeline.predict(X_val)
rf_preds = np.expm1(rf_preds_log)

rf_mse = mean_squared_error(np.expm1(y_val), rf_preds)
rf_rmse = np.sqrt(rf_mse)

print("Random Forest MSE:", rf_mse)
print("Random Forest RMSE:", rf_rmse)

# =========================
# 6. XGBOOST
# =========================

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_xgb),
    ("model", XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1,
        reg_lambda=1,
        tree_method="hist",
        random_state=42
    ))
])

xgb_pipeline.fit(X_train, y_train)

xgb_preds_log = xgb_pipeline.predict(X_val)
xgb_preds = np.expm1(xgb_preds_log)

xgb_mse = mean_squared_error(np.expm1(y_val), xgb_preds)
xgb_rmse = np.sqrt(xgb_mse)

print("XGBoost MSE:", xgb_mse)
print("XGBoost RMSE:", xgb_rmse)

# =========================
# 7. GUARDAR AMBOS MODELOS
# =========================

# Re-entrenar con todos los datos etiquetados disponibles (train + validation)
rf_pipeline.fit(X, y)
xgb_pipeline.fit(X, y)

with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_pipeline, f)

print("Random Forest guardado.")

with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_pipeline, f)

print("XGBoost guardado.")

if rf_mse < xgb_mse:
    print("Mejor modelo: Random Forest")
else:
    print("Mejor modelo: XGBoost")

# =========================
# 8. INFORMACIÓN DEL TARGET (ESCALA REAL)
# =========================

print("Media target (escala real):", np.mean(y_original))
print("Std target (escala real):", np.std(y_original))
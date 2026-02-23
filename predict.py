# predict.py

import pandas as pd
import pickle
import sys

# Uso:
# python predict.py models/xgboost_model.pkl testing_set.csv

model_path = sys.argv[1]
test_path = sys.argv[2]

# Cargar modelo
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Cargar datos nuevos
X_test = pd.read_csv(test_path)

# Predecir
predictions = model.predict(X_test)

# Guardar predicciones
pd.DataFrame(predictions, columns=["prediction"]).to_csv(
    "predictions.csv",
    index=False
)

print("Predicciones guardadas en predictions.csv")
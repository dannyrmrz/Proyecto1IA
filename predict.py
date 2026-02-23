# predict.py

import pandas as pd
import pickle
import sys
from pathlib import Path
import numpy as np

"""Run predictions and save them to a CSV.

Uso:
    python predict.py models/xgboost_model.pkl data/testing_features.csv
    python predict.py models/xgboost_model.pkl data/testing_features.csv output.csv
"""

if len(sys.argv) < 3:
        raise SystemExit(
                "Uso: python predict.py <ruta_modelo.pkl> <ruta_test.csv> [ruta_salida.csv]"
        )

model_path = Path(sys.argv[1])
test_path = Path(sys.argv[2])

script_dir = Path(__file__).resolve().parent
output_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else (script_dir / "predictions.csv")

# Cargar modelo
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Cargar datos nuevos
X_test = pd.read_csv(test_path)

# Predecir
predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log)

# Guardar predicciones
output_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(predictions, columns=["prediction"]).to_csv(output_path, index=False)

print(f"Predicciones guardadas en: {output_path.resolve()}")
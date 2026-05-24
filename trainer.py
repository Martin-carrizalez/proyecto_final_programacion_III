"""
trainer.py — Carga modelos ya entrenados desde el notebook de cardiopatía
"""
import os
import joblib
import numpy as np

MODEL_RED_PATH   = "modelo_red.pkl"
MODEL_SVM_PATH   = "modelo_svm.pkl"
MODEL_ARBOL_PATH = "modelo_arbol.pkl"
SCALER_PATH      = "estandarizador.pkl"
META_PATH        = "meta_cardio.npz"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]


def entrenar_y_guardar():
    """
    Verifica que los modelos del notebook ya existen.
    No necesita descargar ni entrenar nada.
    """
    archivos = [MODEL_RED_PATH, MODEL_SVM_PATH, MODEL_ARBOL_PATH, SCALER_PATH]
    faltantes = [a for a in archivos if not os.path.exists(a)]

    if faltantes:
        raise FileNotFoundError(
            f"Faltan estos archivos del notebook: {faltantes}\n"
            "Cópialos a la carpeta del proyecto."
        )

    # Guardar metadatos si no existen
    if not os.path.exists(META_PATH):
        scaler = joblib.load(SCALER_PATH)
        np.savez(
            META_PATH,
            mean=scaler.mean_.astype(np.float32),
            std=scaler.scale_.astype(np.float32),
            feature_names=FEATURE_NAMES
        )

    print("✅ Modelos verificados correctamente.")
    print(f"   {MODEL_RED_PATH}")
    print(f"   {MODEL_SVM_PATH}")
    print(f"   {MODEL_ARBOL_PATH}")
    print(f"   {SCALER_PATH}")


if __name__ == "__main__":
    entrenar_y_guardar()
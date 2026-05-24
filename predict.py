"""
predict.py — Predicción de cardiopatía con los 3 modelos entrenados
"""
import os
import numpy as np
import joblib

MODEL_RED_PATH   = "modelo_red.pkl"
MODEL_SVM_PATH   = "modelo_svm.pkl"
MODEL_ARBOL_PATH = "modelo_arbol.pkl"
SCALER_PATH      = "estandarizador.pkl"
META_PATH        = "meta_cardio.npz"

ETIQUETAS = {0: "Sano", 1: "Enfermo"}

# Índices de las 4 columnas que se estandarizaron en el notebook
# chol=4, trestbps=3, thalach=7, oldpeak=9
COLS_ESTANDAR = [3, 4, 7, 9]


def cargar_modelos():
    archivos = [MODEL_RED_PATH, MODEL_SVM_PATH, MODEL_ARBOL_PATH, SCALER_PATH]
    for archivo in archivos:
        if not os.path.exists(archivo):
            return None
    red    = joblib.load(MODEL_RED_PATH)
    svm    = joblib.load(MODEL_SVM_PATH)
    arbol  = joblib.load(MODEL_ARBOL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return red, svm, arbol, scaler


def _estandarizar(features, scaler):
    """Estandariza solo las 4 columnas que el scaler espera."""
    x = np.array(features, dtype=np.float64).reshape(1, -1)
    x_std = x.copy()
    x_std[0, COLS_ESTANDAR] = scaler.transform(x[:, COLS_ESTANDAR])[0]
    return x_std


def predecir_red(features, red, scaler):
    x_norm = _estandarizar(features, scaler)
    probs = red.predict_proba(x_norm)[0]
    idx = int(np.argmax(probs))
    return {"modelo": "Red Neuronal", "etiqueta": ETIQUETAS[idx], "confianza": float(probs[idx])}


def predecir_svm(features, svm, scaler):
    x_norm = _estandarizar(features, scaler)
    probs = svm.predict_proba(x_norm)[0]
    idx = int(np.argmax(probs))
    return {"modelo": "SVM", "etiqueta": ETIQUETAS[idx], "confianza": float(probs[idx])}


def predecir_arbol(features, arbol):
    x = np.array(features, dtype=np.float64).reshape(1, -1)
    probs = arbol.predict_proba(x)[0]
    idx = int(np.argmax(probs))
    return {"modelo": "Árbol de Decisión", "etiqueta": ETIQUETAS[idx], "confianza": float(probs[idx])}
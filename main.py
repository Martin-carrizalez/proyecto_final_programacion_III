"""
main.py — Sistema concurrente de predicción de cardiopatía
Arquitectura de hilos:
  Hilo 1 (sensor)     → simula lectura de datos del paciente
  Hilo 2 (prediccion) → corre los 3 modelos en paralelo
  Hilo 3 (imprimir)   → muestra resultados en pantalla + resumen final
"""
import random
import threading
import time
import queue

import predict
import trainer

# ─── Configuración ────────────────────────────────────────────────────────────
MAX_LECTURAS = 10

# ─── Colas de comunicación entre hilos ───────────────────────────────────────
cola_datos      = queue.Queue(maxsize=5)
cola_resultados = queue.Queue(maxsize=5)

# ─── Evento para detener todos los hilos ─────────────────────────────────────
detener = threading.Event()

# ─── Nombres de features ─────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Edad",
    "Sexo (1=M, 0=F)",
    "Tipo dolor pecho",
    "Presión arterial",
    "Colesterol",
    "Glucosa alta",
    "ECG reposo",
    "Frec. cardíaca máx",
    "Angina por ejercicio",
    "Depresión ST",
    "Pendiente ST",
    "Vasos principales",
    "Talasemia"
]

# ─── Rangos realistas del dataset UCI Heart Disease ───────────────────────────
RANGOS = {
    "age":      (29, 77),
    "sex":      (0, 1),
    "cp":       (1, 4),
    "trestbps": (94, 200),
    "chol":     (126, 564),
    "fbs":      (0, 1),
    "restecg":  (0, 2),
    "thalach":  (71, 202),
    "exang":    (0, 1),
    "oldpeak":  (0.0, 6.2),
    "slope":    (1, 3),
    "ca":       (0, 3),
    "thal":     (3, 7),
}


# ─── HILO 1: Sensor ──────────────────────────────────────────────────────────
def hilo_sensor():
    print("[Sensor] Hilo de sensores iniciado.")
    lecturas = 0
    while not detener.is_set() and lecturas < MAX_LECTURAS:
        datos = [
            random.randint(*RANGOS["age"]),
            random.randint(*RANGOS["sex"]),
            random.randint(*RANGOS["cp"]),
            random.randint(*RANGOS["trestbps"]),
            random.randint(*RANGOS["chol"]),
            random.randint(*RANGOS["fbs"]),
            random.randint(*RANGOS["restecg"]),
            random.randint(*RANGOS["thalach"]),
            random.randint(*RANGOS["exang"]),
            round(random.uniform(*RANGOS["oldpeak"]), 1),
            random.randint(*RANGOS["slope"]),
            random.randint(*RANGOS["ca"]),
            random.randint(*RANGOS["thal"]),
        ]
        try:
            cola_datos.put(datos, timeout=1)
        except queue.Full:
            pass
        lecturas += 1
        time.sleep(2)

    print(f"[Sensor] {MAX_LECTURAS} lecturas completadas. Deteniendo sistema...")
    detener.set()
    print("[Sensor] Hilo de sensores detenido.")


# ─── HILO 2: Predicción ──────────────────────────────────────────────────────
def hilo_prediccion(red, svm, arbol, scaler):
    print("[Predicción] Hilo de predicción iniciado.")
    while not detener.is_set() or not cola_datos.empty():
        try:
            datos = cola_datos.get(timeout=1)
        except queue.Empty:
            continue

        resultados = {}
        lock = threading.Lock()

        def predecir_con(nombre, fn):
            res = fn()
            with lock:
                resultados[nombre] = res

        t_red   = threading.Thread(target=predecir_con, args=("Red Neuronal",      lambda: predict.predecir_red(datos, red, scaler)))
        t_svm   = threading.Thread(target=predecir_con, args=("SVM",               lambda: predict.predecir_svm(datos, svm, scaler)))
        t_arbol = threading.Thread(target=predecir_con, args=("Árbol de Decisión", lambda: predict.predecir_arbol(datos, arbol)))

        t_red.start()
        t_svm.start()
        t_arbol.start()

        t_red.join()
        t_svm.join()
        t_arbol.join()

        try:
            cola_resultados.put((datos, resultados), timeout=1)
        except queue.Full:
            pass

    print("[Predicción] Hilo de predicción detenido.")


# ─── HILO 3: Impresión ───────────────────────────────────────────────────────
def hilo_impresion():
    print("[Impresión] Hilo de impresión iniciado.\n")

    # Acumuladores para el resumen final
    historial = []
    aciertos  = {"Red Neuronal": 0, "SVM": 0, "Árbol de Decisión": 0}
    total     = 0

    while not detener.is_set() or not cola_resultados.empty():
        try:
            datos, resultados = cola_resultados.get(timeout=1)
        except queue.Empty:
            continue

        total += 1

        print("=" * 55)
        print(f"  PACIENTE #{total}")
        print("=" * 55)
        for nombre, valor in zip(FEATURE_NAMES, datos):
            print(f"  {nombre:<12}: {valor}")

        print("\n  PREDICCIONES:")
        print("-" * 55)
        votos = {"Sano": 0, "Enfermo": 0}
        fila  = {"paciente": total}

        for nombre, res in resultados.items():
            etiqueta  = res["etiqueta"]
            confianza = res["confianza"]
            barra = "█" * int(confianza * 20)
            print(f"  {nombre:<20} → {etiqueta:<8} [{barra:<20}] {confianza*100:.1f}%")
            votos[etiqueta] += 1
            fila[nombre] = etiqueta

        ganador = max(votos, key=votos.get)
        fila["veredicto"] = ganador
        historial.append(fila)

        # Contar acuerdos con el veredicto por mayoría
        for nombre, res in resultados.items():
            if res["etiqueta"] == ganador:
                aciertos[nombre] += 1

        print(f"\n  VEREDICTO (mayoría): {ganador.upper()} ({votos[ganador]}/3 modelos)")
        print("=" * 55)
        print()

    # ─── RESUMEN FINAL ────────────────────────────────────────────────────────
    print("\n" + "█" * 55)
    print("  RESUMEN FINAL — 10 PACIENTES ANALIZADOS")
    print("█" * 55)

    sanos    = sum(1 for f in historial if f["veredicto"] == "Sano")
    enfermos = sum(1 for f in historial if f["veredicto"] == "Enfermo")

    print(f"\n  Total pacientes analizados : {total}")
    print(f"  Diagnosticados Sanos       : {sanos}")
    print(f"  Diagnosticados Enfermos    : {enfermos}")

    print("\n  CONCORDANCIA CON VEREDICTO POR MAYORÍA:")
    print("-" * 55)
    for modelo, n in aciertos.items():
        pct = (n / total) * 100 if total > 0 else 0
        barra = "█" * int(pct / 5)
        print(f"  {modelo:<20} : {n}/{total} [{barra:<20}] {pct:.1f}%")

    print("\n  DETALLE POR PACIENTE:")
    print("-" * 55)
    print(f"  {'#':<4} {'Red Neuronal':<14} {'SVM':<10} {'Árbol':<10} {'Veredicto'}")
    print("-" * 55)
    for f in historial:
        print(f"  {f['paciente']:<4} {f.get('Red Neuronal','?'):<14} {f.get('SVM','?'):<10} {f.get('Árbol de Decisión','?'):<10} {f['veredicto']}")

    print("█" * 55)
    print("[Impresión] Hilo de impresión detenido.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    cargado = predict.cargar_modelos()
    if cargado is None:
        print("No se encontraron modelos. Entrenando...")
        trainer.entrenar_y_guardar()
        cargado = predict.cargar_modelos()

    red, svm, arbol, scaler = cargado
    print("Modelos cargados correctamente.\n")

    t1 = threading.Thread(target=hilo_sensor,                                     name="Hilo-Sensor",     daemon=True)
    t2 = threading.Thread(target=hilo_prediccion, args=(red, svm, arbol, scaler), name="Hilo-Prediccion", daemon=True)
    t3 = threading.Thread(target=hilo_impresion,                                  name="Hilo-Impresion",  daemon=True)

    print(f"Iniciando sistema — {MAX_LECTURAS} lecturas y para automáticamente.\n")

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("Sistema finalizado correctamente.")


if __name__ == "__main__":
    main()
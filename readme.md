# Sistema Concurrente de Predicción de Cardiopatía

Diseñado e implementado con programación concurrente usando `threading` de Python y tres modelos de Machine Learning entrenados con scikit-learn.

## Requisitos previos

Los modelos fueron entrenados en el notebook `Cardiopatia.ipynb`. Antes de correr el sistema, copia estos archivos del notebook a esta carpeta:

- `modelo_red.pkl` — Red Neuronal entrenada
- `modelo_svm.pkl` — SVM entrenado
- `modelo_arbol.pkl` — Árbol de Decisión entrenado
- `estandarizador.pkl` — StandardScaler

## Instalación

1. Crea un entorno virtual:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```

3. Ejecuta el sistema:
   ```
   python main.py
   ```

El sistema analiza 10 pacientes y para automáticamente mostrando un resumen final.

## Arquitectura de hilos

El sistema utiliza 3 hilos principales que corren concurrentemente:

| Hilo | Nombre | Función |
|------|--------|---------|
| Hilo 1 | Hilo-Sensor | Simula la lectura de 13 variables clínicas del paciente cada 2 segundos |
| Hilo 2 | Hilo-Prediccion | Recibe los datos y corre los 3 modelos en sub-hilos paralelos |
| Hilo 3 | Hilo-Impresion | Muestra resultados, veredicto por mayoría y resumen final |

La comunicación entre hilos se realiza mediante **colas thread-safe** (`queue.Queue`):
- `cola_datos` → del Hilo-Sensor al Hilo-Prediccion
- `cola_resultados` → del Hilo-Prediccion al Hilo-Impresion

## Modelos

- **Red Neuronal** (MLPClassifier) — AUC 0.93, accuracy 85%
- **SVM** — accuracy 87%
- **Árbol de Decisión** — accuracy ~82%, interpretable, no requiere estandarización

El veredicto final se decide por **mayoría de votos** entre los 3 modelos.

## Dataset

UCI Heart Disease (Cleveland) — Janosi et al. (1989)
https://doi.org/10.24432/C52P4X

303 pacientes, 13 variables clínicas, target binario: Sano / Enfermo.
# Week15_15
 Week_15 - Data Journey, Monitoreo y Model Serving

## Integrantes:
DIEGO ALEJANDRO RUIZ ALFONSO - diegoaruiz@ucundinamarca.edu.co PEDRO PASCUAL MURCIA VARGAS - ppmurcia@ucundinamarca.edu.co JHON EDUARD TINJACA CRUZ - jetinjaca@ucundinamarca.edu.co JULIAN DAVID SILVA GUZMAN - jdsilva@ucundinamarca.edu.co

## Objetivo

Comprender y aplicar el ciclo de vida completo de un modelo de aprendizaje profundo: desde el origen y preparacion de los datos hasta el monitoreo del entrenamiento y la puesta en operacion (serving). La prioridad es demostrar trazabilidad del proceso completo, no maximizar el desempeno.

Este proyecto implementa una solución de Deep Learning para la detección de tumores mamarios utilizando el dataset Breast Cancer Wisconsin.

## Explicación del Código

### 1. Arquitectura del Modelo
Se implementó un Perceptrón Multicapa (MLP) utilizando únicamente **NumPy**. La estructura es:
- **Capa de Entrada**: 30 características (dimensiones del núcleo celular).
- **Capas Ocultas**: Dos capas con 32 y 16 neuronas respectivamente, utilizando la función de activación **ReLU** para introducir no linealidad.
- **Capa de Salida**: 1 neurona con función **Sigmoide**, que entrega una probabilidad entre 0 y 1 para la clasificación binaria.

### 2. Optimización y Pérdida
- **Función de Pérdida**: Binary Cross-Entropy (BCE), ideal para problemas de clasificación binaria.
- **Optimizador**: Adam (Adaptive Moment Estimation), que ajusta la tasa de aprendizaje para cada parámetro de forma individual basándose en momentos de primer y segundo orden.

### 3. Trazabilidad (Logging)
Se incluyó un sistema de `ExperimentLogger` que captura métricas por época (Loss, Acc, Gradientes) y las exporta a un archivo JSON, simulando el comportamiento de plataformas como Weights & Biases.

## Resultados Obtenidos

El modelo fue entrenado durante 300 épocas, logrando los siguientes resultados en el conjunto de prueba:

- **Exactitud (Accuracy)**: {acc(y_test, y_prob.reshape(-1,1))*100:.2f}%
- **ROC-AUC**: {auc:.4f}
- **Estado del Modelo**: Convergencia estable sin sobreajuste significativo.

## Requisitos
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn (solo para carga de datos y métricas)

## Archivos del Proyecto
- `dashboard_experimento.png`: Visualización del entrenamiento.
- `evaluacion_modelo.png`: Curvas ROC y Matrices de Confusión.
- `historial_experimento.json`: Datos crudos del experimento.

## Que se implementó

| Componente | Descripcion |
|------------|-------------|
| Data Journey | Diagrama del recorrido completo: imagen FNA -> caracteristicas -> CSV -> carga -> limpieza -> normalizacion -> entrenamiento -> evaluacion -> serving |
| Acceso y manipulacion | Carga del dataset Breast Cancer Wisconsin (scikit-learn), inspeccion de calidad, analisis exploratorio (histogramas, boxplots), division 80/20 estratificada y normalizacion con StandardScaler |
| Modelo | Red neuronal Entrada(30) -> Oculta1(32,ReLU) -> Oculta2(16,ReLU) -> Salida(1,Sigmoid) con Adam y Binary Cross-Entropy, implementada desde cero en NumPy |
| Monitoreo y logging | Sistema ExperimentLogger con registro de loss/accuracy/norma de gradiente por epoca, exportacion a JSON, dashboard de 6 graficas y tabla de hitos. Bloque W&B listo para activar incluido |
| Analisis de metricas | Reporte de clasificacion completo, curva ROC con AUC, curva Precision-Recall, matriz de confusion y analisis del gap train-validacion |
| Model Serving | Simulacion de endpoint POST /predict con preprocesamiento encapsulado, serializacion del artefacto (pesos + scaler) y esquema de codigo FastAPI para produccion |
| Conclusiones | Analisis tecnico de data journey, monitoreo, metricas, serving, dificultades y oportunidades de mejora |

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** — incluido en scikit-learn.

- 569 muestras (357 benignas, 212 malignas)
- 30 caracteristicas numericas (radio, textura, perimetro, area, etc.)
- Sin valores nulos
- Fuente original: UCI Machine Learning Repository

---

## Monitoreo

Se implementa un sistema de logging local equivalente a Weights and Biases que registra por epoca:

- Loss de entrenamiento y validacion
- Accuracy de entrenamiento y validacion
- Norma del gradiente (para detectar vanishing/exploding gradients)
- Timestamp y Run ID unico por experimento

El historial se exporta a `historial_experimento.json` para trazabilidad reproducible. Al final del notebook se incluye el bloque de integracion con W&B comentado y listo para activar.

---

## Resultados principales

La red neuronal implementada con Adam (LR=0.001, 300 epocas) logra convergencia estable sobre el dataset Breast Cancer Wisconsin. El ROC-AUC superior a 0.97 indica alta capacidad discriminativa entre tumores malignos y benignos. El monitoreo del gap train-validacion muestra que el modelo no presenta sobreajuste significativo en las 300 epocas entrenadas.

---

## Como ejecutar el notebook

### Google Colab (recomendado)

1. Descarga `notebook_data_journey.ipynb`.
2. Ve a [https://colab.research.google.com](https://colab.research.google.com).
3. Selecciona **Archivo -> Subir notebook** y carga el archivo.
4. Ejecuta todas las celdas con **Runtime -> Run all**.

No se requiere configuracion adicional. El dataset se descarga automaticamente desde scikit-learn.

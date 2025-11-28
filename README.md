# Wildlife Detection - Fauna Classification Pipeline

Este proyecto tiene como objetivo la detección y clasificación de fauna silvestre mediante un pipeline automatizado que procesa imágenes de cámaras trampa y aplica modelos de clasificación multiclase. El objetivo es predecir la presencia de diferentes especies en áreas específicas usando técnicas de procesamiento de imágenes y aprendizaje automático.

## 1. Descripción del Proyecto

El pipeline está compuesto por varias fases:

1. **Preprocesamiento de Imágenes**:
   - Limpieza de imágenes (eliminación de imágenes corruptas o demasiado pequeñas).
   - Recorte de las imágenes (eliminación del 10% inferior).
   - Redimensionamiento a una resolución estándar de 256x256 píxeles.
   - Conversión a escala de grises.
   - Mejora de contraste utilizando Histogram Equalization, CLAHE y Normalización.

2. **Extracción de Características**:
   - Características espaciales: intensidad media, contraste, entropía, densidad de bordes.
   - Características en el dominio de frecuencia utilizando FFT.
   - Características de textura utilizando Local Binary Patterns (LBP).

3. **Análisis y Clasificación**:
   - Estadísticas descriptivas de las características.
   - Análisis de correlación entre las características.
   - Reducción de dimensionalidad con PCA.
   - Modelos de clasificación multiclase utilizando Random Forest, SVM y Logistic Regression.
   - Evaluación de modelos con métricas como F1-score, precisión, recall y curvas ROC.

## 2. Requisitos

Este proyecto fue desarrollado en Python. Los requisitos para correr el pipeline son los siguientes:

- `pandas`
- `numpy`
- `opencv-python`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scikit-image`
- `scipy`
- `jupyter`

NOTA: solo se debe correr el main.py
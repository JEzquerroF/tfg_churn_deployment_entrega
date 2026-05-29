---
title: TFG Churn Prediction
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
app_file: app.py
pinned: false
license: mit
---

# Sistema de predicción de churn y segmentación de jugadores

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/JEzquerroF/tfg-churn-prediction)

Sistema productivo para predicción de churn (abandono) y segmentación
automática de jugadores en juegos free-to-play mobile. Desarrollado como
TFG con datos reales de una empresa partner.

## Características

- **Predicción de churn** en 3 horizontes temporales (7, 14 y 30 días)
  usando Random Forest con AUC de 0.93/0.91/0.85.
- **Segmentación en 6 arquetipos** de jugadores mediante KMeans clustering
  con caracterización por dominios de juego.
- **Detección de drift**: el sistema combina predicciones OOF (out-of-fold)
  del entrenamiento con predicciones en vivo, sirviendo la más adecuada
  según la magnitud del cambio en los datos del jugador.
- **Informes ejecutivos** descargables en PDF y Excel.
- **Diccionario de códigos** para integración programática del cliente
  con sus sistemas de live ops.

## Uso

1. Sube los CSVs exportados de la base de datos del juego (drag & drop)
2. Pulsa "Procesar datos"
3. Revisa los resultados en pantalla
4. Descarga los outputs (predicciones, PDF, Excel, diccionario)

**Límite del demo público**: 10.000 usuarios por ejecución. Para uso
operacional con datasets completos, despliega el sistema en tu propia
infraestructura (código disponible en el repositorio).

## CSVs requeridos

Obligatorios:
- `users.csv`
- `characters.csv`
- `devices.csv`
- `processed_consumables_iaps.csv`
- `processed_subscriptions_iaps.csv`
- `user_daily_rewards.csv`
- `user_items.csv`
- `user_items_collection.csv`
- `support_user_feedback_by_type.csv`

Recomendados (transaccionales, mejoran la segmentación):
- `currency_transactions.csv`
- `fights_log.csv` (sin columna `actions_log` para reducir tamaño)
- `arena_log.csv`

Período recomendado: últimos 30 días naturales.

## Stack técnico

- Python 3.11+
- Pipeline: pandas, scikit-learn, joblib
- Modelos: Random Forest (churn), KMeans + HDBSCAN (segmentación)
- UI: Streamlit
- Informes: weasyprint (PDF), openpyxl (Excel)
- Deploy: Hugging Face Spaces

## Modelo de churn (v2_rf_L22)

- **Algoritmo**: Random Forest Classifier
- **Features**: 62 (numéricas tras target encoding de 4 cat_cols)
- **Sample de entrenamiento**: L22 (33,598 usuarios filtrados)
- **Métricas test**: AUC 0.93/0.91/0.85 (7d/14d/30d)
- **Calibración**: naturalmente bien calibrado, sin isotonic

## Modelo de gustos (v1_kmeans_k6)

- **Algoritmo Nivel 1**: KMeans K=6
- **Algoritmo Nivel 2**: HDBSCAN
- **Features**: 78
- **Arquetipos detectados**: Recién Llegado, Establecido Activo, Hardcore
  End-Game, Veterano Especializado, Casual Dormido, Veterano Inversor

## Limitaciones documentadas

- HDBSCAN N2 actualmente devuelve -1 para todos los usuarios (modelo sin
  `prediction_data=True` en entrenamiento). Re-entrenamiento pendiente.
- Sub-arquetipos N2 sin narrativa semántica caracterizada.
- El threshold `p75 chars.level = 11` está persistido del training. Si el
  juego evoluciona y la distribución de niveles cambia mucho, conviene
  re-entrenar.

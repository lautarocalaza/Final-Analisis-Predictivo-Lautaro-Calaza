"""
predict_model.py
=================
Script para hacer predicciones con el modelo entrenado.

Uso:
    python predict_model.py

Input:
    - xgb_model_final.pkl (modelo entrenado)
    - scaler_final.pkl (escalador)
    - Movies Daily Update Dataset export 2025-08-18 22-06-40.csv (datos)

Output:
    - predictions_movies.csv (predicciones finales)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

print("\n" + "-" * 80)
print("PREDICCIÓN CON MODELO ENTRENADO")
print("-" * 80 + "\n")

#  PASO 1: CARGAR MODELO Y SCALER 

print("[1/8] Cargando modelo y scaler...")

modelo = joblib.load('xgb_model_final.pkl')
scaler = joblib.load('scaler_final.pkl')
metadata = joblib.load('model_metadata.pkl')

print(f"      Modelo cargado: xgb_model_final.pkl")
print(f"      Scaler cargado: scaler_final.pkl")
print(f"      Features esperados: {len(metadata['features'])}")
print(f"      Threshold: {metadata['threshold']}\n")

#  PASO 2: CARGAR DATOS 

print("[2/8] Cargando datos...")

ruta_archivo = "Movies Daily Update Dataset export 2025-08-18 22-06-40.csv"
df = pd.read_csv(ruta_archivo, encoding='latin-1')

df_valid = df[df['revenue'] > 0].copy()
print(f"      Películas con revenue > 0: {len(df_valid):,}\n")

#  PASO 3: FEATURE ENGINEERING 

print("[3/8] Aplicando feature engineering...")

df_model = df_valid.copy()

drop_columns = ['recommendations', 'tagline', 'backdrop_path', 'poster_path', 'id', 'title']
df_model = df_model.drop(columns=drop_columns, errors='ignore')

# Idiomas
df_model['is_english'] = (df_model['original_language'] == 'en').astype(int)
top_languages = ['en', 'es', 'fr', 'ja', 'ru', 'hi', 'zh', 'de', 'it', 'ko']
df_model['language_group'] = df_model['original_language'].apply(
    lambda x: x if x in top_languages else 'other'
)

# Fechas
df_model['release_date'] = pd.to_datetime(df_model['release_date'], errors='coerce')
df_model['year'] = df_model['release_date'].dt.year
df_model['month'] = df_model['release_date'].dt.month
df_model['quarter'] = df_model['release_date'].dt.quarter
df_model['day_of_week'] = df_model['release_date'].dt.dayofweek
df_model['is_recent'] = (df_model['year'] >= 2019).astype(int)

def get_season(month):
    if pd.isna(month):
        return 'unknown'
    if month in [1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

df_model['season'] = df_model['month'].apply(get_season)

# Géneros
def parse_genres_delimited(genre_str):
    if pd.isna(genre_str) or genre_str == '':
        return []
    genres = [g.strip().lower() for g in str(genre_str).split('-')]
    genres = [g for g in genres if g]
    return genres

df_model['genres_list'] = df_model['genres'].apply(parse_genres_delimited)

top_genres = ['drama', 'comedy', 'action', 'thriller', 'romance', 'adventure', 
              'crime', 'horror', 'family', 'fantasy', 'science fiction', 
              'mystery', 'animation', 'documentary', 'history']

for genre in top_genres:
    df_model[f'genre_{genre}'] = df_model['genres_list'].apply(
        lambda x: 1 if genre in x else 0
    )
df_model['num_genres'] = df_model['genres_list'].apply(len)

# Budget
df_model['has_budget'] = (df_model['budget'] > 0).astype(int)
df_model['budget_log'] = np.log1p(df_model['budget'])

def categorize_budget(budget):
    if budget == 0:
        return 'no_budget'
    elif budget < 1_000_000:
        return 'low'
    elif budget < 10_000_000:
        return 'medium'
    elif budget < 50_000_000:
        return 'high'
    else:
        return 'very_high'

df_model['budget_category'] = df_model['budget'].apply(categorize_budget)

# Status
df_model['is_released'] = (df_model['status'] == 'Released').astype(int)

# Outliers (Winsorizing P1-P99)
numeric_features = ['popularity', 'budget', 'runtime', 'vote_average', 'vote_count']
for feature in numeric_features:
    lower = df_model[feature].quantile(0.01)
    upper = df_model[feature].quantile(0.99)
    df_model[feature] = np.where(df_model[feature] < lower, lower, df_model[feature])
    df_model[feature] = np.where(df_model[feature] > upper, upper, df_model[feature])

# Nulos
df_model['runtime'].fillna(df_model['runtime'].median(), inplace=True)
df_model['overview'].fillna('', inplace=True)

print("      Feature engineering aplicado\n")

#  PASO 4: CODIFICACIÓN DE CATEGÓRICAS 

print("[4/8] Codificando variables categóricas...")

language_dummies = pd.get_dummies(df_model['language_group'], prefix='lang')
season_dummies = pd.get_dummies(df_model['season'], prefix='season')
budget_dummies = pd.get_dummies(df_model['budget_category'], prefix='budget_cat')

df_model = pd.concat([df_model, language_dummies, season_dummies, budget_dummies], axis=1)

print("      Categóricas codificadas\n")

#  PASO 5: PREPARAR MATRIZ X 

print("[5/8] Preparando matriz de features...")

X_features = metadata['features']
X = df_model[X_features].copy()
X.fillna(X.mean(numeric_only=True), inplace=True)

print(f"      Matriz X preparada: {X.shape}\n")

#  PASO 6: NORMALIZAR FEATURES 

print("[6/8] Normalizando features...")

X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X_features, index=X.index)

print("      Features normalizados\n")

#  PASO 7: HACER PREDICCIONES 

print("[7/8] Realizando predicciones...")

threshold = metadata['threshold']

y_pred_proba = modelo.predict_proba(X_scaled)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)

exitosas = (y_pred == 1).sum()
no_exitosas = (y_pred == 0).sum()

print(f"      Predicciones realizadas (threshold={threshold})")
print(f"      Películas predichas como exitosas: {exitosas} ({exitosas/len(y_pred)*100:.2f}%)")
print(f"      Películas predichas como no exitosas: {no_exitosas} ({no_exitosas/len(y_pred)*100:.2f}%)\n")

#  PASO 8: GUARDAR RESULTADOS 

print("[8/8] Guardando resultados...")

results = pd.DataFrame({
    'movie_id': df_valid['id'].values,
    'movie_title': df_valid['title'].values,
    'prediction': y_pred,
    'probability': np.round(y_pred_proba, 4),
    'budget': df_valid['budget'].values,
    'revenue': df_valid['revenue'].values
})

output_path = 'predictions_movies.csv'
results.to_csv(output_path, index=False)

print(f"      Archivo guardado: {output_path}")
print(f"      Total de registros: {len(results):,}\n")

#  RESUMEN FINAL 

print("-" * 80)
print("PREDICCIONES COMPLETADAS")
print("-" * 80)
print(f"\nPelículas predichas como exitosas: {exitosas:,} ({exitosas/len(results)*100:.2f}%)")
print(f"Películas predichas como no exitosas: {no_exitosas:,} ({no_exitosas/len(results)*100:.2f}%)")
print(f"Total de predicciones: {len(results):,}")
print(f"\nProbabilidad promedio: {y_pred_proba.mean():.4f}")
print(f"Rango de probabilidades: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
print(f"\nArchivo de salida: {output_path}")
print("\nPrimeras 10 predicciones:")
print("-" * 80)
print(results[['movie_title', 'prediction', 'probability']].head(10).to_string(index=False))
print("-" * 80 + "\n")
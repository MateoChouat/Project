from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import geopandas as gpd
import numpy as np





def add_missing_flags(gdf, cat_features, num_features):
    for col in cat_features:
        gdf[col + '_is_missing'] = ((gdf[col].isna()) | (gdf[col].astype(str).str.lower() == 'inconnu')).astype(int)
    for col in num_features:
        gdf[col + '_is_missing'] = gdf[col].isna().astype(int)
    return gdf

# Fonction générique pour imputer les données manquantes dans n'importe quelle colonne
def imput_missing_data(gdf, cat_features, num_features, target_column):
    # Ajouter les flags de valeurs manquantes
    gdf = add_missing_flags(gdf, cat_features, num_features)

    # Gérer les 'Inconnu' comme des NaN si c'est une colonne catégorielle
    if target_column in cat_features:
        gdf[target_column] = gdf[target_column].replace('Inconnu', np.nan)

    # Imputation des autres colonnes
    for col in cat_features:
        if col != target_column:
            gdf[col] = gdf[col].replace('Inconnu', np.nan).fillna("UNKNOWN").astype(str)
            
    for col in num_features:
        if col != target_column:
            gdf[col] = gdf[col].fillna(-999)

    # Préparation des features
    all_features = cat_features + num_features + [col + '_is_missing' for col in cat_features + num_features]
    
    # Extraction des données connues
    mask_known = ~gdf[target_column].isna()
    X = gdf.loc[mask_known, all_features]
    y = gdf.loc[mask_known, target_column]

    # Entraînement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(
        iterations=1000,
        depth=7,
        learning_rate=0.1,
        cat_features=[i for i, col in enumerate(all_features) if col in cat_features],
        verbose=200
    )
    model.fit(X_train, y_train)

    # Prédiction
    mask_missing = gdf[target_column].isna()
    gdf.loc[mask_missing, target_column] = model.predict(gdf.loc[mask_missing, all_features])

    return gdf


# Fonction pour imputer SHAPE_LEN en tant que somme des SHAPE_LEN pour un IDCANA donné
def imput_shape_len(gdf):
    gdf['SHAPE_LEN_TOT'] = gdf.groupby('IDCANA')['SHAPE_LEN'].transform('sum')
    return gdf
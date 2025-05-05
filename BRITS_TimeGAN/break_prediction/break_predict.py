# -*- coding: utf-8 -*-
"""
Version enrichie avec DBSCAN, Analyse de Survie par cluster, et poids_survie dynamique (fonction)
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import optuna
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# --------------------
# Fonctions auxiliaires
# --------------------

def find_best_dbscan(X_train_scaled):
    def objective(trial):
        eps = trial.suggest_float("eps", 0.1, 5.0)
        min_samples = trial.suggest_int("min_samples", 2, 20)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_train_scaled)
        if len(set(labels)) <= 1:
            return -1
        return silhouette_score(X_train_scaled, labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params

def bootstrap_survival(data, duration_col, event_col, n_bootstrap=300):
    kmf = KaplanMeierFitter()
    times = np.linspace(data[duration_col].min(), data[duration_col].max(), 100)
    surv_curves = []

    for _ in range(n_bootstrap):
        sample = data.sample(frac=1, replace=True)
        kmf.fit(sample[duration_col], event_observed=sample[event_col])
        surv_curves.append(kmf.survival_function_at_times(times).values)

    surv_curves = np.array(surv_curves)
    surv_mean = surv_curves.mean(axis=0)
    return times, surv_mean

# --------------------
# Programme principal
# --------------------

def prediction(GDF_CANA, GDF_DEFA):
    
    # 1. CHARGEMENT DES DONNÉES
    cana = GDF_CANA.copy()
    defa = GDF_DEFA.copy()

    cana['ID_ORIG'] = cana.index  # Sauvegarde index original

    if 'ANNEEPOSE' in cana.columns:
        cana['ANNEEPOSE'] = pd.to_datetime(cana['ANNEEPOSE'], errors='coerce')
    if 'DATEREPA' in defa.columns:
        defa['DATEREPA'] = pd.to_datetime(defa['DATEREPA'], errors='coerce')

    ANNEE_FIN_OBS = defa['DATEREPA'].dt.year.max()
    cana = cana.merge(defa, how='left', on='IDCANA')
    cana = cana.dropna(subset=['ANNEEPOSE'])

    cana['age'] = (cana['DATEREPA'] - cana['ANNEEPOSE']).dt.days / 365.25
    cana['age'] = cana['age'].fillna(ANNEE_FIN_OBS - cana['ANNEEPOSE'].dt.year)

    cana['start'] = cana['age']
    cana['stop'] = np.where(cana['DATEREPA'].notnull(), cana['age'], ANNEE_FIN_OBS - cana['ANNEEPOSE'].dt.year)
    cana['status'] = np.where(cana['DATEREPA'].notnull(), 1, 0)
    cana.loc[cana['start'] >= cana['stop'], 'stop'] += 1
    cana = cana[cana['stop'] >= cana['start']]

    cana['poids_longueur'] = cana['SHAPE_LEN'] / cana['SHAPE_LEN'].max()
    cana.loc[cana['MATERIAU'] == 'Fonte', 'MATERIAU'] = 'Fonte grise'

    ohe = OneHotEncoder(drop='first', sparse_output=False)
    materiaux_encoded = ohe.fit_transform(cana[['MATERIAU']])
    materiaux_cols = ohe.get_feature_names_out(['MATERIAU'])
    materiaux_df = pd.DataFrame(materiaux_encoded, columns=materiaux_cols, index=cana.index)

    cana = pd.concat([cana, materiaux_df], axis=1)

    variables = ['DIAMETRE', 'poids_longueur', 'age'] + list(materiaux_cols)
    data_survie = cana[['start', 'stop', 'status', 'ID_ORIG'] + variables].dropna()

    # 3. SPLIT
    X = data_survie[variables]
    y = data_survie[['start', 'stop', 'status', 'ID_ORIG']]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. DBSCAN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    best_params = find_best_dbscan(X_train_scaled)
    print(f"Best DBSCAN params: {best_params}")
    db = DBSCAN(**best_params)
    clusters_train = db.fit_predict(X_train_scaled)

    X_train = X_train.copy()
    y_train = y_train.copy()
    X_train['cluster'] = clusters_train
    y_train['cluster'] = clusters_train

    # 5. Fonctions de survie par cluster
    dict_survie = {}
    for cluster_id in sorted(np.unique(clusters_train)):
        cohort = y_train[y_train['cluster'] == cluster_id]
        times, surv_mean = bootstrap_survival(cohort, duration_col='stop', event_col='status')
        surv_func_interp = interp1d(times, surv_mean, bounds_error=False, fill_value=(1.0, surv_mean[-1]))
        dict_survie[cluster_id] = surv_func_interp

        # Optionnel : visualisation
        plt.figure(figsize=(8, 5))
        plt.plot(times, surv_mean, label=f"Cluster {cluster_id}", color='blue')
        plt.title(f"Fonction de survie - Cluster {cluster_id}")
        plt.xlabel("Age")
        plt.ylabel("P(survie)")
        plt.grid()
        plt.legend()
        plt.show()

    print("\n✅ Fonctions de survie par cluster générées avec succès.")

    # 6. Reconstruire le GeoDataFrame avec clusters
    df_clustered = pd.concat([X_train, y_train[['start', 'stop', 'status', 'ID_ORIG']]], axis=1)
    df_clustered['cluster'] = df_clustered['cluster'].astype(int)

    gdf_clustered = GDF_CANA.loc[df_clustered['ID_ORIG']].copy()
    gdf_clustered = gdf_clustered.reset_index(drop=True)
    df_clustered = df_clustered.reset_index(drop=True)

    gdf_clustered['cluster'] = df_clustered['cluster']
    gdf_clustered['start'] = df_clustered['start']
    gdf_clustered['stop'] = df_clustered['stop']
    gdf_clustered['status'] = df_clustered['status']

    return dict_survie, gdf_clustered


# Exemple d'appel :
dict_survie, gdf_clustered = prediction(
    gpd.read_file("C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/cana.shp"),
    gpd.read_file("C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/def.shp")
)
print(dict_survie)
print(gdf_clustered.head())


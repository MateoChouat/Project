# -*- coding: utf-8 -*-
"""
Version enrichie avec DBSCAN + Optuna + Bootstrap
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import WeibullAFTFitter, KaplanMeierFitter
from lifelines.utils import survival_table_from_events
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

import joblib
import optuna
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
            return -1  # Mauvais clustering
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
    surv_std = surv_curves.std(axis=0)
    return times, surv_mean, surv_std

# --------------------
# Programme principal
# --------------------
def prediction(CANA_PATH, DEFA_PATH):
    # 1. CHARGEMENT DES DONNÉES
    cana = gpd.read_file(CANA_PATH)
    defa = gpd.read_file(DEFA_PATH)

    # 2. PRÉPARATION DES DONNÉES
    if 'ANNEEPOSE' in cana.columns:
        cana['ANNEEPOSE'] = pd.to_datetime(cana['ANNEEPOSE'], errors='coerce')
    if 'DATEREPA' in defa.columns:
        defa['DATEREPA'] = pd.to_datetime(defa['DATEREPA'], errors='coerce')

    ANNEE_DEBUT_OBS = defa['DATEREPA'].dt.year.min()
    ANNEE_FIN_OBS = defa['DATEREPA'].dt.year.max()
    
    

    cana = cana.merge(defa, how='left', on='IDCANA')

    # Calcul de l'âge de la canalisation en années
    cana = cana.dropna(subset=['ANNEEPOSE'])

    cana['age'] = (cana['DATEREPA'] - cana['ANNEEPOSE']).dt.days/365.25
    cana['age'] = cana['age'].fillna(ANNEE_FIN_OBS - cana['ANNEEPOSE'].dt.year)

    cana['start'] = cana['age']  # Le début de l'âge est l'âge de la canalisation au moment de l'observation
    cana['stop'] = np.where(cana['DATEREPA'].notnull(), cana['age'], ANNEE_FIN_OBS - cana['ANNEEPOSE'].dt.year)
    cana['status'] = np.where(cana['DATEREPA'].notnull(), 1, 0)

    cana.loc[cana['start'] >= cana['stop'], 'stop'] += 1
    cana = cana[cana['stop'] >= cana['start']]

    # Poids longueur normalisée
    cana['poids_longueur'] = cana['SHAPE_LEN'] / cana['SHAPE_LEN'].max()
    
    # Modifier directement les valeurs du DataFrame
    cana.loc[cana['MATERIAU'] == 'Fonte', 'MATERIAU'] = 'Fonte grise'


    # Encodage
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    materiaux_encoded = ohe.fit_transform(cana[['MATERIAU']])
    materiaux_cols = ohe.get_feature_names_out(['MATERIAU'])
    materiaux_df = pd.DataFrame(materiaux_encoded, columns=materiaux_cols, index=cana.index)

    cana = pd.concat([cana, materiaux_df], axis=1)

    variables = ['DIAMETRE', 'poids_longueur', 'age'] + list(materiaux_cols)
    data_survie = cana[['start', 'stop', 'status'] + variables].dropna()

    # 3. SPLIT TRAIN / TEST
    X = data_survie[variables]
    y = data_survie[['start', 'stop', 'status']]
    y['age'] = cana['age']
    y['DIAMETRE'] = cana['DIAMETRE']
    y['SHAPE_LEN'] = cana['SHAPE_LEN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. CLUSTERING DBSCAN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    best_params = find_best_dbscan(X_train_scaled)
    print(f"Best DBSCAN params: {best_params}")

    db = DBSCAN(**best_params)
    clusters_train = db.fit_predict(X_train_scaled)
    X_train['cluster'] = clusters_train
    y_train['cluster'] = clusters_train

    # Enlever bruit (-1)
    X_train = X_train[X_train['cluster'] != -1]
    y_train = y_train[y_train['cluster'] != -1]

    # 5. ANALYSE PAR COHORTES
    cluster_summary = []  # Pour stocker les informations sur les clusters
    for cluster_id in sorted(X_train['cluster'].unique()):
        print(f"\n\n=== Cluster {cluster_id} ===")
        cohort = y_train[y_train['cluster'] == cluster_id]
        

        times, surv_mean, surv_std = bootstrap_survival(cohort, duration_col='stop', event_col='status')

        plt.figure(figsize=(10,6))
        plt.plot(times, surv_mean, label=f"Cluster {cluster_id}", color='blue')
        plt.fill_between(times, surv_mean-surv_std, surv_mean+surv_std, alpha=0.3, color='blue')
        plt.title(f"Fonction de survie - Cluster {cluster_id}")
        plt.xlabel("Age de la canalisation (années)")
        plt.ylabel("Probabilité de survie")
        plt.grid()
        plt.legend()
        plt.show()
        

        # Collecter les statistiques du cluster
        cluster_stats = {
            'Cluster': cluster_id,
            'Taille': len(cohort),
            'Age moyen': cohort['age'].mean(),
            'Diamètre moyen': cohort['DIAMETRE'].mean(),
            'Longueur moyenne': cohort['SHAPE_LEN'].mean()
        }
        cluster_summary.append(cluster_stats)

    cluster_summary_df = pd.DataFrame(cluster_summary)
    print("\nRésumé des clusters :")
    print(cluster_summary_df)

    # 6. Modèle paramétrique global
    X_train_full = pd.concat([X_train.drop('cluster', axis=1), y_train[['start', 'stop', 'status']]], axis=1)
    #X_train_full = X_train_full.drop(columns=['poids_longueur'])
    aft = WeibullAFTFitter()
    aft.fit(X_train_full, duration_col='stop', event_col='status', entry_col='start', weights_col="poids_longueur")
    aft.print_summary()

    joblib.dump(aft, 'modele_survie_weibull_dbscan.pkl')

    print("\n✅ Script terminé avec survie par cohortes et modèle paramétrique global.")


# ---------------------
# Execution
# ---------------------
prediction("C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/cana.shp", "C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/def.shp")

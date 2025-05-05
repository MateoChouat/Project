# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:02:07 2025

@author: mc
"""

# =========================================
# MODULES À INSTALLER AVANT
# pip install pandas geopandas lifelines scikit-learn matplotlib seaborn joblib
# =========================================

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import joblib

def find_best_k(inertia, seuil_baisse=0.1):
    """
    Trouve automatiquement le meilleur nombre de clusters avec plus de sélectivité.
    
    seuil_baisse : float
        Le seuil minimal de baisse relative pour accepter un coude.
        Plus ce chiffre est bas, plus on attend une vraie stabilisation.
    """
    variations = -np.diff(inertia) / inertia[:-1]  # Signe positif pour les baisses

    # Chercher où la variation relative est importante puis se stabilise
    for i in range(1, len(variations)):
        # Si la baisse devient inférieure au seuil après une forte baisse, on considère que le coude est avant
        if variations[i] < seuil_baisse and variations[i-1] >= seuil_baisse:
            return i + 1  # +1 car np.diff réduit de 1 la taille

    # Si pas trouvé, prendre le coude au plus fort décrochement
    best_k = np.argmax(variations) + 1
    return best_k if best_k > 1 else 2


def prediction(CANA_PATH, DEFA_PATH):
    # --- 1. CHARGEMENT DES DONNÉES ---
    cana = gpd.read_file(CANA_PATH)
    defa = gpd.read_file(DEFA_PATH)

    # --- 2. PRÉPARATION DES DONNÉES ---
    if 'ANNEEPOSE' in cana.columns:
        cana['ANNEEPOSE'] = pd.to_datetime(cana['ANNEEPOSE'], errors='coerce')
    if 'DATEREPA' in defa.columns:
        defa['DATEREPA'] = pd.to_datetime(defa['DATEREPA'], errors='coerce')

    ANNEE_DEBUT_OBS = defa['DATEREPA'].dt.year.min()
    ANNEE_FIN_OBS = defa['DATEREPA'].dt.year.max()

    cana = cana.merge(defa, how='left', on='IDCANA')

    cana['ANNEEPOSE_YEAR'] = cana['ANNEEPOSE'].dt.year
    cana['DATEREPA_YEAR'] = cana['DATEREPA'].dt.year

    cana['ANNEEPOSE_YEAR'] = cana['ANNEEPOSE_YEAR'].fillna(ANNEE_DEBUT_OBS)

    cana['start'] = np.where(cana['ANNEEPOSE_YEAR'] > ANNEE_DEBUT_OBS, cana['ANNEEPOSE_YEAR'], ANNEE_DEBUT_OBS)
    cana['stop'] = np.where(cana['DATEREPA_YEAR'].notnull(), cana['DATEREPA_YEAR'], ANNEE_FIN_OBS)
    cana['status'] = np.where(cana['DATEREPA_YEAR'].notnull(), 1, 0)

    cana.loc[cana['start'] >= cana['stop'], 'stop'] += 1
    cana = cana[cana['stop'] >= cana['start']]

    # --- 3. ENCODAGE DES VARIABLES ---
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    materiaux_encoded = ohe.fit_transform(cana[['MATERIAU']])
    materiaux_cols = ohe.get_feature_names_out(['MATERIAU'])
    materiaux_df = pd.DataFrame(materiaux_encoded, columns=materiaux_cols, index=cana.index)
    cana = pd.concat([cana, materiaux_df], axis=1)

    variables = ['DIAMETRE', 'SHAPE_LEN', 'ANNEEPOSE_YEAR'] + list(materiaux_cols)
    data_survie = cana[['start', 'stop', 'status'] + variables].dropna()

    print("✅ Colonnes présentes dans data_survie après l'encodage one-hot :")
    print(data_survie.columns)

    # --- 4. CLUSTERING AUTOMATIQUE ---

    features = data_survie[['DIAMETRE', 'SHAPE_LEN', 'ANNEEPOSE_YEAR'] + list(materiaux_cols)]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Trouver automatiquement le meilleur nombre de clusters
    inertia = []
    K = range(1, 51)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)

    # Afficher la courbe du coude
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude pour choisir k')
    plt.grid()
    plt.show()

    n_clusters = find_best_k(inertia)
    print(f"✅ Nombre de clusters automatiquement choisi : {n_clusters}")

    # Appliquer KMeans avec n_clusters optimal
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data_survie['cluster'] = kmeans.fit_predict(features_scaled)

    # --- 5. CALCUL DE L'ÂGE DE LA CANALISATION ---
    data_survie['age'] = ANNEE_FIN_OBS - data_survie['ANNEEPOSE_YEAR']

    # --- 6. ANALYSE DES CLUSTERS ---
    variables_numeriques = ['DIAMETRE', 'SHAPE_LEN', 'age']  # Utilisation de l'âge ici

    for var in variables_numeriques:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data_survie, x='cluster', y=var)
        plt.title(f'Distribution de {var} par cluster')
        plt.grid()
        plt.show()

    for cluster_id in sorted(data_survie['cluster'].unique()):
        print(f"\n🔵 Matériaux pour le cluster {cluster_id} :")
        materiaux_in_cluster = cana.loc[data_survie.index[data_survie['cluster'] == cluster_id], 'MATERIAU']
        print(materiaux_in_cluster.value_counts())

    # --- 7. ENTRAINEMENT DU MODELE ---
    aft = WeibullAFTFitter()
    aft.fit(data_survie, duration_col='stop', event_col='status', entry_col='start')

    aft.print_summary()

    # --- 8. PRÉDICTION FLEXIBLE DES HORIZONS ---
    HORIZONS = [5, 10, 15]
    horizons_years = [ANNEE_FIN_OBS + h for h in HORIZONS]

    surv_funcs = aft.predict_survival_function(data_survie, times=horizons_years)

    for idx, h in enumerate(HORIZONS):
        data_survie[f'proba_survie_{h}ans'] = surv_funcs.T.iloc[:, idx]
        data_survie[f'proba_casse_{h}ans'] = 1 - data_survie[f'proba_survie_{h}ans']

    # --- 9. VISUALISATION GLOBALE ---
    plt.figure(figsize=(12, 6))
    times = np.linspace(ANNEE_DEBUT_OBS, ANNEE_FIN_OBS + 30, 300)
    mean_survival = aft.predict_survival_function(data_survie, times=times).mean(axis=1)

    plt.plot(times, mean_survival, label="Survie moyenne réseau", color='blue')
    plt.fill_between(times,
                     mean_survival * 0.95,
                     mean_survival * 1.05,
                     color='blue',
                     alpha=0.2,
                     label="±5% (approximatif)")
    plt.title("Courbe de Survie Moyenne du Réseau")
    plt.xlabel("Année")
    plt.ylabel("Probabilité de survie")
    plt.axvline(ANNEE_FIN_OBS, color='red', linestyle='--', label="Fin observation")
    plt.legend()
    plt.grid()
    plt.show()

    # --- 10. VISUALISATION DE QUELQUES CANALISATIONS INDIVIDUELLES ---
    ids_exemples = np.random.choice(data_survie.index, size=5, replace=False)

    plt.figure(figsize=(14, 8))
    for idx in ids_exemples:
        surv = aft.predict_survival_function(data_survie.loc[[idx]], times=times)
        plt.plot(times, surv.squeeze(), label=f"Cana {idx}")

    plt.title("Fonctions de survie pour quelques canalisations")
    plt.xlabel("Année")
    plt.ylabel("Probabilité de survie")
    plt.axvline(ANNEE_FIN_OBS, color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()

    # --- 11. VISUALISATION PAR CLUSTER ---
    plt.figure(figsize=(14, 10))
    for cluster_id in sorted(data_survie['cluster'].unique()):
        cluster_data = data_survie[data_survie['cluster'] == cluster_id]
        surv = aft.predict_survival_function(cluster_data, times=times)
        plt.plot(times, surv.mean(axis=1), label=f"Cluster {cluster_id}")

    plt.title("Courbes de survie par cluster")
    plt.xlabel("Année")
    plt.ylabel("Probabilité de survie")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.show()

    # --- 12. SAUVEGARDE DU MODELE ---
    joblib.dump(aft, 'modele_survie_weibull_avance.pkl')

    print("✅ Script avancé terminé avec succès.")

# Exemple d'exécution :
prediction("C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/cana.shp", "C:/Users/mc/Desktop/BRITS_TimeGAN/data/raw/ANTIBES/def.shp")

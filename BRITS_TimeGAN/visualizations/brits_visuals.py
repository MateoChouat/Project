# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:52:57 2025

@author: mc
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import random

# === Chargement des données ===
GDF_PATH = "data/your_clustered_canalisations.shp"
IMPUTED_PATH = "brits_imputed_dict.pkl"
ORIGINAL_PATH = "brits_original_dict.pkl"

import pickle
with open(IMPUTED_PATH, "rb") as f:
    imputed_dict = pickle.load(f)
with open(ORIGINAL_PATH, "rb") as f:
    original_dict = pickle.load(f)

gdf = gpd.read_file(GDF_PATH)

# === Fonction d'affichage par canalisation ===
def plot_cana_series(idcana, imputed_series, original_series, years):
    plt.figure(figsize=(10, 3))
    plt.plot(years, original_series, label="Original (avec NaNs)", linestyle='--', color='gray')
    plt.plot(years, imputed_series, label="Imputé par BRITS", color='blue')
    plt.title(f"IDCANA: {idcana}")
    plt.xlabel("Année")
    plt.ylabel("Taux de casse ou mesure")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# === 1. Exemple sur 3 canalisations ===
sample_ids = random.sample(list(imputed_dict.keys()), 3)
for idcana in sample_ids:
    imp = imputed_dict[idcana]
    orig = original_dict[idcana]
    row = gdf[gdf['IDCANA'] == idcana]
    start_year = int(row['Year_MDWindow_Start'].values[0])
    end_year = int(row['Year_MDWindow_End'].values[0])
    years = np.arange(start_year, end_year + 1)
    
    plot_cana_series(idcana, imp.flatten(), orig.flatten(), years)

# === 2. Moyennes par cluster ===
clusters = gdf['cluster'].unique()
for cluster in sorted(clusters):
    cluster_ids = gdf[gdf['cluster'] == cluster]['IDCANA'].values
    aligned_orig, aligned_imp = [], []

    for idcana in cluster_ids:
        if idcana not in imputed_dict or idcana not in original_dict:
            continue
        orig, imp = original_dict[idcana], imputed_dict[idcana]
        if orig.shape != imp.shape:
            continue
        aligned_orig.append(orig)
        aligned_imp.append(imp)

    if not aligned_orig:
        continue

    aligned_orig = np.stack(aligned_orig, axis=0)
    aligned_imp = np.stack(aligned_imp, axis=0)
    mean_orig = np.nanmean(aligned_orig, axis=0)
    mean_imp = np.nanmean(aligned_imp, axis=0)

    plt.figure(figsize=(10, 3))
    plt.plot(mean_orig, label="Original (NaNs)", linestyle='--', color='gray')
    plt.plot(mean_imp, label="Imputé", color='green')
    plt.title(f"Moyenne du cluster {cluster}")
    plt.xlabel("Temps relatif")
    plt.ylabel("Valeur moyenne")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# === 3. Heatmap du taux de valeurs imputées par canalisation ===
impute_ratios = {}
for idcana in original_dict:
    orig = original_dict[idcana]
    if isinstance(orig, np.ndarray):
        n_total = orig.size
        n_nan = np.isnan(orig).sum()
        impute_ratios[idcana] = n_nan / n_total

gdf['impute_ratio'] = gdf['IDCANA'].map(impute_ratios)

plt.figure(figsize=(10, 4))
sns.histplot(gdf['impute_ratio'].dropna(), bins=30, kde=True, color='orange')
plt.title("Distribution du taux de valeurs imputées (NaNs)")
plt.xlabel("Ratio de valeurs manquantes")
plt.tight_layout()
plt.show()

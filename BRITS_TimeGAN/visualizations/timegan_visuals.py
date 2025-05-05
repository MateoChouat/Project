# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:48:56 2025

@author: mc
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import geopandas as gpd

# === Paramètres ===
PICKLE_PATH = "synthetic_time_series_per_cana.pkl"
GDF_PATH = "data/your_cana_data_imputed.shp"  # si carte
DISPLAY_IDS = 5  # nombre de courbes à visualiser
FEATURE_INDEX = 0  # feature à tracer

# === Chargement ===
with open(PICKLE_PATH, "rb") as f:
    synthetic_data = pickle.load(f)

gdf = gpd.read_file(GDF_PATH)

# === 1. Distribution des longueurs ===
lengths = [seq.shape[0] for seq in synthetic_data.values()]
plt.figure(figsize=(10, 4))
sns.histplot(lengths, bins=30, kde=True)
plt.title("Distribution des longueurs de séries synthétiques")
plt.xlabel("Durée (années)")
plt.ylabel("Nombre de canalisations")
plt.tight_layout()
plt.show()

# === 2. Affichage de séries individuelles ===
print(f"Affichage de {DISPLAY_IDS} séries synthétiques :")
for i, (idcana, seq) in enumerate(synthetic_data.items()):
    if i >= DISPLAY_IDS:
        break
    plt.figure(figsize=(8, 3))
    plt.plot(seq[:, FEATURE_INDEX])
    plt.title(f"IDCANA: {idcana} | Feature {FEATURE_INDEX}")
    plt.xlabel("Années")
    plt.ylabel("Valeur")
    plt.grid()
    plt.tight_layout()
    plt.show()

# === 3. Moyenne / std par temps ===
max_len = max(lengths)
feature_dim = list(synthetic_data.values())[0].shape[1]
tensor = np.zeros((len(synthetic_data), max_len, feature_dim))

for i, seq in enumerate(synthetic_data.values()):
    tensor[i, :seq.shape[0], :] = seq

means = np.mean(tensor, axis=0)
stds = np.std(tensor, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(means[:, FEATURE_INDEX], label='Moyenne')
plt.fill_between(range(max_len), 
                 means[:, FEATURE_INDEX] - stds[:, FEATURE_INDEX],
                 means[:, FEATURE_INDEX] + stds[:, FEATURE_INDEX],
                 alpha=0.3, label='± Écart type')
plt.title(f"Moyenne et écart type - Feature {FEATURE_INDEX}")
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.legend()
plt.tight_layout()
plt.show()

# === 4. t-SNE des séquences moyennes ===
flat_seq = [np.mean(seq, axis=0) for seq in synthetic_data.values()]
flat_seq = np.array(flat_seq)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeds = tsne.fit_transform(flat_seq)

plt.figure(figsize=(8, 6))
plt.scatter(embeds[:, 0], embeds[:, 1], alpha=0.6)
plt.title("Projection t-SNE des séquences synthétiques")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()
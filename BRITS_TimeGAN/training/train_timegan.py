# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:30:33 2025

@author: mc
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from timegan_model import TimeGAN  # Le modèle que tu as déjà défini
import numpy as np
import pickle
import os

def prepare_dataloader(time_series_dict, batch_size=32):
    all_seqs = []

    for series in time_series_dict.values():
        all_seqs.append(torch.tensor(series, dtype=torch.float32))

    data_tensor = torch.stack(all_seqs)  # shape: [num_samples, seq_len, feature_dim]
    dataset = TensorDataset(data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_timegan(model, dataloader, device, epochs=100, lr=1e-3):
    # Séparation Embedder/Recovery (phase 1)
    optimizerE = torch.optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0].to(device)
            h = model.embedder(x)
            x_tilde = model.recovery(h)
            loss = mse_loss(x_tilde, x)

            optimizerE.zero_grad()
            loss.backward()
            optimizerE.step()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Embedder Loss: {loss.item():.4f}")

    # Tu peux ajouter ici les phases supplémentaires : superviseur, adversarial, etc.

    return model


def save_model(model, path="timegan_model.pt"):
    torch.save(model.state_dict(), path)
    print("✅ Modèle sauvegardé.")


def load_time_series_by_cana(gdf, features, max_len=20):
    """
    Transforme le GeoDataFrame en dictionnaire : {IDCANA: time_series_array}
    Chaque série a shape (max_len, num_features)
    """
    cana_dict = {}

    for idcana, group in gdf.groupby("IDCANA"):
        group = group.sort_values("DATE", ascending=True)
        values = group[features].values[-max_len:]
        padded = np.zeros((max_len, len(features)))
        padded[-len(values):] = values
        cana_dict[idcana] = padded

    return cana_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔽 Charge les données prétraitées (gdf + features temporelles)
    with open("gdf_imputed.pkl", "rb") as f:
        gdf = pickle.load(f)

    features = ['NB_CASSES', 'DIAMETRE', 'MATERIAU_CODE', 'AGE']  # exemple

    time_series_dict = load_time_series_by_cana(gdf, features)
    dataloader = prepare_dataloader(time_series_dict)

    model = TimeGAN(feature_dim=len(features)).to(device)
    model = train_timegan(model, dataloader, device)

    save_model(model)

    with open("cana_ids.pkl", "wb") as f:
        pickle.dump(list(time_series_dict.keys()), f)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
BRITS - Utilisation du modèle entraîné pour imputation (avec scaler déjà fourni)
"""
import torch
import pandas as pd
import numpy as np
from models.brits import BRITS
from generate_mask import generate_masks_deltas


def load_model(model_path, input_dim, hidden_dim=64):
    model = BRITS(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def impute(model, data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor):
    with torch.no_grad():
        imputations, _ = model(data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor)
    return imputations


def run_brits_inference(df_input, model_path, fitted_scaler):
    """
    Applique BRITS à des données standardisées pour imputation.

    Args:
        df_input (pd.DataFrame): Données normalisées à imputer (NaNs autorisés)
        model_path (str): Chemin vers le modèle BRITS entraîné
        fitted_scaler (sklearn.preprocessing.StandardScaler): Scaler déjà entraîné

    Returns:
        pd.DataFrame: Données imputées et remises à l’échelle d’origine
    """
    df = df_input.copy()

    # Générer masques et deltas
    masks, deltas, deltas_rev = generate_masks_deltas(df)

    # Remplissage temporaire des NaNs pour éviter crash dans torch (sera remplacé)
    df_temp = df.fillna(df.mean())
    df_scaled = fitted_scaler.transform(df_temp)

    # Conversion en tenseurs
    data_tensor = torch.tensor(df_scaled, dtype=torch.float32).unsqueeze(0)
    masks_tensor = torch.tensor(masks, dtype=torch.float32).unsqueeze(0)
    deltas_tensor = torch.tensor(deltas, dtype=torch.float32).unsqueeze(0)
    deltas_rev_tensor = torch.tensor(deltas_rev, dtype=torch.float32).unsqueeze(0)

    # Chargement du modèle
    input_dim = data_tensor.shape[2]
    model = load_model(model_path, input_dim=input_dim)

    # Exécution de l'imputation
    imputations_tensor = impute(model, data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor)
    imputations_array = imputations_tensor.squeeze(0).numpy()

    # Remise à l’échelle originale
    imputations_array_rescaled = fitted_scaler.inverse_transform(imputations_array)
    df_imputed = pd.DataFrame(imputations_array_rescaled, columns=df.columns)

    return df_imputed

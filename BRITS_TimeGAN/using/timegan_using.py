import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
from datetime import datetime
from timegan_model import TimeGAN


def load_model(model_path, feature_dim, device):
    model = TimeGAN(feature_dim=feature_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_sequence(model, seq_len, feature_dim, device):
    with torch.no_grad():
        z = torch.randn((1, seq_len, feature_dim)).to(device)
        h = model.generator(z)
        h_hat = model.supervisor(h)
        x_hat = model.recovery(h_hat)
        return x_hat.squeeze(0).cpu().numpy()


def get_operation_years(row, current_year):
    if pd.isnull(row['ANNEEPOSE']):
        return 0  # skip if no install date
    start = pd.to_datetime(row['ANNEEPOSE']).year
    end = (
        int(row['DATEABANDON']) if isinstance(row['DATEABANDON'], (int, float)) and row['DATEABANDON'] > 0
        else current_year
    )
    return max(end - start, 1)


def use_timegan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🗂️ Charger les données
    gdf = gpd.read_file("data/your_cana_data_imputed.shp")  # adapt path
    feature_dim = 4  # nombre de features du TimeGAN

    model = load_model("timegan_model.pt", feature_dim, device)

    current_year = datetime.now().year
    synthetic_data = {}

    for _, row in gdf.iterrows():
        idcana = row['IDCANA']
        years = get_operation_years(row, current_year)
        if years <= 0:
            continue

        seq = generate_sequence(model, seq_len=years, feature_dim=feature_dim, device=device)
        synthetic_data[idcana] = seq

    # 💾 Sauvegarde
    with open("synthetic_time_series_per_cana.pkl", "wb") as f:
        pickle.dump(synthetic_data, f)

    print("✅ Séries synthétiques générées pour chaque canalisation.")




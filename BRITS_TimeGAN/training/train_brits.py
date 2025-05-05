import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
import numpy as np
import optuna
import joblib

from models.brits_model import BRITS
from generate_mask import generate_masks_deltas


def preprocess(df, scaler_path=None, fit=True):
    df = df.copy()
    # Imputation simple pour permettre un passage dans BRITS
    df_interp = df.fillna(df.mean())
    return df_interp.values


def prepare_tensors(df):
    """
    Prépare les tenseurs pour BRITS sans appliquer de scaling.
    """
    data_np = preprocess(df)

    masks, deltas, deltas_rev = generate_masks_deltas(df)

    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    masks_tensor = torch.tensor(masks, dtype=torch.float32)
    deltas_tensor = torch.tensor(deltas, dtype=torch.float32)
    deltas_rev_tensor = torch.tensor(deltas_rev, dtype=torch.float32)

    return data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor



def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        data, masks, deltas, deltas_rev = [x.to(device) for x in batch]
        optimizer.zero_grad()
        imputations, loss = model(data, masks, deltas, deltas_rev)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            data, masks, deltas, deltas_rev = [x.to(device) for x in batch]
            imputations, loss = model(data, masks, deltas, deltas_rev)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def optimize_brits(data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor, device, n_trials=30):
    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        dataset = TensorDataset(data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = BRITS(input_dim=data_tensor.size(-1), hidden_dim=hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(10):
            train_epoch(model, train_loader, nn.MSELoss(), optimizer, device)

        val_loss = eval_epoch(model, val_loader, nn.MSELoss(), device)
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)

    return study.best_params


def main(df, optuna_mode=False, scaler_path="brits_scaler.gz", model_out_path="brits_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor = prepare_tensors(df, scaler_path, fit_scaler=True)

    if optuna_mode:
        best_params = optimize_brits(data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor, device)
        hidden_dim = best_params["hidden_dim"]
        lr = best_params["lr"]
        batch_size = best_params["batch_size"]
    else:
        hidden_dim = 128
        lr = 1e-3
        batch_size = 64

    dataset = TensorDataset(data_tensor, masks_tensor, deltas_tensor, deltas_rev_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BRITS(input_dim=data_tensor.size(-1), hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, nn.MSELoss(), optimizer, device)
        val_loss = eval_epoch(model, val_loader, nn.MSELoss(), device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), model_out_path)
    print(f"✅ Modèle entraîné et sauvegardé à {model_out_path}")

# -*- coding: utf-8 -*-
"""
Created on Fri May  2 11:02:22 2025

@author: mc
"""

# src/models/brits.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RITS(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RITS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.rnn_cell = nn.LSTMCell(input_dim * 2 + 2, hidden_dim)  # +1 pour delta, +1 pour poids_survie
        self.impute_layer = nn.Linear(hidden_dim, input_dim)

        self.temp_decay_layer = nn.Linear(1, hidden_dim)
        self.hist_reg = nn.Linear(hidden_dim, input_dim)
        self.feat_reg = nn.Linear(input_dim, input_dim)

    def forward(self, data, masks, deltas, ages, clusters, cluster_survival_fns):
        batch_size, seq_len, _ = data.size()
        device = data.device

        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)

        imputations = []
        losses = []

        for t in range(seq_len):
            x = data[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            a = ages[:, t]  # âge au temps t
            cl = clusters[:, t]  # cluster au temps t (entier)

            # Calcul poids_survie dynamiquement
            w_survie = torch.tensor([
                cluster_survival_fns[int(cl[i].item())](a[i].item()) if int(cl[i].item()) in cluster_survival_fns else 1.0
                for i in range(batch_size)
            ], dtype=torch.float32, device=device).unsqueeze(1)  # (B, 1)

            h_decay = torch.exp(-F.relu(self.temp_decay_layer(d)))
            h = h * h_decay

            x_h = self.hist_reg(h)
            x_c = m * x + (1 - m) * x_h

            inputs = torch.cat([x_c, m, d, w_survie], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            x_h = self.hist_reg(h)
            x_f = self.feat_reg(x)

            x_combined = m * x + (1 - m) * (x_h + x_f)

            imputations.append(x_combined.unsqueeze(1))

            # Ponderer la perte avec poids_survie
            loss = torch.sum((x - x_h) ** 2 * m * w_survie) / (torch.sum(m) + 1e-5)
            losses.append(loss)

        imputations = torch.cat(imputations, dim=1)
        loss = torch.stack(losses).mean()

        return imputations, loss


class BRITS(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BRITS, self).__init__()
        self.rits_f = RITS(input_dim, hidden_dim)
        self.rits_b = RITS(input_dim, hidden_dim)

    def forward(self, data, masks, deltas, deltas_rev, ages, clusters, cluster_survival_fns):
        imputations_f, loss_f = self.rits_f(data, masks, deltas, ages, clusters, cluster_survival_fns)

        imputations_b, loss_b = self.rits_b(
            torch.flip(data, dims=[1]),
            torch.flip(masks, dims=[1]),
            torch.flip(deltas_rev, dims=[1]),
            torch.flip(ages, dims=[1]),
            torch.flip(clusters, dims=[1]),
            cluster_survival_fns
        )

        imputations_b = torch.flip(imputations_b, dims=[1])

        imputations = (imputations_f + imputations_b) / 2
        loss = (loss_f + loss_b) / 2

        return imputations, loss

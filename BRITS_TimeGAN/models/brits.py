# src/models/brits.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RITS(nn.Module):
    """
    Recurrent Imputation for Time Series (RITS) model.
    Imputation unidirectionnelle.
    """

    def __init__(self, input_dim, hidden_dim):
        super(RITS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder RNN
        self.rnn_cell = nn.LSTMCell(input_dim * 2 + 1, hidden_dim)

        # Régression pour l'imputation
        self.impute_layer = nn.Linear(hidden_dim, input_dim)

        # Décalage temporel (input: delta time)
        self.temp_decay_layer = nn.Linear(1, hidden_dim)
        self.hist_reg = nn.Linear(hidden_dim, input_dim)
        self.feat_reg = nn.Linear(input_dim, input_dim)

    def forward(self, data, masks, deltas):
        """
        data: [batch_size, seq_len, input_dim] — valeurs observées (avec NaN)
        masks: [batch_size, seq_len, input_dim] — masque (1 = observé, 0 = manquant)
        deltas: [batch_size, seq_len, 1] — temps écoulé depuis la dernière observation
        """

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

            # Estimation à partir du passé
            h_decay = torch.exp(-F.relu(self.temp_decay_layer(d)))
            h = h * h_decay

            # Estimation des valeurs manquantes
            x_h = self.hist_reg(h)
            x_c = m * x + (1 - m) * x_h  # On utilise la prédiction si c'est manquant

            # Mise à jour de l'état caché
            inputs = torch.cat([x_c, m, d], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            # Correction
            x_h = self.hist_reg(h)
            x_f = self.feat_reg(x)

            x_combined = m * x + (1 - m) * (x_h + x_f)

            imputations.append(x_combined.unsqueeze(1))

            # Perte de reconstruction
            loss = torch.sum((x - x_h) ** 2 * m) / (torch.sum(m) + 1e-5)
            losses.append(loss)

        imputations = torch.cat(imputations, dim=1)
        loss = torch.stack(losses).mean()

        return imputations, loss

class BRITS(nn.Module):
    """
    Bidirectional Recurrent Imputation for Time Series (BRITS) model.
    Combine un RITS avant et arrière.
    """

    def __init__(self, input_dim, hidden_dim):
        super(BRITS, self).__init__()
        self.rits_f = RITS(input_dim, hidden_dim)
        self.rits_b = RITS(input_dim, hidden_dim)

    def forward(self, data, masks, deltas, deltas_rev):
        """
        data: valeurs observées
        masks: masque de présence
        deltas: temps écoulés en avant
        deltas_rev: temps écoulés en arrière
        """

        # Forward
        imputations_f, loss_f = self.rits_f(data, masks, deltas)

        # Backward
        imputations_b, loss_b = self.rits_b(
            torch.flip(data, dims=[1]),
            torch.flip(masks, dims=[1]),
            torch.flip(deltas_rev, dims=[1])
        )

        imputations_b = torch.flip(imputations_b, dims=[1])

        # Fusion des imputations forward et backward
        imputations = (imputations_f + imputations_b) / 2

        loss = (loss_f + loss_b) / 2

        return imputations, loss

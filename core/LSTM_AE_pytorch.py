# ============================================================
# LSTM_AE_pytorch.py
# LSTM Autoencoder w PyTorchu
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMAENetwork(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ENCODER
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # DECODER
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            batch_first=True,
        )

    def forward(self, x):
        # x: (batch, timesteps, features)

        # -------- Encoder --------
        _, (h_n, _) = self.encoder(x)
        # h_n: (1, batch, hidden_dim)
        latent = h_n[-1]

        # -------- Repeat vector --------
        seq_len = x.size(1)
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)

        # -------- Decoder --------
        decoded, _ = self.decoder(repeated)
        return decoded


class LSTM_AE:
    """
    LSTM Autoencoder w PyTorchu
    Kompatybilny ze wszystkimi wariantami PARAM w SKAB.

    Obsługiwane formaty PARAM:
    - [hidden_dim, lr, validation_split]
    - [hidden_dim, lr, batch_size]
    - [hidden_dim, lr, batch_size, epochs]
    """

    def __init__(self, params=None):
        if params is None:
            params = [64, 1e-3, 64, 30]

        self.hidden_dim = int(params[0])
        self.lr = params[1]

        # ---- DOMYŚLNE WARTOŚCI ----
        self.batch_size = 64
        self.epochs = 30

        # ---- ROZPOZNANIE PARAM[2] ----
        if len(params) >= 3:
            if isinstance(params[2], float) and params[2] < 1:
                # to jest validation_split z TF → ignorujemy w PyTorch
                pass
            else:
                self.batch_size = int(params[2])

        # ---- OPCJONALNE EPOCHS ----
        if len(params) >= 4:
            self.epochs = int(params[3])

        self.device = torch.device("cpu")
        self.model = None
    def fit(self, X):
        n_features = X.shape[2]

        self.model = _LSTMAENetwork(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)

                optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, xb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(X_tensor)
            return output.cpu().numpy()

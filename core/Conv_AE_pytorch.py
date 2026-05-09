# ============================================================
# Conv_AE_pytorch.py
# Convolutional Autoencoder w PyTorchu
# Odpowiednik core/Conv_AE.py (TensorFlow)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _ConvAENetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, n_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Conv_AE:
    """
    Convolutional Autoencoder (PyTorch)
    Interfejs zgodny z wersją TensorFlow:
    - fit(X)
    - predict(X)
    """

    def __init__(self, lr=1e-3, batch_size=128, epochs=20):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cpu")
        self.model = None

    def fit(self, X):
        # X shape: (samples, timesteps, features)
        n_features = X.shape[2]

        # build model
        self.model = _ConvAENetwork(n_features).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss()

        # transpose: (samples, features, timesteps)
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
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
            X_tensor = (
                torch.tensor(X, dtype=torch.float32)
                .permute(0, 2, 1)
                .to(self.device)
            )
            output = self.model(X_tensor)

            # back to (samples, timesteps, features)
            return output.permute(0, 2, 1).cpu().numpy()
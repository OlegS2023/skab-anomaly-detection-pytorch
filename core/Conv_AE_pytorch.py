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

    def __init__(self, lr=1e-3, batch_size=128, epochs=40):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cpu")
        self.model = None

    def fit(self, X, val_split=0.2):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split

        # =========================
        # HISTORY (ważne)
        # =========================
        self.history = {
            "train_loss": [],
            "val_loss": []
        }

        # =========================
        # model setup
        # =========================
        n_features = X.shape[2]
        self.model = _ConvAENetwork(n_features).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss()

        # =========================
        # tensor
        # =========================
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        dataset = TensorDataset(X_tensor, X_tensor)

        # =========================
        # train/val split
        # =========================
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # =========================
        # training loop
        # =========================
        self.model.train()

        for epoch in range(self.epochs):

            # ---- TRAIN ----
            train_loss = 0.0
            for xb, _ in train_loader:
                xb = xb.to(self.device)

                optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, xb)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xb.size(0)

            train_loss /= train_size

            # ---- VAL ----
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(self.device)
                    output = self.model(xb)
                    loss = criterion(output, xb)

                    val_loss += loss.item() * xb.size(0)

            val_loss /= val_size

            self.model.train()

            # =========================
            # HISTORY SAVE
            # =========================
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1}/{self.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f}")

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
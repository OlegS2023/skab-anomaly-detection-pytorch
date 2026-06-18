# ============================================================
# LSTM_VAE_pytorch.py
# LSTM Variational Autoencoder w PyTorchu
# Odpowiednik core/LSTM_VAE.py (TensorFlow/Keras)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMVAENetwork(nn.Module):
    def __init__(self, n_features, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # ENCODER
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Warstwy mu i log_var (reparametrization trick)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Projekcja latent -> hidden (wejscie dekodera)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)

        # DECODER
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_features,
            batch_first=True,
        )

    def reparametrize(self, mu, log_var):
        """Reparametrization trick: z = mu + eps * std"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (batch, timesteps, features)

        # -------- Encoder --------
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]  # (batch, hidden_dim)

        # -------- Latent space --------
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.reparametrize(mu, log_var)

        # -------- Repeat vector --------
        seq_len = x.size(1)
        z_proj = self.fc_decode(z)
        repeated = z_proj.unsqueeze(1).repeat(1, seq_len, 1)

        # -------- Decoder --------
        decoded, _ = self.decoder(repeated)

        return decoded, mu, log_var


class LSTM_VAE:
    """
    LSTM Variational Autoencoder w PyTorchu.
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
        self.latent_dim = max(8, self.hidden_dim // 2)

        # ---- DOMYSLNE WARTOSCI ----
        self.batch_size = 64
        self.epochs = 40

        # ---- ROZPOZNANIE PARAM[2] ----
        if len(params) >= 3:
            if isinstance(params[2], float) and params[2] < 1:
                # to jest validation_split z TF -> ignorujemy w PyTorch
                pass
            else:
                self.batch_size = int(params[2])

        # ---- OPCJONALNE EPOCHS ----
        if len(params) >= 4:
            self.epochs = int(params[3])

        self.device = torch.device("cpu")
        self.model = None

    @staticmethod
    def _vae_loss(x_recon, x, mu, log_var):
        """
        Laczna strata VAE = Reconstruction loss (L1) + KL divergence
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        """
        recon_loss = nn.L1Loss()(x_recon, x)
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        return recon_loss + kl_loss

    def fit(self, X):
        n_features = X.shape[2]
        self.model = _LSTMVAENetwork(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        n = len(X_tensor)
        n_val = int(n * 0.2)
        n_train = n - n_val

        X_train = X_tensor[:n_train]
        X_val = X_tensor[n_train:]

        dataset = TensorDataset(X_train, X_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        self.history = {"train_loss": [], "val_loss": []}
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, _ in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad()
                x_recon, mu, log_var = self.model(xb)
                loss = self._vae_loss(x_recon, xb, mu, log_var)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.model.eval()
            with torch.no_grad():
                x_recon_val, mu_val, log_var_val = self.model(X_val.to(self.device))
                val_loss = self._vae_loss(x_recon_val, X_val.to(self.device), mu_val, log_var_val).item()
            self.model.train()
            self.history["train_loss"].append(epoch_loss / len(loader))
            self.history["val_loss"].append(val_loss)
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            x_recon, _, _ = self.model(X_tensor)
            return x_recon.cpu().numpy()
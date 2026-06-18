# ============================================================
# Vanilla_LSTM_pytorch.py
# LSTM zaimplementowany w PyTorchu
# Odpowiednik oryginalnego Vanilla_LSTM.py (TensorFlow/Keras)
# ============================================================

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── 1. ARCHITEKTURA SIECI ────────────────────────────────────
# Keras: model.add(LSTM(100, return_sequences=True))
#        model.add(LSTM(100))
#        model.add(Dense(n_features))
#
# PyTorch: nn.LSTM zwraca (output, (h_n, c_n))
#   - output: wszystkie kroki czasowe  kształt: (batch, seq, hidden)
#   - h_n:    ostatni ukryty stan      kształt: (n_layers, batch, hidden)
# My chcemy tylko ostatni krok → bierzemy output[:, -1, :]

class _LSTMNetwork(nn.Module):

    def __init__(self, n_features, hidden_size=100):
        super().__init__()

        # Keras: LSTM(100, return_sequences=True) + LSTM(100)
        # PyTorch: jeden nn.LSTM z num_layers=2 robi to samo
        # batch_first=True → kształt (batch, seq, features)
        # zamiast domyślnego (seq, batch, features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        # Keras: Dense(n_features)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x kształt: (batch, n_steps, n_features)
        out, _ = self.lstm(x)
        # out kształt: (batch, n_steps, hidden_size)
        # bierzemy TYLKO ostatni krok czasowy
        last_step = out[:, -1, :]
        # last_step kształt: (batch, hidden_size)
        return self.fc(last_step)
        # wynik kształt: (batch, n_features) — przewidujemy następny krok


# ── 2. KLASA ZEWNĘTRZNA ───────────────────────────────────────
# Zachowujemy identyczny interfejs: fit(X, y) i predict(data)

class Vanilla_LSTM:
    """
    LSTM w PyTorchu — identyczny interfejs jak wersja TensorFlow.

    params : list [N_STEPS, EPOCHS, BATCH_SIZE, VAL_SPLIT]
    """

    def __init__(self, params):
        self.params = params
        self.model = None
        self.device = torch.device("cpu")

    def _set_seed(self, seed=0):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def fit(self, X, y):
        """
        X kształt: (n_próbek, N_STEPS, n_cech)
        y kształt: (n_próbek, n_cech)
        — dokładnie to co zwraca split_sequences z notebooka
        """
        self._set_seed(0)
        self.n_features = X.shape[2]

        self.model = _LSTMNetwork(
            n_features=self.n_features
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters())

        # ReduceLROnPlateau — odpowiednik Keras callback
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=5, min_lr=0.0001
        )

        criterion = nn.L1Loss()  # loss="mae"

        # Podział train/val
        n = len(X)
        n_val = int(n * self.params[3])
        n_train = n - n_val

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        X_train = X_tensor[:n_train]
        y_train = y_tensor[:n_train]
        X_val   = X_tensor[n_train:]
        y_val   = y_tensor[n_train:]

        # WAŻNE: shuffle=False — dane czasowe muszą być w kolejności!
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.params[2],
            shuffle=False,
        )

        # Early stopping — patience=10 jak w oryginale
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        self.history = {
        "train_loss": [],
        "val_loss": [],
        }

        for epoch in range(self.params[1]):

            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Walidacja
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val.to(self.device))
                val_loss = criterion(val_pred, y_val.to(self.device)).item()

            scheduler.step(val_loss)

            self.history["train_loss"].append(
            train_loss / len(train_loader)
            )
            self.history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def predict(self, data):
        """
        data kształt: (n_próbek, N_STEPS, n_cech)
        zwraca: numpy array kształt (n_próbek, n_cech)
        — identycznie jak oryginał
        """
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
            return self.model(tensor).cpu().numpy()
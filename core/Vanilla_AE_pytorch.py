# ============================================================
# Vanilla_AE_pytorch.py
# ============================================================

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── 1. ARCHITEKTURA SIECI ────────────────────────────────────


class _AENetwork(nn.Module):
    """Sama sieć neuronowa — enkoder + dekoder."""

    def __init__(self, input_dim, hidden1, bottleneck, hidden2):
        super().__init__()

        # ENKODER
        # Keras: Dense(hidden1) -> BatchNorm -> ReLU
        # PyTorch: nn.Linear odpowiada Dense, reszta tak samo
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden2),  # bottleneck (activation=linear)
        )

        # DEKODER — lustrzane odbicie enkodera
        self.decoder = nn.Sequential(
            nn.Linear(hidden2, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),   # wyjście (activation=linear)
        )

    def forward(self, x):
        # forward() to odpowiednik "przepływu" przez model w Keras
        # x -> enkoder -> dekoder -> rekonstrukcja
        z = self.encoder(x)
        return self.decoder(z)


# ── 2. KLASA ZEWNĘTRZNA (taki sam interfejs jak oryginał) ─────
# Notebook używa: model.fit(data) i model.predict(data)
# Zachowujemy dokładnie ten sam interfejs żeby notebook
# działał bez żadnych zmian!

class Vanilla_AE:
    """
    Feed-forward autoencoder w PyTorchu.
    Identyczny interfejs jak oryginalna wersja TensorFlow.

    params : list [hidden1, bottleneck, hidden2, lr, batch_size]
    """

    def __init__(self, params):
        self.param = params
        self.model = None
        self.device = torch.device("cpu")  # CPU — brak GPU

    def _set_seed(self, seed=0):
        # Odpowiednik metody _Random() z oryginału
        # Ustawiamy seed wszędzie żeby wyniki były powtarzalne
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def fit(
        self,
        data,
        early_stopping=True,
        validation_split=0.2,
        epochs=40,
        verbose=0,
        shuffle=True,
    ):
        """
        Trenuje autoencoder na danych.
        Parametry identyczne jak w oryginale.
        """
        self._set_seed(0)
        self.shape = data.shape[1]

        # Budujemy sieć — teraz znamy wymiar wejścia
        self.model = _AENetwork(
            input_dim=self.shape,
            hidden1=self.param[0],
            bottleneck=self.param[1],
            hidden2=self.param[2],
        ).to(self.device)

        # ── OPTYMALIZATOR ──────────────────────────────────────
        # Keras: Adam(self.param[3])
        # PyTorch: torch.optim.Adam(parametry_modelu, lr=...)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param[3])

        # ── FUNKCJA STRATY ─────────────────────────────────────
        # Keras: loss="mae"
        # PyTorch: nn.L1Loss() to dokładnie MAE
        criterion = nn.L1Loss()

        # ── PODZIAŁ NA TRAIN/VALIDATION ────────────────────────
        n = len(data)
        n_val = int(n * validation_split)
        n_train = n - n_val

        data_tensor = torch.tensor(data, dtype=torch.float32)

        if shuffle:
            idx = torch.randperm(n)
            data_tensor = data_tensor[idx]

        train_data = data_tensor[:n_train]
        val_data   = data_tensor[n_train:]

        # TensorDataset + DataLoader = odpowiednik batch_size w Keras
        train_loader = DataLoader(
            TensorDataset(train_data, train_data),
            batch_size=self.param[4],
            shuffle=shuffle,
        )

        # ── PĘTLA TRENINGOWA ───────────────────────────────────
        # To jest największa różnica vs Keras!
        # Keras: model.fit(data, data, epochs=40)
        # PyTorch: piszemy pętlę sami — daje nam pełną kontrolę

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 3  # taka sama jak EarlyStopping(patience=3)

        for epoch in range(epochs):

            # --- tryb treningowy ---
            # WAŻNE: BatchNorm i Dropout działają inaczej
            # w trybie train vs eval — trzeba to przełączać!
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)

                optimizer.zero_grad()   # zeruj gradienty (inaczej się sumują!)
                output = self.model(X_batch)
                loss = criterion(output, X_batch)
                loss.backward()         # oblicz gradienty
                optimizer.step()        # zaktualizuj wagi

                train_loss += loss.item()

            # --- walidacja ---
            self.model.eval()  # tryb ewaluacji — BatchNorm używa statystyk
            with torch.no_grad():  # nie liczymy gradientów przy walidacji
                val_output = self.model(val_data.to(self.device))
                val_loss = criterion(val_output, val_data.to(self.device)).item()

            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} — "
                      f"train_loss: {train_loss/len(train_loader):.4f} — "
                      f"val_loss: {val_loss:.4f}")

            # --- early stopping ---
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose > 0:
                            print(f"Early stopping na epoce {epoch+1}")
                        break

    def predict(self, data):
        """
        Rekonstruuje dane przez autoencoder.
        Zwraca numpy array — identycznie jak oryginał.
        """
        self.model.eval()
        with torch.no_grad():
            # numpy -> tensor -> przez model -> z powrotem numpy
            tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
            output = self.model(tensor)
            return output.cpu().numpy()

# ============================================================
# trainer.py
# Zbieranie historii uczenia i rysowanie krzywych
# Działa z: Vanilla_AE, Conv_AE, Vanilla_LSTM, LSTM_AE, LSTM_VAE
# ============================================================

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.rcParams["font.family"] = "DejaVu Sans"


# ============================================================
# 1. PATCH – dodaje zbieranie historii do istniejącego fit()
# ============================================================

def patch_vanilla_ae(model_instance):
    """
    Modyfikuje obiekt Vanilla_AE tak żeby fit() zbierał historię.
    Wywołaj RAZ przed model.fit(...).

    Użycie:
        model = Vanilla_AE(BEST_PARAMS)
        patch_vanilla_ae(model)
        model.fit(X_train)
        plot_learning_curve(model.history, 'Vanilla AE', 'images/lc_vanilla_ae.png')
    """
    original_fit = model_instance.fit.__func__  # oryginalna metoda

    def fit_with_history(
        self,
        data,
        early_stopping=True,
        validation_split=0.2,
        epochs=40,
        verbose=0,
        shuffle=True,
    ):
        self._set_seed(0)
        self.shape = data.shape[1]

        from torch.utils.data import DataLoader, TensorDataset

        self.model = self.__class__.__mro__[0]  # placeholder
        # Odbuduj model tak jak w oryginale
        from Vanilla_AE_pytorch import _AENetwork
        self.model = _AENetwork(
            input_dim=self.shape,
            hidden1=self.param[0],
            bottleneck=self.param[1],
            hidden2=self.param[2],
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param[3])
        criterion = nn.L1Loss()

        n = len(data)
        n_val = int(n * validation_split)
        n_train = n - n_val

        data_tensor = torch.tensor(data, dtype=torch.float32)
        if shuffle:
            idx = torch.randperm(n)
            data_tensor = data_tensor[idx]

        train_data = data_tensor[:n_train]
        val_data   = data_tensor[n_train:]

        train_loader = DataLoader(
            TensorDataset(train_data, train_data),
            batch_size=self.param[4],
            shuffle=shuffle,
        )

        # --- HISTORIA ---
        self.history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 3

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, _ in train_loader:
                X_batch = X_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, X_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_loader)

            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_data.to(self.device))
                val_loss = criterion(val_output, val_data.to(self.device)).item()

            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(val_loss)

            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} — "
                      f"train: {avg_train:.4f}  val: {val_loss:.4f}")

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

    import types
    model_instance.fit = types.MethodType(fit_with_history, model_instance)
    return model_instance


# ============================================================
# 2. UNIWERSALNA PĘTLA TRENINGOWA dla modeli PyTorch
#    (Conv_AE, Vanilla_LSTM, LSTM_AE, LSTM_VAE)
# ============================================================

def train_and_collect(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    n_epochs=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True,
    is_vae=False,
    beta=1.0,
):
    """
    Pętla treningowa zbierająca historię loss.

    Parametry:
        model        – model PyTorch (nn.Module)
        train_loader – DataLoader zbioru treningowego
        val_loader   – DataLoader zbioru walidacyjnego
        optimizer    – np. torch.optim.Adam(...)
        criterion    – np. nn.MSELoss()
        n_epochs     – liczba epok
        device       – 'cuda' lub 'cpu'
        verbose      – czy drukować postęp co 10 epok
        is_vae       – True dla LSTM VAE (dodaje człon KL do straty)
        beta         – waga członu KL (tylko dla VAE)

    Zwraca:
        history : dict {"train_loss": [...], "val_loss": [...]}
    """
    model = model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):

        # --- TRENING ---
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()

            if is_vae:
                x_hat, mu, log_var = model(x)
                recon = criterion(x_hat, x)
                kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon + beta * kl
            else:
                x_hat = model(x)
                loss = criterion(x_hat, x)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        epoch_train = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train)

        # --- WALIDACJA ---
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                if is_vae:
                    x_hat, mu, log_var = model(x)
                    recon = criterion(x_hat, x)
                    kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = recon + beta * kl
                else:
                    x_hat = model(x)
                    loss = criterion(x_hat, x)
                running_val += loss.item() * x.size(0)

        epoch_val = running_val / len(val_loader.dataset)
        history["val_loss"].append(epoch_val)

        if verbose and epoch % 10 == 0:
            print(f"Epoka {epoch:3d}/{n_epochs}  "
                  f"train={epoch_train:.6f}  val={epoch_val:.6f}")

    return history


# ============================================================
# 3. RYSOWANIE KRZYWYCH UCZENIA
# ============================================================

def plot_learning_curve(history, model_name, save_path=None):
    """
    Rysuje krzywą uczenia na podstawie historii zebranej przez
    train_and_collect() lub patch_vanilla_ae().

    Parametry:
        history    – dict {"train_loss": [...], "val_loss": [...]}
        model_name – nazwa modelu do tytułu wykresu
        save_path  – ścieżka zapisu PNG, np. 'images/lc_vanilla_ae.png'
    """
    train_losses = history["train_loss"]
    val_losses   = history["val_loss"]
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, train_losses,
            label="Zbiór treningowy",
            color="steelblue", linewidth=1.8)
    ax.plot(epochs, val_losses,
            label="Zbiór walidacyjny",
            color="tomato", linewidth=1.8, linestyle="--")

    ax.set_xlabel("Epoka", fontsize=11)
    ax.set_ylabel("Strata (MAE / MSE)", fontsize=11)
    ax.set_title(f"Krzywe uczenia – {model_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Zapisano: {save_path}")

    plt.show()
    return fig


# ============================================================
# 4. PRZYKŁADY UŻYCIA
# ============================================================

"""
# ── Vanilla AE (patch istniejącej klasy) ──────────────────────
from Vanilla_AE_pytorch import Vanilla_AE
from trainer import patch_vanilla_ae, plot_learning_curve

model = Vanilla_AE(BEST_PARAMS)
patch_vanilla_ae(model)          # <-- dodaje zbieranie historii
model.fit(X_train, verbose=1)
plot_learning_curve(model.history, 'Vanilla AE', 'images/lc_vanilla_ae.png')


# ── Conv AE / Vanilla LSTM / LSTM AE ─────────────────────────
from trainer import train_and_collect, plot_learning_curve

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

history = train_and_collect(
    model, train_loader, val_loader,
    optimizer, criterion,
    n_epochs=50,
    verbose=True,
)
plot_learning_curve(history, 'Conv AE', 'images/lc_conv_ae.png')


# ── LSTM VAE (z członem KL) ───────────────────────────────────
history = train_and_collect(
    model, train_loader, val_loader,
    optimizer, criterion,
    n_epochs=50,
    is_vae=True,
    beta=1.0,
)
plot_learning_curve(history, 'LSTM VAE', 'images/lc_lstm_vae.png')
"""

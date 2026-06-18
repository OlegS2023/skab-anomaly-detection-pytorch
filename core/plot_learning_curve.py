# ============================================================
# plot_learning_curve.py
# Wrzuć do katalogu core/ obok modeli
# ============================================================

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def plot_learning_curve(model, model_name, save_path=None):
    """
    Rysuje krzywe uczenia na podstawie model.history.

    Parametry:
        model      – wytrenowany model (musi mieć atrybut .history)
        model_name – nazwa do tytułu, np. 'Vanilla AE'
        save_path  – ścieżka zapisu PNG, np. 'images/lc_vanilla_ae.png'
    """
    train_losses = model.history["train_loss"]
    val_losses   = model.history["val_loss"]
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, train_losses,
            label="Zbiór treningowy",
            color="steelblue", linewidth=1.8)
    ax.plot(epochs, val_losses,
            label="Zbiór walidacyjny",
            color="tomato", linewidth=1.8, linestyle="--")

    ax.set_xlabel("Epoka", fontsize=11)
    ax.set_ylabel("Strata (MAE)", fontsize=11)
    ax.set_title(f"Krzywe uczenia – {model_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Zapisano: {save_path}")

    plt.show()

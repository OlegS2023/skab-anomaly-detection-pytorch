import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve

RESULTS_DIR = Path('notebooks/results')
IMAGES_DIR  = Path('notebooks/images')
IMAGES_DIR.mkdir(exist_ok=True)

MODELS = {
    'Conv AE':          'scores_conv_ae.npy',
    'Vanilla AE':       'scores_vanilla_ae.npy',
    'Vanilla LSTM':     'scores_vanilla_lstm.npy',
    'LSTM AE':          'scores_lstm_ae.npy',
    'LSTM VAE':         'scores_lstm_vae.npy',
    'Isolation Forest': 'scores_isolation_forest.npy',
    'MSET':             'scores_mset.npy',
}

LABELS = {
    'Conv AE':          'labels_conv_ae.npy',
    'Vanilla AE':       'labels_vanilla_ae.npy',
    'Vanilla LSTM':     'labels_vanilla_lstm.npy',
    'LSTM AE':          'labels_lstm_ae.npy',
    'LSTM VAE':         'labels_lstm_vae.npy',
    'Isolation Forest': 'labels_isolation_forest.npy',
    'MSET':             'labels_mset.npy',
}

COLORS = {
    'Conv AE':          '#1f77b4',  # niebieski
    'Vanilla AE':       '#9467bd',  # fioletowy
    'Vanilla LSTM':     '#7f7f7f',  # szary
    'LSTM AE':          '#d62728',  # czerwony
    'LSTM VAE':         '#2ca02c',  # zielony
    'Isolation Forest': '#ff7f0e',  # pomarańczowy
    'MSET':             '#e377c2',  # różowy
}
# ============================================================
# SAFE FLATTEN (KLUCZ FIX)
# ============================================================
def safe_flatten(scores_list, labels_list):
    all_scores = []
    all_labels = []

    for s, l in zip(scores_list, labels_list):
        s = np.asarray(s).ravel()
        l = np.asarray(l).ravel()

        m = min(len(s), len(l))
        all_scores.append(s[:m])
        all_labels.append(l[:m])

    return np.concatenate(all_scores), np.concatenate(all_labels)

# ============================================================
# Wczytanie danych
# ============================================================
data = {}

for model_name in MODELS:

    scores_path = RESULTS_DIR / MODELS[model_name]
    labels_path = RESULTS_DIR / LABELS[model_name]

    if not scores_path.exists() or not labels_path.exists():
        print(f"BRAK: {model_name}")
        continue

    scores = np.load(scores_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    scores_flat, labels_flat = safe_flatten(scores, labels)

    # 🔴 dodatkowy sanity check (WAŻNE)
    print(f"{model_name}:")
    print(" scores:", len(scores_flat), " labels:", len(labels_flat))

    data[model_name] = (scores_flat, labels_flat)

# ============================================================
# ROC + PR
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for model_name, (scores_flat, labels_flat) in data.items():

    # ROC
    fpr, tpr, _ = roc_curve(labels_flat, scores_flat)
    roc_auc = auc(fpr, tpr)

    axes[0].plot(
        fpr, tpr,
        color=COLORS[model_name],
        lw=2,
        label=f"{model_name} (AUC={roc_auc:.3f})"
    )

    # PR
    precision, recall, _ = precision_recall_curve(labels_flat, scores_flat)
    pr_auc = auc(recall, precision)

    axes[1].plot(
        recall, precision,
        color=COLORS[model_name],
        lw=2,
        label=f"{model_name} (PR={pr_auc:.3f})"
    )

# ============================================================
# ROC baseline
# ============================================================
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_title("ROC comparison")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].legend()

# ============================================================
# PR baseline
# ============================================================
if data:
    baseline = list(data.values())[0][1].mean()
    axes[1].axhline(baseline, linestyle='--', color='k')

axes[1].set_title("Precision-Recall comparison")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()

# ============================================================
# SAVE
# ============================================================
plt.tight_layout()
plt.savefig(IMAGES_DIR / "roc_pr_all_models.png", dpi=150)
plt.show()

print("Zapisano: images/roc_pr_all_models.png")
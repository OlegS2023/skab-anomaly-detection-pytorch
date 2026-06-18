# ============================================================
# plot_roc_pr.py
# Krzywe ROC, PR oraz metryki: AUC-ROC, PR-AUC, MCC
# Wrzuć do katalogu core/ obok modeli
#
# Użycie w notebooku — dopisz PO pętli inference:
#
#   from core.plot_roc_pr import plot_roc_pr
#
#   metrics = plot_roc_pr(
#       scores_list  = all_scores,
#       labels_list  = true_outlier,
#       model_name   = 'Vanilla AE',
#       save_roc     = 'images/roc_vanilla_ae.png',
#       save_pr      = 'images/pr_vanilla_ae.png',
#   )
#   print(metrics)
#   # {'auc_roc': 0.xxxx, 'pr_auc': 0.xxxx, 'mcc': 0.xxxx}
# ============================================================

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef,
)

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def plot_roc_pr(scores_list, labels_list, model_name,
                save_roc=None, save_pr=None):
    """
    Rysuje krzywą ROC i krzywą PR, oblicza AUC-ROC, PR-AUC i MCC.
    Agreguje wyniki ze wszystkich plików testowych SKAB.

    Parametry:
        scores_list  – lista pd.Series z residuals_test (surowy anomaly score)
                       jeden element na plik testowy — zbierasz w all_scores
        labels_list  – lista pd.Series z y_test["anomaly"] (0/1)
                       jeden element na plik testowy — to jest true_outlier
        model_name   – nazwa modelu do tytułu wykresów
        save_roc     – ścieżka zapisu PNG krzywej ROC (opcjonalna)
        save_pr      – ścieżka zapisu PNG krzywej PR  (opcjonalna)

    Zwraca:
        dict z kluczami 'auc_roc', 'pr_auc', 'mcc'
    """
    # Sklejamy wszystkie pliki testowe w jeden wektor
    all_scores = np.concatenate([s.values for s in scores_list])
    all_labels = np.concatenate([l.values for l in labels_list])

    # ── KRZYWA ROC ──────────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auc_roc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr,
            color="steelblue", linewidth=2,
            label=f"AUC-ROC = {auc_roc:.3f}")
    ax.plot([0, 1], [0, 1],
            color="gray", linewidth=1, linestyle="--",
            label="Losowy klasyfikator")
    ax.set_xlabel("Odsetek fałszywych alarmów (FPR)", fontsize=11)
    ax.set_ylabel("Czułość (TPR)", fontsize=11)
    ax.set_title(f"Krzywa ROC – {model_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_roc:
        fig.savefig(save_roc, dpi=150, bbox_inches="tight")
        print(f"Zapisano ROC: {save_roc}")
    plt.show()

    # ── KRZYWA PR ────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    pr_auc = average_precision_score(all_labels, all_scores)
    baseline = all_labels.mean()

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(recall, precision,
             color="tomato", linewidth=2,
             label=f"PR-AUC = {pr_auc:.3f}")
    ax2.axhline(y=baseline, color="gray", linewidth=1, linestyle="--",
                label=f"Baseline ({baseline:.2f})")
    ax2.set_xlabel("Czułość (Recall)", fontsize=11)
    ax2.set_ylabel("Precyzja (Precision)", fontsize=11)
    ax2.set_title(f"Krzywa Precision-Recall – {model_name}",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()
    if save_pr:
        fig2.savefig(save_pr, dpi=150, bbox_inches="tight")
        print(f"Zapisano PR: {save_pr}")
    plt.show()

    # ── MCC ──────────────────────────────────────────────────────
    # optymalny próg = maksymalizacja (TPR - FPR) z krzywej ROC
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    binary_pred = (all_scores >= optimal_threshold).astype(int)
    mcc = matthews_corrcoef(all_labels, binary_pred)

    print(f"\n{'='*50}")
    print(f"Model:    {model_name}")
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"MCC:      {mcc:.4f}  (próg optymalny = {optimal_threshold:.4f})")
    print(f"{'='*50}\n")

    return {
        "auc_roc": round(auc_roc, 4),
        "pr_auc":  round(pr_auc, 4),
        "mcc":     round(mcc, 4),
    }

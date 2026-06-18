# SKAB Anomaly Detection — PyTorch

A Master's thesis project comparing unsupervised anomaly detection methods on industrial time-series data. Models are implemented in PyTorch and benchmarked on [SKAB](https://github.com/waico/SKAB) — a real-world dataset from a pump testbed at the Skolkovo Institute of Science and Technology.

Two tasks are evaluated:

- **Outlier detection** — single-point anomalies (metric: F1 score)
- **Changepoint detection** — regime shifts (metric: NAB score)

---

## Results

### Full Model Comparison (Outlier Detection)

*Sorted by F1 score (higher is better). FAR = False Alarm Rate, MAR = Missing Alarm Rate (lower is better).*

| Model | F1 | AUC-ROC | PR-AUC | FAR, % | MAR, % |
|---|---|---|---|---|---|
| **Conv AE** (PyTorch) | **0.86** | **0.848** | **0.873** | 20.87 | 11.49 |
| MSET | 0.78 | 0.817 | 0.826 | 39.73 | 14.13 |
| T²+Q (PCA-based) | 0.76 | — | — | 26.62 | 24.92 |
| LSTM AE (PyTorch) | 0.67 | 0.786 | 0.810 | 13.18 | 44.34 |
| LSTM VAE (PyTorch) | 0.65 | 0.793 | 0.819 | 13.02 | 46.63 |
| T² Hotelling | 0.66 | — | — | 19.21 | 42.60 |
| Vanilla AE (PyTorch) | 0.40 | 0.786 | 0.811 | 3.28 | 73.91 |
| Isolation Forest | 0.29 | 0.737 | 0.748 | 2.56 | 82.89 |
| Vanilla LSTM (PyTorch) | 0.28 | 0.789 | 0.821 | 0.32 | 84.01 |

### Changepoint Detection (NAB Score)

*Sorted by NAB Standard (higher is better). Isolation Forest excels here despite low F1 — it reacts instantly to the first sign of change.*

| Model | NAB Standard | NAB LowFP | NAB LowFN |
|---|---|---|---|
| Isolation Forest | 26.16 | 19.50 | 30.82 |
| **Conv AE** (PyTorch) | **26.17** | 23.96 | 31.09 |
| LSTM VAE (PyTorch) | 23.72 | 19.28 | 26.57 |
| LSTM AE (PyTorch) | 25.28 | 21.66 | 27.62 |
| T²+Q (PCA-based) | 25.35 | 14.51 | 31.33 |
| T² Hotelling | 19.54 | 10.20 | 24.31 |
| MSET | 13.84 | 10.22 | 17.37 |
| Vanilla AE (PyTorch) | 9.52 | -0.30 | 13.17 |
| Vanilla LSTM (PyTorch) | 7.84 | 7.56 | 8.37 |

---

## Key Findings

**1. Conv AE dominates in precision (F1 = 0.86).** Convolutional layers effectively capture local spatio-temporal correlations in multi-sensor industrial data — outperforming all LSTM-based models.

**2. Isolation Forest wins in early warning (NAB = 26.16).** Its point-in-time detection reacts instantly to the first anomalous observation, while window-based models need time to accumulate reconstruction error.

**3. Classical MSET beats deep LSTM (F1 0.78 vs <0.65).** For physically stable systems like a pump, direct state-vector similarity works better than learned temporal predictions.

**4. LSTM VAE ≈ LSTM AE (both F1 = 0.64).** The added probabilistic complexity of the VAE did not yield measurable benefit on this dataset.

**Model selection guide:**

| Goal | Best choice |
|---|---|
| Minimum missed failures | Conv AE (lowest MAR) |
| Fastest early warning | Isolation Forest |
| No GPU / stable system | MSET |

---

## Visualizations

The `notebooks/figures/` and `notebooks/images/` directories contain all plots generated during the study:

- Multivariate sensor signals with anomaly regions
- Per-model ROC and Precision-Recall curves
- Learning curves (LSTM AE, Vanilla AE)
- Sensor histograms and correlation matrix
- Combined ROC/PR comparison across all models (`images/roc_pr_all_models.png`)

---

## Models

### Deep Learning — PyTorch

| Model | File | Status |
|---|---|---|
| Vanilla Autoencoder | `core/Vanilla_AE_pytorch.py` | ✅ Ready |
| LSTM Autoencoder | `core/LSTM_AE_pytorch.py` | ✅ Ready |
| Convolutional Autoencoder | `core/Conv_AE_pytorch.py` | ✅ Ready |
| Vanilla LSTM | `core/Vanilla_LSTM_pytorch.py` | ✅ Ready |
| LSTM Variational Autoencoder | `core/LSTM_VAE_pytorch.py` | ✅ Ready |
| MSCRED | `core/MSCRED_pytorch.py` | ✅ Ready |

### Classical Methods

| Model | File | Description |
|---|---|---|
| Isolation Forest | `core/Isolation_Forest.py` | Ensemble-based outlier detection |
| MSET | `core/MSET.py` | Multivariate State Estimation Technique |
| T² Hotelling | `core/t2.py` | Statistical process control |

---

## Project Structure

```
skab-anomaly-detection-pytorch/
├── core/                           # Model implementations
│   ├── Vanilla_AE_pytorch.py
│   ├── LSTM_AE_pytorch.py
│   ├── Conv_AE_pytorch.py
│   ├── Vanilla_LSTM_pytorch.py
│   ├── LSTM_VAE_pytorch.py
│   ├── MSCRED_pytorch.py
│   ├── Isolation_Forest.py
│   ├── MSET.py
│   ├── t2.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── plot_learning_curve.py
│   ├── plot_roc_pr.py
│   └── utils.py
├── data/                           # SKAB datasets (.csv)
├── images/                         # Combined result plots
├── notebooks/                      # Jupyter notebooks with experiments
│   ├── figures/                    # EDA and signal visualizations
│   ├── images/                     # Per-model ROC/PR plots
│   └── results/                    # Saved score/label arrays (.npy)
├── results/                        # Saved model outputs (.pkl)
├── requirements.txt
└── README.md
```

---

## Setup

**1. Create environment**

```bash
conda create -n skabpytorch python=3.11 -y
conda activate skabpytorch
```

**2. Install PyTorch (CPU)**

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

**3. Install dependencies**

```bash
conda install numpy pandas scikit-learn scipy matplotlib seaborn jupyter notebook ipykernel tqdm -c conda-forge -y
```

**4. Launch notebooks**

```bash
jupyter notebook
```

---

## Dataset

SKAB v0.9 contains 35 multivariate time-series files from IoT sensors (flow rate, pressure, vibration, temperature). Each file represents a single experiment with one anomaly event. Data is sampled at 1 Hz.

Data is located in the `data/` folder. Full dataset documentation: [waico/SKAB](https://github.com/waico/SKAB).

---

## License

GPL v3.0 — see [LICENSE](LICENSE).

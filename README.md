# SKAB Anomaly Detection — PyTorch

Deep learning models for multivariate time-series anomaly detection, implemented in PyTorch and evaluated on the [Skoltech Anomaly Benchmark (SKAB)](https://github.com/waico/SKAB).

---

## About

This project implements and evaluates deep learning architectures for anomaly detection on industrial IoT sensor data. The models are trained and tested on SKAB — a benchmark dataset collected from a physical pump testbed at the Skolkovo Institute of Science and Technology.

The benchmark covers two anomaly detection tasks:

- **Outlier detection** — detecting single-point anomalies
- **Changepoint detection** — detecting collective anomalies (regime shifts)

---

## Results

### Outlier Detection

*Sorted by F1 score (higher is better). FAR = False Alarm Rate, MAR = Missing Alarm Rate (lower is better).*

| Model | F1 | FAR, % | MAR, % |
|---|---|---|---|
| **Conv AE** (PyTorch) | **0.86** | 20.87 | 11.49 |
| MSET | 0.78 | 39.73 | 14.13 |
| T²+Q (PCA-based) | 0.76 | 26.62 | 24.92 |
| LSTM AE (PyTorch) | 0.67 | 13.18 | 44.34 |
| T² Hotelling | 0.66 | 19.21 | 42.60 |
| Vanilla AE (PyTorch) | 0.40 | 3.28 | 73.91 |
| Isolation Forest | 0.29 | 2.56 | 82.89 |
| Vanilla LSTM (PyTorch) | 0.28 | 0.32 | 84.01 |

### Changepoint Detection

*Sorted by NAB Standard (higher is better).*

| Model | NAB Standard | NAB LowFP | NAB LowFN |
|---|---|---|---|
| **Conv AE** (PyTorch) | **26.17** | 23.96 | 31.09 |
| Isolation Forest | 26.16 | 19.50 | 30.82 |
| LSTM AE (PyTorch) | 25.28 | 21.66 | 27.62 |
| T²+Q (PCA-based) | 25.35 | 14.51 | 31.33 |
| T² Hotelling | 19.54 | 10.20 | 24.31 |
| MSET | 13.84 | 10.22 | 17.37 |
| Vanilla AE (PyTorch) | 9.52 | -0.30 | 13.17 |
| Vanilla LSTM (PyTorch) | 7.84 | 7.56 | 8.37 |

---

## Models

### Deep Learning — PyTorch

| Model | File | Status |
|---|---|---|
| Vanilla Autoencoder | `core/Vanilla_AE_pytorch.py` | ✅ Ready |
| LSTM Autoencoder | `core/LSTM_AE_pytorch.py` | ✅ Ready |
| Convolutional Autoencoder | `core/Conv_AE_pytorch.py` | ✅ Ready |
| Vanilla LSTM | `core/Vanilla_LSTM_pytorch.py` | ✅ Ready |
| LSTM Variational Autoencoder | — | 🔄 In progress |
| MSCRED | — | 🔄 In progress |

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
│   ├── Vanilla_AE_pytorch.py       # PyTorch models
│   ├── LSTM_AE_pytorch.py
│   ├── Conv_AE_pytorch.py
│   ├── Vanilla_LSTM_pytorch.py
│   ├── Isolation_Forest.py         # Classical methods
│   ├── MSET.py
│   ├── t2.py
│   ├── metrics.py                  # Evaluation metrics
│   └── utils.py                    # Shared utilities
├── data/                           # SKAB datasets (.csv)
├── notebooks/                      # Jupyter notebooks with experiments
├── results/                        # Saved results and outputs
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

SKAB v0.9 contains 35 multivariate time-series files from IoT sensors (flow rate, pressure, vibration, temperature). Each file represents a single experiment with one anomaly event.

Data is located in the `data/` folder. Full dataset documentation: [waico/SKAB](https://github.com/waico/SKAB).

---

## License

GPL v3.0 — see [LICENSE](LICENSE).

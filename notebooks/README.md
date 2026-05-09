# SKAB Anomaly Detection — PyTorch

Deep learning models for multivariate time-series anomaly detection, implemented in PyTorch and evaluated on the [Skoltech Anomaly Benchmark (SKAB)](https://github.com/waico/SKAB).

---

## About

This project implements and evaluates deep learning architectures for anomaly detection on industrial IoT sensor data. The models are trained and tested on SKAB — a benchmark dataset collected from a physical pump testbed at the Skolkovo Institute of Science and Technology.

The benchmark covers two anomaly detection tasks:

- **Outlier detection** — detecting single-point anomalies
- **Changepoint detection** — detecting collective anomalies (regime shifts)

---

## Models

### Deep Learning — PyTorch

| Model | File | Status |
|---|---|---|
| Vanilla Autoencoder | `core/Vanilla_AE_pytorch.py` | ✅ Ready |
| LSTM Autoencoder | `core/LSTM_AE_pytorch.py` | ✅ Ready |
| Convolutional Autoencoder | `core/Conv_AE_pytorch.py` | ✅ Ready |
| Vanilla LSTM | `core/Vanilla_LSTM_pytorch.py` | ✅ Ready |
| LSTM Variational Autoencoder | `core/LSTM_VAE_pytorch.py` | 🔄 In progress |
| MSCRED | `core/MSCRED_pytorch.py` | 🔄 In progress |

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

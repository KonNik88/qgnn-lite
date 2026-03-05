# QNN Experiments: Hybrid Quantum–Classical Graph Neural Networks

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-green)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A small research project exploring **hybrid quantum–classical graph neural networks**.

The project compares two classification heads on top of a **GCN backbone**:

- **MLP head (classical baseline)**
- **VQC head (Variational Quantum Circuit)** implemented using PennyLane.

The goal is to understand the **practical trade-offs between classical and quantum components** in graph classification pipelines.

---

## Project Idea

The architecture tested in this project:

Graph → GCN Backbone → Graph Embedding → Prediction Head  
&nbsp;&nbsp;&nbsp;&nbsp;├── MLP (classical)  
&nbsp;&nbsp;&nbsp;&nbsp;└── VQC (quantum)

The backbone extracts graph features, while the prediction head performs final classification.

This allows a direct comparison between a classical neural head and a quantum circuit head.

---

## Datasets

Experiments were conducted on graph classification datasets from **TUDataset**:

| Dataset | Description |
|-------|-------------|
| MUTAG | Molecular graphs used for mutagenicity prediction |
| PROTEINS | Graph representations of protein structures |

---

## Experiments

Each configuration was evaluated using **three random seeds** to estimate performance stability.

Models compared:

| Model | Description |
|------|-------------|
| GCN + MLP | Classical baseline |
| GCN + VQC | Hybrid quantum–classical architecture |

---

## Results (Summary)

| Dataset | Model | Test Accuracy | Runtime |
|--------|------|--------------|---------|
| MUTAG | GCN + MLP | ~0.62 | ~1–3 s |
| MUTAG | GCN + VQC | ~0.52 | ~8 s |
| PROTEINS | GCN + MLP | ~0.67 | ~1 s |
| PROTEINS | GCN + VQC | ~0.65 | ~48 s |

### Observations

- Classical MLP performs slightly better on MUTAG.
- Both approaches achieve comparable accuracy on PROTEINS.
- VQC runtime is significantly higher due to **classical quantum circuit simulation**.

---

## Repository Structure

QNN/
├── data/                     # datasets (ignored in git)
├── docs/                     # project notes
├── projects/
│   └── qgnn-lite/
│       ├── configs/          # experiment configs
│       ├── notebooks/        # analysis notebooks
│       ├── results/          # experiment outputs
│       ├── scripts/          # experiment runners
│       └── src/              # model implementation
├── shared/                   # shared utilities
├── run_seeds.ps1             # experiment runner
├── requirements.txt
└── README.md

---

## Running Experiments

Activate environment:

conda activate qgnn_env

Run all seeds:

.\run_seeds.ps1

Results are saved to:

projects/qgnn-lite/results/results.csv

---

## Notebook Analysis

Experiment analysis is available in:

projects/qgnn-lite/notebooks/01_results_plots.ipynb

The notebook:
- loads experiment results
- computes aggregated metrics
- visualizes accuracy and runtime
- compares classical vs quantum heads

---

## Key Takeaway

Hybrid quantum–classical architectures can achieve competitive performance in graph classification tasks.

However, when executed via **classical quantum circuit simulation**, variational quantum circuits introduce substantial computational overhead.

---

## Future Work

Possible extensions:

- deeper quantum circuits
- larger graph datasets
- GPU → CPU hybrid execution pipelines
- execution on real quantum hardware

---

## License

MIT License

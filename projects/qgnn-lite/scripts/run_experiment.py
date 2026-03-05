# projects/qgnn-lite/scripts/run_experiment.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import yaml
from torch_geometric.datasets import TUDataset

# --- make imports work from monorepo root ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]        
QGNN_LITE_ROOT = THIS_FILE.parents[1]   

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(QGNN_LITE_ROOT) not in sys.path:
    sys.path.insert(0, str(QGNN_LITE_ROOT))

from shared.qnn_utils.seed import set_global_seed
from shared.qnn_utils.io import append_row_csv

from src.models.backbones.gcn import GCNBackbone
from src.models.heads.mlp import MLPHead
from src.models.heads.vqc import VQCHead
from src.train import train_model


def _split_dataset(ds, train_ratio: float, val_ratio: float):
    ds = ds.shuffle()  # seed уже зафиксирован set_global_seed(seed)
    n = len(ds)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_ds = ds[:n_train]
    val_ds = ds[n_train:n_train + n_val]
    test_ds = ds[n_train + n_val:n_train + n_val + n_test]
    return train_ds, val_ds, test_ds


class GraphModel(torch.nn.Module):
    def __init__(self, bb: torch.nn.Module, hd: torch.nn.Module):
        super().__init__()
        self.bb = bb
        self.hd = hd

    def forward(self, batch):
        g = self.bb(batch.x, batch.edge_index, batch.batch)
        return self.hd(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- seed ----
    seed = int(cfg["run"]["seed"])
    set_global_seed(seed)

    # ---- devices ----
    gnn_dev = cfg["devices"]["gnn"]
    gnn_device = torch.device("cuda" if (gnn_dev == "cuda" and torch.cuda.is_available()) else "cpu")
    vqc_device = torch.device("cpu")  # lightning.qubit -> CPU

    # ---- data ----
    dataset_name = cfg["data"]["dataset_name"]
    root_dir = cfg["data"]["root_dir"]
    ds = TUDataset(root=root_dir, name=dataset_name)

    train_ratio = float(cfg["run"]["split"]["train"])
    val_ratio = float(cfg["run"]["split"]["val"])
    train_ds, val_ds, test_ds = _split_dataset(ds, train_ratio, val_ratio)

    # ---- model ----
    hidden_dim = int(cfg["model"]["hidden_dim"])
    head_type = cfg["model"]["head"]

    backbone = GCNBackbone(in_dim=ds.num_features, hidden_dim=hidden_dim).to(gnn_device)

    # defaults for logging
    vqc_n_qubits = None
    vqc_n_layers = None
    vqc_shots = None

    if head_type == "mlp":
        head = MLPHead(hidden_dim=hidden_dim, num_classes=ds.num_classes).to(gnn_device)
        device = gnn_device

    elif head_type == "vqc":
        # read VQC params safely
        vqc_cfg = cfg.get("vqc", {})
        vqc_n_qubits = int(vqc_cfg.get("n_qubits", 4))
        vqc_n_layers = int(vqc_cfg.get("n_layers", 1))
        vqc_shots = vqc_cfg.get("shots", None)

        head = VQCHead(
            hidden_dim=hidden_dim,
            num_classes=ds.num_classes,
            n_qubits=vqc_n_qubits,
            n_layers=vqc_n_layers,
            shots=vqc_shots,
        ).to(vqc_device)

        # MVP: whole model on CPU to avoid device mismatch
        backbone = backbone.to(vqc_device)
        device = vqc_device

    else:
        raise ValueError(f"Unknown head: {head_type}. Use 'mlp' or 'vqc'.")

    model = GraphModel(backbone, head).to(device)

    # ---- train ----
    result = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        epochs=int(cfg["run"]["epochs"]),
        batch_size=int(cfg["run"]["batch_size"]),
        lr=float(cfg["run"]["lr"]),
        weight_decay=float(cfg["run"]["weight_decay"]),
        device=device,
        num_workers=int(cfg["run"]["num_workers"]),
    )

    # ---- save result row ----
    out_csv = REPO_ROOT / "projects" / "qgnn-lite" / "results" / "results.csv"
    row = {
        "ts": int(time.time()),
        "dataset": dataset_name,
        "backbone": cfg["model"]["backbone"],
        "head": head_type,
        "hidden_dim": hidden_dim,
        "epochs": int(cfg["run"]["epochs"]),
        "batch_size": int(cfg["run"]["batch_size"]),
        "lr": float(cfg["run"]["lr"]),
        "seed": seed,
        "device": str(device),
        "train_loss": result.train_loss,
        "train_acc": result.train_acc,
        "val_acc": result.val_acc,
        "test_acc": result.test_acc,
        "seconds": result.seconds,
        "vqc_n_qubits": vqc_n_qubits,
        "vqc_n_layers": vqc_n_layers,
        "vqc_shots": vqc_shots,
    }
    append_row_csv(out_csv, row)

    print("OK. Saved:", out_csv)
    print("Result:", row)


if __name__ == "__main__":
    main()

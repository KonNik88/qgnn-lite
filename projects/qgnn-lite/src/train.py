# projects/qgnn-lite/src/train.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


@dataclass
class TrainResult:
    train_loss: float
    train_acc: float
    val_acc: float
    test_acc: float
    seconds: float


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def _run_epoch(model, loader, device, optimizer=None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    n_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad()

        logits = model(batch)  # [B, C]
        loss = F.cross_entropy(logits, batch.y)

        if is_train:
            loss.backward()
            optimizer.step()

        bs = batch.num_graphs
        total_loss += float(loss.item()) * bs
        total_acc += _accuracy(logits, batch.y) * bs
        n_graphs += bs

    n_graphs = max(n_graphs, 1)
    return {
        "loss": total_loss / n_graphs,
        "acc": total_acc / n_graphs,
    }


@torch.no_grad()
def _eval(model, loader, device) -> float:
    model.eval()
    total_acc = 0.0
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        bs = batch.num_graphs
        total_acc += _accuracy(logits, batch.y) * bs
        n_graphs += bs
    return total_acc / max(n_graphs, 1)


def train_model(
    model,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    num_workers: int = 0,
) -> TrainResult:
    t0 = time.time()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None

    last_train = {"loss": 0.0, "acc": 0.0}

    for _ in range(epochs):
        last_train = _run_epoch(model, train_loader, device, optimizer=optimizer)
        val_acc = _eval(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    val_acc = _eval(model, val_loader, device)
    test_acc = _eval(model, test_loader, device)

    dt = time.time() - t0
    return TrainResult(
        train_loss=float(last_train["loss"]),
        train_acc=float(last_train["acc"]),
        val_acc=float(val_acc),
        test_acc=float(test_acc),
        seconds=float(dt),
    )

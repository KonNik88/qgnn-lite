# scripts/smoke_train_vqc.py
# End-to-end smoke test:
# - Load MUTAG (PyG)
# - Train 1 epoch with:
#   1) GCN + MLP head (GPU if available)
#   2) GCN + VQC head (CPU, because PennyLane lightning.qubit is CPU)
#
# Expected:
# - MLP runs fast (likely on CUDA)
# - VQC runs slower but completes without errors

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import pennylane as qml


class GCNBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_mean_pool(x, batch)  # [B, hidden]
        return g


class MLPHead(nn.Module):
    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, g):
        return self.net(g)


class VQCHead(nn.Module):
    """
    VQC head: graph embedding -> angles -> quantum expvals -> logits
    NOTE: lightning.qubit is CPU. For this smoke test we run the whole VQC model on CPU.
    """
    def __init__(self, hidden: int, n_classes: int, n_qubits: int = 4, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.pre = nn.Linear(hidden, n_qubits)         # embedding -> angles
        self.post = nn.Linear(n_qubits, n_classes)     # expvals -> logits

        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(angles, weights):
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        # trainable quantum weights
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))

    def forward(self, g):
        if g.is_cuda:
            raise RuntimeError(
                "VQCHead received CUDA tensor, but lightning.qubit runs on CPU. "
                "Run VQC model on CPU for this smoke test."
            )

        angles = torch.tanh(self.pre(g)) * 3.14159  # [B, n_qubits]

        # circuit(a, weights) returns a Python list of Tensors -> convert to Tensor
        expvals = torch.stack(
            [torch.stack(self.circuit(a, self.q_weights)) for a in angles],
            dim=0
        )  # [B, n_qubits]

        logits = self.post(expvals)
        return logits


class GraphModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch):
        g = self.backbone(batch.x, batch.edge_index, batch.batch)
        return self.head(g)


def train_one_epoch(model, loader, device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_loss = 0.0
    correct = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()

        out = model(batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        opt.step()

        total_loss += float(loss) * batch.num_graphs
        pred = out.argmax(dim=-1)
        correct += int((pred == batch.y).sum())
        n += batch.num_graphs

    return total_loss / max(n, 1), correct / max(n, 1)


def main():
    # For the smoke:
    # - MLP on CUDA if available
    # - VQC on CPU (PennyLane lightning.qubit)
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    print("cuda_device:", cuda_device)
    print("cpu_device:", cpu_device)

    ds = TUDataset(root="data/TUDataset", name="MUTAG")
    ds = ds.shuffle()
    train_ds = ds[:150]
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # ----- 1) MLP baseline (GPU if available)
    backbone_mlp = GCNBackbone(in_dim=ds.num_features, hidden=32).to(cuda_device)
    head_mlp = MLPHead(hidden=32, n_classes=ds.num_classes).to(cuda_device)
    model_mlp = GraphModel(backbone=backbone_mlp, head=head_mlp).to(cuda_device)

    t0 = time.time()
    loss, acc = train_one_epoch(model_mlp, loader, cuda_device)
    dt = time.time() - t0
    print(f"MLP: loss={loss:.4f} acc={acc:.3f} time={dt:.2f}s")

    # ----- 2) VQC head (CPU)
    backbone_vqc = GCNBackbone(in_dim=ds.num_features, hidden=32).to(cpu_device)
    head_vqc = VQCHead(hidden=32, n_classes=ds.num_classes, n_qubits=4, n_layers=1).to(cpu_device)
    model_vqc = GraphModel(backbone=backbone_vqc, head=head_vqc).to(cpu_device)

    t0 = time.time()
    loss, acc = train_one_epoch(model_vqc, loader, cpu_device)
    dt = time.time() - t0
    print(f"VQC: loss={loss:.4f} acc={acc:.3f} time={dt:.2f}s")


if __name__ == "__main__":
    main()

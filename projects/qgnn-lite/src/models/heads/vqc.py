# projects/qgnn-lite/src/models/heads/vqc.py
from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml


class VQCHead(nn.Module):
    """
    VQC head: graph embedding -> angles -> quantum expvals -> logits

    Важно:
    - lightning.qubit работает на CPU
    - поэтому VQC-ветку держим на CPU (и вход g должен быть CPU tensor)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        n_qubits: int = 4,
        n_layers: int = 1,
        shots: int | None = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

        self.pre = nn.Linear(hidden_dim, n_qubits)
        self.post = nn.Linear(n_qubits, num_classes)

        # PennyLane device (CPU)
        self.dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(angles, weights):
            qml.AngleEmbedding(angles, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        if g.is_cuda:
            raise RuntimeError(
                "VQCHead получил CUDA tensor. Для lightning.qubit держим VQC на CPU. "
                "Переведи граф-эмбеддинги на CPU перед VQC."
            )

        angles = torch.tanh(self.pre(g)) * 3.1415926535  # [B, n_qubits]

        # circuit(...) -> list[tensor], поэтому делаем torch.stack внутри
        expvals = torch.stack(
            [torch.stack(self.circuit(a, self.q_weights)) for a in angles],
            dim=0,
        )  # [B, n_qubits]

        logits = self.post(expvals)
        return logits

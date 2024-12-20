import torch
import pennylane as qml
from pennylane import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ring_circuit(wires, params):
    n_qubits = len(wires)
    weights = params.get("weights", torch.randn(3*n_qubits,  2, device=device))

    for wire in wires:
        qml.RY(weights[wire, 0], wires = wire)
        qml.RZ(weights[wire, 1], wires = wire)

    for i in range(n_qubits):
        if i != n_qubits-1:
            qml.CZ(wires = [i, i + 1])
        else:
            qml.CZ(wires = [i, 0])
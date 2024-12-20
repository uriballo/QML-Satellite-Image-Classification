import torch
import pennylane as qml
from pennylane import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NQ_circuit(wires, params):
    
    n_qubits = len(wires)
    weights = params.get('weights', torch.randn(3*n_qubits,  2, device=device) % np.pi)

    
    for wire in range(n_qubits-1):
        qml.RY(weights[wire, 0], wires = wire)
        qml.RZ(weights[wire, 1], wires = wire)

    qml.Barrier()
    
    for wire in range(n_qubits-2):
        qml.CNOT(wires = [wires[-wire-3], wires[-wire -2]])

    qml.Barrier()

    for wire in range(n_qubits-1):
        qml.RY(weights[wire + 4, 0], wires = wire)
        qml.RZ(weights[wire + 4, 1], wires = wire)

    qml.Barrier()
    
    for wire in range(n_qubits-2):
        qml.CNOT(wires = [wires[-wire-3], wires[-wire -2]])

    qml.Barrier()

    for wire in range(n_qubits-1):
        qml.RY(weights[wire + 8, 0], wires = wire)
        qml.RZ(weights[wire + 8, 1], wires = wire)
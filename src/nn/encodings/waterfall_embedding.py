import pennylane as qml
from pennylane import numpy as np

def waterfall_embedding(inputs, wires, params):
    n_qubits = len(wires)

    for idx, wire in enumerate(wires):
        qml.Hadamard(wires = wire)
        qml.RY(inputs[:, idx], wires = wire)
        # Aplicar las puertas CNOT en cascada
    for control in range(n_qubits):
        for target in range(control + 1, n_qubits):
            qml.CNOT(wires=[control, target])

    qml.Barrier()
    for idx, wire in enumerate(wires):
        qml.RZ(inputs[:, idx], wires = wire)
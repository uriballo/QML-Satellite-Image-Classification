import pennylane as qml
from pennylane import numpy as np

def ring_embedding(inputs, wires, params):

    n_repeats = params.get("n_repeats", 1)
    n_qubits = len(wires)
    
    # Repeat the pattern n_repeats times
    for _ in range(n_repeats):
        
        for idx, wire in enumerate(wires):

            # Apply Hadamard gates to all wires
            qml.Hadamard(wires = wire)

            # Apply RZ rotations with inputs as parameters
            qml.RY(inputs[:, idx], wires=wire)


        for i in range(n_qubits):
            if i < n_qubits - 1:   
                qml.CNOT(wires = [i, i + 1])
            else:
                qml.CNOT(wires = [i, 0])

        for idx, wire in enumerate(wires):

            # Apply RZ rotations with inputs as parameters
            qml.RY(inputs[:, idx], wires=wire)

        for i in range(n_qubits):
            if i < n_qubits - 1:   
                qml.CNOT(wires = [i + 1, i])
            else:
                qml.CNOT(wires = [0, i])
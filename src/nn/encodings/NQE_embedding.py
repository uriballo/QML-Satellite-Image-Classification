import pennylane as qml
from pennylane import numpy as np

def NQE_embedding(inputs, wires, params):

    n_qubits = len(wires)
    n_repeats = params.get('n_repeats', 1)

    for _ in range(n_repeats):

        for idx, wire in enumerate(wires):

            qml.Hadamard(wires = wire)
            qml.RZ(inputs[:, idx], wires = wire)

        cont = n_qubits - 1
        a = 0
        for _ in range(n_qubits - 1):
            for j in range(cont):
                wire1 = wires[a]
                wire2 = wires[j + 1 + a]
                phi = (np.pi - inputs[:, a]) * (np.pi - inputs[:, j + a + 1])

                qml.CNOT(wires = [wire1, wire2])
                qml.RZ(phi, wires = wire2)
                qml.CNOT(wires = [wire1, wire2])
            cont -= 1
            a += 1
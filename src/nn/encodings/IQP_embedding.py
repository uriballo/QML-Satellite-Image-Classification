import pennylane as qml

def custom_iqp_embedding(inputs, wires, params):
    n_qubits = len(wires)
    n_repeats = params.get("n_repeats", 1)
    pattern = params.get("pattern")


    # Repeat the pattern n_repeats times
    for _ in range(n_repeats):
        
        for idx, wire in enumerate(wires):

            # Apply Hadamard gates to all wires
            qml.Hadamard(wires = wire)

            # Apply RZ rotations with inputs as parameters
            qml.RZ(inputs[:, idx], wires=wire)

        # Apply entangling gates ZZ
        if pattern is None:
            # Default pattern: all combinations of qubits
            cont = n_qubits - 1
            a = 0
            for i in range(n_qubits - 1):
                for j in range(cont):
                    wire1 = wires[a]
                    wire2 = wires[j + 1 + a]
                    phi = inputs[:, a] * inputs[:, j + a + 1]
                    qml.IsingZZ(phi, wires = [wire1, wire2])
                cont -= 1
                a += 1
                
        else:
            a = 0
            # Custom pattern PROVISIONAL
            for pair in pattern:
                a += 1
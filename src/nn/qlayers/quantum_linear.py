import torch.nn as nn
import pennylane as qml

class QuantumLinear(nn.Module):
    def __init__(self, in_features, out_features, q_circuit, weight_shapes, num_qubits, embedding="n"):
        """
        :param in_features: number of input features.
        :param out_features: number of output features (currently not used).
        :param q_circuit: quantum circuit used to process the input features.
        :param weight_shapes: shape of the quantum circuit weights.
        :param num_qubits: number of qubits of the quantum circuit.
        :param embedding: either `n` or `2n`, depending on the embedding used. Use `n` for embeddings like angle and
                            `2n` for amplitude.
        """
        super(QuantumLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features  # Not used, currently depends on the embedding.
        self.num_qubits = num_qubits
        self.embedding = embedding
        
        # TODO: Support more embeddings
        if embedding == "n":
            assert in_features % num_qubits == 0, ("The number of input features should be divisible by the number of "
                                                   "qubits")
            self.n_circuits = in_features // num_qubits
            self.circuits = nn.ModuleList([
                qml.qnn.TorchLayer(q_circuit, weight_shapes) for _ in range(self.n_circuits)
            ])
        elif embedding == "2n":
            self.n_circuits = in_features // 2 ** num_qubits
            self.circuits = nn.ModuleList([
                qml.qnn.TorchLayer(q_circuit, weight_shapes) for _ in range(self.n_circuits)
            ])
        else:
            ValueError("Embedding not recognized use either `n` or `2n`.") 

        # TODO: We could parametrize this    
        for circ in self.circuits:
            nn.init.xavier_uniform_(circ.weights)

    def forward(self, x):
        x = x.view(x.shape[0], self.n_circuits, self.num_qubits)
        x = torch.cat([circ(x[:, i, :]) for i, circ in enumerate(self.circuits)], dim=1)
        return x
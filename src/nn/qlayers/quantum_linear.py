import torch.nn as nn
import pennylane as qml
import torch

from src.nn.ansatz.default import default_circuit
from src.nn.encodings.IQP_embedding import custom_iqp_embedding
from src.nn.encodings.pennylane_templates import angle_embedding, amplitude_embedding
from src.nn.encodings.waterfall_embedding import waterfall_embedding
from src.nn.measurements.default import default_measurement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuantumLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 num_qubits_per_circuit,
                 embedding=angle_embedding,
                 circuit=default_circuit,
                 measurement=default_measurement,
                 params=None):
        super(QuantumLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features  # Not used, depends on the embedding.
        self.num_qubits = num_qubits_per_circuit

        self.embedding = embedding["func"]
        self.embedding_params = embedding["func_params"]

        self.circuit = circuit["func"]
        self.circuit_params = circuit["func_params"]

        self.measurement = measurement["func"]
        self.measurement_params = measurement["func_params"]

        dev = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def unit_circuit(inputs, weights):
            wires = range(num_qubits_per_circuit)

            self.embedding(inputs, wires, self.embedding_params)
            self.circuit_params["weights"] = weights
            self.circuit(wires, self.circuit_params)

            return self.measurement(wires, self.measurement_params)


        # Determine the number of input features per circuit.
        if self.embedding_params is amplitude_embedding:
            assert in_features % (2**self.num_qubits) == 0, (
                "The number of input features should be divisible by 2^(number of qubits per circuit)."
            )
            self.input_per_circuit = 2 ** self.num_qubits
        else:
            assert in_features % self.num_qubits == 0, (
                "The number of input features should be divisible by the number of qubits per circuit."
            )
            self.input_per_circuit = self.num_qubits

        self.n_circuits = in_features // self.input_per_circuit
        self.circuits = nn.ModuleList([
            qml.qnn.TorchLayer(unit_circuit, self.circuit_params["weight_shapes"])
            for _ in range(self.n_circuits)
        ])

    def forward(self, x):
        # Reshape such that each circuit gets self.input_per_circuit features.
        x = x.view(x.shape[0], self.n_circuits, self.input_per_circuit)
        # Apply each circuit on its corresponding slice
        x = torch.cat([circ(x[:, i, :]) for i, circ in enumerate(self.circuits)], dim=1)
        return x

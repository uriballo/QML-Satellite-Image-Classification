import torch
import torch.nn as nn
import pennylane as qml
from src.nn.ansatz.default import default_circuit
from src.nn.encodings.pennylane_templates import amplitude_embedding
from src.nn.measurements.default import default_measurement

class QuanvLayer(nn.Module):
    def __init__(self, qkernel_shape, embedding=None, circuit=None, measurement=None, qdevice_kwargs=None):
        super(QuanvLayer, self).__init__()
        self.qkernel_shape = qkernel_shape
        self.embedding = embedding["func"] or amplitude_embedding
        self.embedding_params = embedding["func_params"] or None
        self.circuit = circuit["func"] or default_circuit
        self.circuit_params = circuit["func_params"] or None
        self.measurement = measurement["func"] or default_measurement
        self.measurement_params = measurement["func_params"] or None
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qml_device = None
        self.qnode = None

    def quantum_circuit(self, inputs):
        wires = range(self.qkernel_shape ** 2)

        # Embedding block
        self.embedding(inputs, wires, self.embedding_params)

        # Circuit block
        self.circuit(wires, self.circuit_params)

        # Measurement block
        return self.measurement(wires, self.measurement_params)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.to(self.device)
        patch_size = self.qkernel_shape ** 2

        if self.qnode is None:
            qml_device_name = self.qdevice_kwargs.pop('qml_device_name', 'default.qubit')
            self.qml_device = qml.device(
                qml_device_name, wires=patch_size, **self.qdevice_kwargs
            )
            self.qnode = qml.QNode(
                self.quantum_circuit,
                self.qml_device,
                interface='torch',
                diff_method='backprop'
            )

        # Extract patches
        patches = x.unfold(2, self.qkernel_shape, 1).unfold(3, self.qkernel_shape, 1)
        patches = patches.contiguous().view(-1, self.qkernel_shape ** 2)

        # Process patches
        outputs = self.qnode(patches)

        # Remove the torch.stack line
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.float()

        # Reshape outputs
        out_height = height - self.qkernel_shape + 1
        out_width = width - self.qkernel_shape + 1
        outputs = outputs.view(batch_size, -1, out_height, out_width)

        return outputs

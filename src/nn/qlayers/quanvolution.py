import torch
import torch.nn as nn
import pennylane as qml

import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuanvLayer(nn.Module):
    def __init__(self, qkernel_shape, embedding=None, circuit=None, measurement=None, params=None, qdevice_kwargs=None):
        super(QuanvLayer, self).__init__()
        self.qkernel_shape = qkernel_shape
        self.embedding = embedding
        self.circuit = circuit
        self.measurement = measurement
        self.params = params or {}
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.torch_device = device
        self.qml_device = None
        self.qnode = None

    def quantum_circuit(self, inputs):
        wires = range(self.qkernel_shape ** 2)
        params = self.params

        # Embedding block
        self.embedding(inputs, wires, params.get('embedding', {}))

        # Circuit block
        self.circuit(wires, params.get('circuit', {}))

        # Measurement block
        return self.measurement(wires, params.get('measurement', {}))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.to(self.torch_device)
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

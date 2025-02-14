import torch
import torch.nn as nn
import pennylane as qml

import sys
import os

class DefaultReshaper:
    def __init__(self):
        self.reshape = self._flatten
    
    def _flatten(self,x):
        return x.reshape(x.shape[0],-1)
        
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.set_default_device(device)

class QuantumCircuitModel(nn.Module):
    def __init__(self, n_wires,embedding, circuit=None, measurement=None, params=None, weight_shapes=None,reshaper=None, qdevice_kwargs=None):
        super(QuantumCircuitModel, self).__init__()
        self.n_wires = n_wires
        self.embedding = embedding
        self.circuit = circuit
        self.params = params or {}
        self.weight_shapes = weight_shapes or {}
        self.qdevice_kwargs = qdevice_kwargs or {}
        self.torch_device = device
        self.measurement = measurement
        self.reshaper = reshaper or DefaultReshaper()
        
        qml_device_name = self.qdevice_kwargs.pop('qml_device_name', 'default.qubit')
        self.qml_device = qml.device(
            qml_device_name, wires=self.n_wires, **self.qdevice_kwargs
        )
        self.qnode = qml.QNode(
            self.quantum_circuit,
            self.qml_device,
            interface='torch',
            diff_method='backprop'
        )
        self.qlayer = qml.qnn.TorchLayer(self.qnode,weight_shapes)
        
    def quantum_circuit(self, inputs,weights):
        wires = range(self.n_wires)
        params = self.params

        # Embedding block
        self.embedding(inputs, wires, params.get('embedding', {}))

        # Circuit block
        params_circuit = params.get('circuit', {})
        params_circuit['weights'] = weights
        self.circuit(wires,params_circuit)

        # Measurement block
        return self.measurement(wires, params.get('measurement', {}))

    def forward(self,x):
        x = x.to(self.torch_device)
        x = self.reshaper.reshape(x)

        return self.qlayer(x)
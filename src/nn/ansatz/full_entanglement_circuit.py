import torch
import pennylane as qml
from pennylane import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def full_entanglement_circuit(wires, params):
    num_layers = params.get('num_layers', 1)
    weights = params.get('weights', torch.randn(num_layers, len(wires), 3, device=device) % np.pi)
    qml.templates.StronglyEntanglingLayers(weights, wires=wires)
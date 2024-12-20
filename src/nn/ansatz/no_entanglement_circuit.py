import torch
import pennylane as qml
from pennylane import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def no_entanglement_random_circuit(wires, params):
    weights = params.get("weights", torch.rand(len(wires), device = device) % np.pi)

    for wire in range(len(wires)):
        rand_num = np.random.choice([0, 1])
        if rand_num == 0:
            qml.Identity(wires=wire)
        else:
            qml.RZ(weights[wire].item(), wires=wire)
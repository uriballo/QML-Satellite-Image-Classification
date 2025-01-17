import torch
import pennylane as qml

def amplitude_embedding(inputs, wires, params):
    """
    An embedding using the AmplitudeEmbedding template from PennyLane.

    This embedding is differentiable and suitable for qubit-based devices.
    """
    normalize = params.get('normalize', True)
    pad_with = params.get('pad_with', 0.0)
    qml.AmplitudeEmbedding(inputs, wires = wires, pad_with = pad_with, normalize = normalize)

def angle_embedding(inputs, wires, params):
    """
    An embedding using the AngleEmbedding template from PennyLane.

    This embedding is differentiable and suitable for qubit-based devices.
    """
    rotation = params.get('rotation', 'Z')
    qml.AngleEmbedding(inputs, wires = wires, rotation=rotation)

def QAOA_embedding(inputs, wires, params):
    """
    An embedding using the QAOAEmbedding template from PennyLane.

    This embedding is differentiable and suitable for qubit-based devices.
    """
    weights = params.get('weights')
    local_field = params.get('local_field', 'Y')
    n_layers = params.get('n_layers', 1)
    if weights is None:
        weights_shape = qml.templates.QAOAEmbedding.shape(n_layers=n_layers, n_wires=len(wires))
        weights = torch.randn(weights_shape, requires_grad=True)
    qml.templates.QAOAEmbedding(features=inputs, weights=weights, wires=wires, local_field=local_field)
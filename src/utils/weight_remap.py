import torch

# From: https://arxiv.org/abs/2212.14807

def sigmoid_map(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid function. Scaled appropriatly to interval [-\\pi;\\pi].
    """
    return 2 * torch.pi * (1 / (1 + torch.pow(torch.e, -x))) - torch.pi


def arctan_map(x: torch.Tensor) -> torch.Tensor:
    """
    Arcus Tangent function. Scaled appropriatly to interval [-\\pi;\\pi].
    """
    return 2.0 * torch.arctan(2 * x)


def tanh_map(x: torch.Tensor) -> torch.Tensor:
    """
    Tangent Hyperbolicus. Scaled appropriatly to interval [-\\pi;\\pi]
    """
    return torch.pi * torch.tanh(x)
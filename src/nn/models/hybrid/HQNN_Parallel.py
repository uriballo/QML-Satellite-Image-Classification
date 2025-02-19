import torch.nn as nn
import torch.nn.functional as F
import torch
from src.nn.qlayers.quantum_linear import QuantumLinear

class HQNN_Parallel(nn.Module):
    def __init__(self,
                 embedding_params,
                 variational_params,
                 measurement_params,
                 n_classes=10,
                 input_size=32,
                 use_quantum=True,):
        super(HQNN_Parallel, self).__init__()

        base_channels = 6
        self.use_quantum = use_quantum

        # Feature extractor with depth based on input size
        if input_size == 32:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),  # (32x32) -> (32x32)
                nn.LeakyReLU(),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # (32x32) -> (16x16)
                nn.LeakyReLU(),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # (16x16) -> (8x8)
                nn.LeakyReLU(),
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),  # (8x8) -> (4x4)
                nn.LeakyReLU()
            )
            feature_dims = 4 * 4 * base_channels * 8

        elif input_size == 16:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),  # (16x16) -> (16x16)
                nn.LeakyReLU(),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # (16x16) -> (8x8)
                nn.LeakyReLU(),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # (8x8) -> (4x4)
                nn.LeakyReLU()
            )
            feature_dims = 4 * 4 * base_channels * 4

        else:
            raise ValueError("Unsupported input size. Use 32 or 16.")


        num_qubits = variational_params["func_params"]["weight_shapes"]["weights"][1]
        self.qfc = QuantumLinear(in_features=feature_dims,
                                 out_features=feature_dims,
                                 num_qubits_per_circuit=num_qubits,
                                 embedding=embedding_params,
                                 circuit=variational_params,
                                 measurement=measurement_params)

        self.fc1 = nn.Linear(feature_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)

        # Quantum thingy
        x = x.view(x.shape[0], -1)  # Shape: (batch_size, feature_dims)

        if self.use_quantum:
            x = x / x.abs().max(dim=1, keepdim=True)[0]  # Scale to [-1, 1]
            x = 2 * torch.pi * x  # Keep Pi-scaling
            x = self.qfc(x)

        # Classical fully connected layers
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x

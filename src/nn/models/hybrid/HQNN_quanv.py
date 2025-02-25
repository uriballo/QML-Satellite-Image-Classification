import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any, Callable
# Make sure to adjust import to your QuanvLayer file
from src.nn.qlayers.quanvolution import QuanvLayer


class FlexHybridCNN(nn.Module):
    """
    A flexible hybrid CNN model that can optionally use a QuanvLayer or 
    a classical convolution layer as the first convolutional operation.
    Args:
        n_classes (int): Number of output classes.
        use_quantum (bool): If True, apply QuanvLayer as the first layer; otherwise classical.
        qkernel_shape (int): Dimension for the quantum patch size.
        n_filters_1 (int): Number of filters in the first convolution layer (or output channels for Quanv).
        fc_hidden_dim (int): Dimension of the hidden FC layer.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        quanv_params (dict): Parameters for QuanvLayer (embedding, circuit, measurement, trainable, random_params, etc.).
    """
    
    def __init__(self,
        embedding_params: dict,
        variational_params: dict,
        measurement_params: dict,
        n_classes: int = 10,
        use_quantum: bool = True,
        qkernel_shape: int = 2,
        n_filters_1: int = 32,
        fc_hidden_dim: int = 128,
        epochs: int = 10,
        dataset: str = 'EuroSAT',
        image_size: int = 32,
    ):
        super().__init__()
        self.embedding = embedding_params
        self.variational = variational_params
        self.measurement = measurement_params
        self.n_classes = n_classes
        self.use_quantum = use_quantum
        self.qkernel_shape = qkernel_shape
        self.n_filters_1 = n_filters_1
        self.fc_hidden_dim = fc_hidden_dim
        self.epochs = epochs
        self.dataset = dataset
        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if (self.dataset == "DeepSat4" or self.dataset == "DeepSat6"):
            if use_quantum:
                self.in_channels_1, self.in_channels_2, self.kernel_size_1, self.kernel_size_2 = 3, 16, 7, 27
                if self.image_size == 16:
                    self.kernel_size_2 = 21
                if self.qkernel_shape == 3:
                    self.in_channels_2, self.kernel_size_2 = 36, 26
            else:
                self.in_channels_1, self.in_channels_2, self.kernel_size_1, self.kernel_size_2 = 4, 32, 32, 27
                if self.image_size == 16:
                    self.kernel_size_1 = 16
                    self.kernel_size_2 = 17
        elif self.dataset == "EuroSAT":
            self.in_channels_1, self.in_channels_2, self.kernel_size_1, self.kernel_size_2 = 3, 3 * self.qkernel_shape**2, 7, 7


        # Initialize the first layer
        if self.use_quantum:
            # Quanvolutional layer with optional params
            self.quanv = QuanvLayer(
                qkernel_shape=self.qkernel_shape,
                embedding=self.embedding,
                circuit=self.variational,
                measurement=self.measurement,
            ).to(self.device)

            # The "output channels" from QuanvLayer is typically the measurement dimension.
            # For a single measurement per patch, we treat that as n_filters_1.
            # If your measurement returns multiple values, adjust here.
                
        else:
            # Classical convolution for the first layer
            self.conv1_classical = nn.Conv2d(
                in_channels = self.in_channels_1,
                out_channels = self.n_filters_1,
                kernel_size = self.kernel_size_1
            ).to(self.device)
    
        # Second convolution
        if not use_quantum:
            self.in_channels_2 = 32
            
        self.conv2 = nn.Conv2d(
            in_channels = self.in_channels_2,
            out_channels = self.n_filters_1,
            kernel_size = self.kernel_size_2,
            padding = (self.kernel_size_1 // 2), # example: same-ish padding
        ).to(self.device)

        # We define the linear layers later once we know the shape after conv2
        self.fc1 = None
        self.fc2 = nn.Linear(self.fc_hidden_dim, self.n_classes).to(self.device)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If use_quantum is True, pass data through QuanvLayer, 
        otherwise through a classical conv1. Then follow with conv2, flatten, FC layers.
        """
        x = x.to(self.device)
        if self.use_quantum:
            x = self.quanv(x)
        else:
            x = self.conv1_classical(x)
            x = torch.relu(x)
            
        x = self.conv2(x)
        x = torch.relu(x) 

        # Flatten
        x = x.view(x.size(0), -1)

        # Lazy initialization for fc1
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), self.fc_hidden_dim).to(self.device)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
 
    

    def __name__(self) -> str:
        return self.__class__.__name__

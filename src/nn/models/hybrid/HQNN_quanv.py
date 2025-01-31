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
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        n_classes (int): Number of output classes.
        use_quantum (bool): If True, apply QuanvLayer as the first layer; otherwise classical.
        qkernel_shape (int): Dimension for the quantum patch size.
        n_filters_1 (int): Number of filters in the first convolution layer (or output channels for Quanv).
        n_filters_2 (int): Number of filters in the second convolution layer.
        kernel_size_1 (int): Kernel size for the second layer if using quantum first layer or the first classical layer.
        kernel_size_2 (int): Kernel size for the second classical layer.
        fc_hidden_dim (int): Dimension of the hidden FC layer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        quanv_params (dict): Parameters for QuanvLayer (embedding, circuit, measurement, trainable, random_params, etc.).
    """
    
    def __init__(self,
        in_channels: int = 3, # 1
        n_classes: int = 10,
        use_quantum: bool = True,
        qkernel_shape: int = 2,
        n_filters_1: int = 32,
        n_filters_2: int = 32,
        kernel_size_1: int = 8, # 3
        kernel_size_2: int = 3,
        fc_hidden_dim: int = 128,
        epochs: int = 10,
        batch_size: int = 32,
        quanv_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.use_quantum = use_quantum
        self.qkernel_shape = qkernel_shape
        self.n_filters_1 = n_filters_1
        self.n_filters_2 = n_filters_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.fc_hidden_dim = fc_hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self._best_val_loss = float('inf')

        self.device = torch.device("cuda" if torch.cuda.is_available() else cpu)

        self.labels = ['AnnualCrop', 'Forest',
                  'HerbaceousVegetation',
                  'Highway', 'Industrial',
                  'Pasture', 'PermanentCrop',
                  'Residential', 'River',
                  'SeaLake']

        
        # Initialize the first layer
        if self.use_quantum:
            # Quanvolutional layer with optional params
            self.quanv = QuanvLayer(
                qkernel_shape=self.qkernel_shape,
                **(quanv_params or {})
            ).to(self.device)

            # The "output channels" from QuanvLayer is typically the measurement dimension.
            # For a single measurement per patch, we treat that as n_filters_1.
            # If your measurement returns multiple values, adjust here.
            
            first_in_channels = self.n_filters_1
                
        else:
            in_channels = 3
            # Classical convolution for the first layer
            self.conv1_classical = nn.Conv2d(
                in_channels = in_channels,
                out_channels = self.n_filters_1,
                kernel_size = self.kernel_size_1
            ).to(self.device)
            first_in_channels = self.n_filters_1
    
        # Second convolution
        if use_quantum:
            in_channels = 3*self.qkernel_shape**2
        else:
            in_channels = 32
        self.conv2 = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = first_in_channels, 
            kernel_size = self.kernel_size_1,
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
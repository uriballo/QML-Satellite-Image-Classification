import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Union, List, Dict, Any, Callable

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Make sure to adjust import to your QuanvLayer file
from quanvolution import QuanvLayer


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
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        early_stopping (bool): If True, use early stopping based on validation loss.
        patience (int): Number of epochs to wait for improvement before stopping.
        log_wandb (bool): If True, log metrics to Weights & Biases.
        wandb_project (str): W&B project name (used if log_wandb=True).
        wandb_run_name (str): W&B run name (used if log_wandb=True).
        optimizer_class (torch.optim.Optimizer): Optimizer class, e.g. optim.Adam.
        optimizer_kwargs (dict): Extra arguments to pass to the optimizer.
        quanv_params (dict): Parameters for QuanvLayer (embedding, circuit, measurement, trainable, random_params, etc.).
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 10,
        use_quantum: bool = True,
        qkernel_shape: int = 2,
        n_filters_1: int = 32,
        n_filters_2: int = 32,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
        fc_hidden_dim: int = 128,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 32,
        early_stopping: bool = False,
        patience: int = 3,
        log_wandb: bool = False,
        wandb_project: str = "my_wandb_project",
        wandb_run_name: str = "my_experiment",
        optimizer_class: Callable = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
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
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.log_wandb = log_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self._best_val_loss = float('inf')
        self._early_stop_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            # Classical convolution for the first layer
            self.conv1_classical = nn.Conv2d(
                in_channels=self.in_channels, 
                out_channels=self.n_filters_1, 
                kernel_size=self.kernel_size_1
            ).to(self.device)
            first_in_channels = self.n_filters_1

        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=first_in_channels,
            out_channels=self.n_filters_2,
            kernel_size=self.kernel_size_2,
            padding=(self.kernel_size_2 // 2),  # example: same-ish padding
        ).to(self.device)

        # We define the linear layers later once we know the shape after conv2
        self.fc1 = None  
        self.fc2 = nn.Linear(self.fc_hidden_dim, self.n_classes).to(self.device)

        # Define criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_class(self.parameters(), lr=self.lr, **optimizer_kwargs)

        # Initialize W&B if requested
        if self.log_wandb and WANDB_AVAILABLE:
            wandb.init(project=self.wandb_project, name=self.wandb_run_name)
            wandb.config.update({
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "qkernel_shape": self.qkernel_shape,
                "use_quantum": self.use_quantum,
            })

        # Track metrics
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []

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

    def fit(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None
    ) -> None:
        """
        Train the model on the given train DataLoader, optionally use val_loader 
        for validation and early stopping.
        """
        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for (inputs, labels) in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation phase (optional)
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                # Early stopping check
                if self.early_stopping:
                    if val_loss < self._best_val_loss:
                        self._best_val_loss = val_loss
                        self._early_stop_counter = 0
                    else:
                        self._early_stop_counter += 1
                        if self._early_stop_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break

            # Logging
            if self.log_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })

            print(f"Epoch [{epoch+1}/{self.epochs}]: "
                  f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    def _evaluate(self, loader: DataLoader) -> (float, float):
        """
        Evaluate the model on a given DataLoader. Returns average loss and accuracy.
        """
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for (inputs, labels) in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = running_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc

    def predict(self, X: Union[torch.Tensor, np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """
        Predict on a given dataset X. Batches the data and runs forward pass.
        """
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds_list = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                outputs = self(batch)
                _, predicted = torch.max(outputs, 1)
                preds_list.append(predicted.cpu())

        return torch.cat(preds_list)

    def __name__(self) -> str:
        return self.__class__.__name__

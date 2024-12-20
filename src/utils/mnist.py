# Copyright 2024 CTIC (Technological Center for Information and Communication).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import random

def load_mnist_data(batch_size=4, img_size=8, limit=None, output="dl"):
    """
    Load the MNIST dataset and return the data in DataLoader or NumPy format.
    
    Args:
        batch_size (int): Batch size.
        img_size (int): Image size (img_size x img_size).
        limit (int or None): Number of samples to load. If None, loads the entire dataset.
        output (str): Output format, "dl" for DataLoader or "np" for NumPy.
    
    Returns:
        train_loader, test_loader (DataLoader): If output="dl".
        X_train, y_train, X_val, y_val (NumPy): If output="np".
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Limit dataset size if the limit parameter is provided
    if limit is not None:
        train_dataset = Subset(train_dataset, range(limit))
        test_dataset = Subset(test_dataset, range(limit))

    if output == "dl":
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    elif output == "np":
        # Collect data into NumPy arrays
        X_train = []
        y_train = []
        for data, target in DataLoader(train_dataset, batch_size=len(train_dataset)):
            X_train.append(data.numpy())
            y_train.append(target.numpy())

        X_test = []
        y_test = []
        for data, target in DataLoader(test_dataset, batch_size=len(test_dataset)):
            X_test.append(data.numpy())
            y_test.append(target.numpy())

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)

        return X_train, y_train, X_test, y_test
    else:
        raise ValueError(f"Unsupported format: {output}. Use 'dl' or 'np'.")


def load_eurosat_data(batch_size=4, img_size=8, limit=None, output="dl"):
    """
    Load the EuroSAT dataset and return the data in DataLoader or NumPy format.
    
    Args:
        batch_size (int): Batch size.
        img_size (int): Image size (img_size x img_size).
        limit (int or None): Number of samples to load. If None, loads the entire dataset.
        output (str): Output format, "dl" for DataLoader or "np" for NumPy.
    
    Returns:
        train_loader, test_loader (DataLoader): If output="dl".
        X_train, y_train, X_val, y_val (NumPy): If output="np".
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Redimension
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize (RGB)
    ])

    # Load full dataset
    dataset = datasets.EuroSAT(root='./data', download=True, transform=transform)

    # Shuffle indices
    indices = list(range(len(dataset)))  # Crear una lista de índices
    random.shuffle(indices)  # Mezclar los índices
    
    if limit is not None:
        indices = indices[:limit] 
        dataset = Subset(dataset, indices)
    
    # Train and val (80%-20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if output == "dl":
        return train_loader, val_loader
    elif output == "np":
        X_train, y_train = [], []
        for inputs, labels in train_loader:
            X_train.append(inputs.numpy())
            y_train.append(labels.numpy())

        X_val, y_val = [], []
        for inputs, labels in val_loader:
            X_val.append(inputs.numpy())
            y_val.append(labels.numpy())

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)

        return X_train, y_train, X_val, y_val
    else:
        raise ValueError(f"Unsupported format: {output}. Use 'dl' or 'np'.")
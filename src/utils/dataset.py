import os
import requests
import zipfile
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import random

class EuroSAT:
    def __init__(self, root="dataset/EuroSAT_RGB", num_classes=None, image_size=64, batch_size=256,
                 test_size=0.2, random_state=42, examples_per_class=1000,
                 allowed_classes=None, output='np'):
        """
        Initializes the dataset loader for EuroSAT with specified parameters.
        Downloads and extracts dataset if not found in root directory.
        """
        self.root = root
        if not os.path.exists(root):
            self._download_and_extract_dataset()
            
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.examples_per_class = examples_per_class
        self.output = output
        self.class_dict = {
            'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2, 'Highway': 3, 'Industrial': 4,
            'Pasture': 5, 'PermanentCrop': 6, 'Residential': 7, 'River': 8, 'SeaLake': 9
        }
        
        self.allowed_classes = [self.class_dict[c] for c in allowed_classes] if allowed_classes else None
        self.labels = list(self.class_dict.keys())
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _download_and_extract_dataset(self):
        """Downloads and extracts the EuroSAT dataset."""
        os.makedirs("dataset", exist_ok=True)
        zip_path = "dataset/EuroSAT_RGB.zip"
        
        # Download the dataset
        url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
        print("Downloading EuroSAT dataset...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset")
        
        # Remove the zip file
        os.remove(zip_path)
        print("Dataset ready!")

    def get_loaders(self):
        """Returns DataLoader objects or NumPy arrays based on output parameter."""
        dataset = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform)
        
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.allowed_classes)} if self.allowed_classes else None
        
        if self.allowed_classes:
            dataset = self.filter_classes(dataset)
            
        if self.examples_per_class > 0:
            dataset = self.limit_examples_per_class(dataset)

        train_set, val_set = self.split_dataset(dataset)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.print_class_counts(train_set, "train")
        self.print_class_counts(val_set, "validation")
        
        if self.output == "dl":
            return train_loader, val_loader
        elif self.output == "np":
            X_train, y_train = self._loader_to_numpy(train_loader)
            X_test, y_test = self._loader_to_numpy(val_loader)
            return X_train, y_train, X_test, y_test, index_mapping
        else:
            raise ValueError(f"Unsupported format: {self.output}. Use 'dl' for DataLoader or 'np' for NumPy.")

    def _loader_to_numpy(self, loader):
        """Converts a DataLoader to NumPy arrays."""
        X, y = [], []
        for inputs, labels in loader:
            X.append(inputs.numpy())
            y.append(labels.numpy())
        return np.concatenate(X), np.concatenate(y)

    def filter_classes(self, dataset):
        """Filters the dataset to only include allowed classes."""
        filtered_indices = [i for i, (_, target) in enumerate(dataset.samples) if target in self.allowed_classes]
        return Subset(dataset, filtered_indices)

    def limit_examples_per_class(self, dataset):
        """Limits the number of examples per class."""
        targets = [dataset.dataset.targets[idx] for idx in dataset.indices]
        unique_classes = np.unique(targets)
        class_indices = {cls: [] for cls in unique_classes}

        for idx in dataset.indices:
            target = dataset.dataset.targets[idx]
            class_indices[target].append(idx)

        limited_indices = []
        for cls, indices in class_indices.items():
            if len(indices) > self.examples_per_class:
                limited_indices.extend(random.sample(indices, self.examples_per_class))
            else:
                limited_indices.extend(indices)

        return Subset(dataset.dataset, limited_indices)

    def split_dataset(self, dataset):
        """Splits the dataset into training and validation sets."""
        targets = [dataset.dataset.targets[idx] for idx in dataset.indices]
        train_indices, val_indices = train_test_split(
            dataset.indices, test_size=self.test_size, random_state=self.random_state, stratify=targets
        )
        return Subset(dataset.dataset, train_indices), Subset(dataset.dataset, val_indices)

    def print_class_counts(self, dataset, set_type):
        """Prints the class distribution in the given dataset."""
        class_counts = collections.Counter(dataset.dataset.targets[idx] for idx in dataset.indices)
        suma = sum(class_counts.values())
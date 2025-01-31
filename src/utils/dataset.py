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
        os.makedirs("dataset", exist_ok=True)
        zip_path = "dataset/EuroSAT_RGB.zip"
        
        url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
        print("Downloading EuroSAT dataset...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset")
        
        os.remove(zip_path)
        print("Dataset ready!")

    def get_loaders(self):
        dataset = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform)
        
        if self.allowed_classes:
            indices = [i for i, (_, target) in enumerate(dataset.samples) if target in self.allowed_classes]
            dataset = Subset(dataset, indices)

        if self.examples_per_class > 0:
            indices = self._limit_examples_per_class(dataset)
            dataset = Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, indices)

        train_indices, val_indices = self._split_dataset(dataset)
        
        train_set = Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, train_indices)
        val_set = Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self._print_class_counts(train_set, "train")
        self._print_class_counts(val_set, "validation")
        
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.allowed_classes)} if self.allowed_classes else None
        
        if self.output == "dl":
            return train_loader, val_loader
        elif self.output == "np":
            X_train, y_train = self._loader_to_numpy(train_loader)
            X_test, y_test = self._loader_to_numpy(val_loader)
            return X_train, y_train, X_test, y_test, index_mapping
        else:
            raise ValueError(f"Unsupported format: {self.output}. Use 'dl' for DataLoader or 'np' for NumPy.")

    def _loader_to_numpy(self, loader):
        X, y = [], []
        for inputs, labels in loader:
            X.append(inputs.numpy())
            y.append(labels.numpy())
        return np.concatenate(X), np.concatenate(y)

    def _limit_examples_per_class(self, dataset):
        if isinstance(dataset, Subset):
            samples = [dataset.dataset.samples[i] for i in dataset.indices]
        else:
            samples = dataset.samples
            
        class_indices = collections.defaultdict(list)
        for idx, (_, target) in enumerate(samples):
            class_indices[target].append(idx)
            
        limited_indices = []
        for target_class, indices in class_indices.items():
            if len(indices) > self.examples_per_class:
                limited_indices.extend(random.sample(indices, self.examples_per_class))
            else:
                limited_indices.extend(indices)
                
        return limited_indices

    def _split_dataset(self, dataset):
        if isinstance(dataset, Subset):
            targets = [dataset.dataset.targets[i] for i in dataset.indices]
            indices = dataset.indices
        else:
            targets = dataset.targets
            indices = range(len(targets))
            
        return train_test_split(
            indices, test_size=self.test_size,
            random_state=self.random_state,
            stratify=targets
        )

    def _print_class_counts(self, dataset, set_type):
        if isinstance(dataset, Subset):
            targets = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            targets = dataset.targets
        class_counts = collections.Counter(targets)
        print(f"\nClass distribution in {set_type} set:")
        for label in sorted(class_counts.keys()):
            print(f"Class {label}: {class_counts[label]}")
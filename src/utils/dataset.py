import os
import requests
import zipfile
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import random
import pandas as pd

class EuroSAT:
    def __init__(self, root="dataset/EuroSAT_RGB", image_size=64, batch_size=256,
                 test_size=0.2, random_state=42, examples_per_class=1000,
                 allowed_classes=None, output='np'):
        self.root = root
        if not os.path.exists(root):
            self._download_and_extract_dataset()

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


class DeepSatCSV(Dataset):
    def __init__(self, x_file, y_file, transform=None, max_samples=None):
        # Cargar datos desde CSV
        self.X = pd.read_csv(x_file, header=None).values  # Cargar imágenes
        self.y = pd.read_csv(y_file, header=None).values  # Cargar etiquetas
        self.transform = transform
        
        # Normalizar imágenes (de [0, 255] a [0, 1])
        self.X = self.X.astype(np.float32) / 255.0
        
        # Convertir etiquetas de one-hot a índices de clase
        self.y = np.argmax(self.y, axis=1)  # Si ya está en índices, omitir esto

        # Redimensionar imágenes (DeepSat usa 4 canales y 28x28 píxeles por imagen)
        num_samples = self.X.shape[0]
        self.X = self.X.reshape((num_samples, 4, 28, 28))  # [N, C, H, W]

        # Limitar el número de muestras si se especifica
        if max_samples is not None:
            self.X = self.X[:max_samples]
            self.y = self.y[:max_samples]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = torch.tensor(self.X[idx])  # Convertir a tensor
        label = torch.tensor(self.y[idx], dtype=torch.long)  # Convertir a tensor
        
        if self.transform:
            image = self.transform(image)

        return image, label
    

def load_dataset(dataset, output, limit, allowed_classes, image_size, test_size, batch_size=4):
    if dataset == "EuroSAT":
        data = EuroSAT(root= 'dataset/EuroSAT_RGB',
                                image_size=image_size,
                                examples_per_class=limit,
                                batch_size=batch_size,
                                test_size=test_size,
                                allowed_classes=allowed_classes,
                                output = output
                        )
        
        if output == 'dl':
            return data.get_loaders()
        elif output == 'np':
            X_train, y_train, X_val, y_val, index_mapping = data.get_loaders()
        
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)
        
            train_dataset = TensorDataset(X_train, y_train) 
            val_dataset = TensorDataset(X_val, y_val)

            return DataLoader(train_dataset, batch_size = 32, shuffle = True), DataLoader(val_dataset, batch_size = 32, shuffle = False)
        else:
            raise ValueError("Invalid output. Accepted values are 'dl' or 'np'")
    else:
        if dataset == "DeepSat4":
            # Root
            data_path = "dataset/DeepSat4/"
            x_train_file = data_path + "X_train_sat4.csv"
            y_train_file = data_path + "y_train_sat4.csv"
            x_test_file = data_path + "X_test_sat4.csv"
            y_test_file = data_path + "y_test_sat4.csv"
            
        elif dataset == "DeepSat6":
            # Root
            data_path = "dataset/DeepSat6/"
            x_train_file = data_path + "X_train_sat6.csv"
            y_train_file = data_path + "y_train_sat6.csv"
            x_test_file = data_path + "X_test_sat6.csv"
            y_test_file = data_path + "y_test_sat6.csv"

        else:
            raise ValueError("Invalid dataset. Accepted values are 'EuroSAT', 'DeepSat4' or 'DeepSat6'")
            
        # Limit
        train_size = 1 - test_size
        max_train_samples = int(train_size * limit) 
        max_test_samples = int(test_size * limit)   
        
        # Create DataLoaders
        train_dataset = DeepSatCSV(x_train_file, y_train_file, max_samples=max_train_samples)
        test_dataset = DeepSatCSV(x_test_file, y_test_file, max_samples=max_test_samples)

        return DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(test_dataset, batch_size=32, shuffle=False)
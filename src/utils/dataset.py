import os
import requests
import zipfile
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import random
import pandas as pd
from loguru import logger

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
        logger.info("Downloading EuroSAT dataset...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset")
        
        os.remove(zip_path)
        logger.info("Dataset ready!")

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
        logger.trace(f"\nClass distribution in {set_type} set:")
        for label in sorted(class_counts.keys()):
            logger.trace(f"Class {label}: {class_counts[label]}")


class DeepSatCSV(Dataset):
    def __init__(self, x_file, y_file, transform=None, limit=None, target_size=None):
        self.transform = transform
        self.image_size = 28
        self.target_size = target_size if target_size else 28

        # Cargar desde archivos .npy
        self.X = np.load(x_file).astype(np.float32) / 255.0
        self.y = np.load(y_file)

        if self.y.shape[1] > 1:  
            self.y = np.argmax(self.y, axis=1)

        self.X = self.X.reshape((-1, 4, 28, 28))

        if limit:
            self.X, self.y = self.balance_classes(limit)

    def balance_classes(self, limit):
        unique_classes, class_counts = np.unique(self.y, return_counts=True)
        num_classes = len(unique_classes)

        selected_X, selected_y = [], []
        for cls in unique_classes:
            indices = np.where(self.y == cls)[0]
            np.random.shuffle(indices)

            if len(indices) < limit:
                extra_indices = np.random.choice(indices, limit - len(indices), replace=True)
                indices = np.concatenate([indices, extra_indices])

            selected_X.append(self.X[indices[:limit]])
            selected_y.append(self.y[indices[:limit]])

        X_selected = np.vstack(selected_X)
        y_selected = np.hstack(selected_y)

        perm = np.random.permutation(len(y_selected))
        return X_selected[perm], y_selected[perm]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = torch.tensor(self.X[idx])
        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.target_size != self.image_size:
            image = self.resize_image(image, self.target_size)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def resize_image(self, image, target_size):
        _, h, w = image.shape
        if target_size > h:
            pad = (target_size - h) // 2
            image = F.pad(image, (pad, pad, pad, pad), mode='constant', value=0)
        elif target_size < h:
            image = F.interpolate(image.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        return image

def load_dataset(dataset, output, limit, allowed_classes, image_size, test_size, batch_size = 4):
    if dataset == "EuroSAT":
        data = EuroSAT(root= '../dataset/EuroSAT_RGB',
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
            path = "../dataset/DeepSat4/"
            if not os.path.exists(path + "X_train_sat4.npy"):
                print("Transforming .csv into .npy (just the first time)...")
                x_train_file = path + "X_train_sat4.csv"
                y_train_file = path + "y_train_sat4.csv"
                x_test_file = path + "X_test_sat4.csv"
                y_test_file = path + "y_test_sat4.csv"
                
                X1 = pd.read_csv(x_train_file, header=None, dtype=np.float32).values
                y1 = pd.read_csv(y_train_file, header=None, dtype=np.int32).values
                X2 = pd.read_csv(x_test_file, header=None, dtype=np.float32).values
                y2 = pd.read_csv(y_test_file, header=None, dtype=np.int32).values

                np.save(path + "X_train_sat4.npy", X1)
                np.save(path + "y_train_sat4.npy", y1)
                np.save(path + "X_test_sat4.npy", X2)
                np.save(path + "y_test_sat4.npy", y2)

            x_train_file = path + "X_train_sat4.npy"
            y_train_file = path + "y_train_sat4.npy"
            x_test_file = path + "X_test_sat4.npy"
            y_test_file = path + "y_test_sat4.npy"
            
        elif dataset == "DeepSat6":
            path = "../dataset/DeepSat6/"
            if not os.path.exists(path + "X_train_sat6.npy"):
                print("Transforming .csv into .npy (just the first time)...")
                x_train_file = path + "X_train_sat6.csv"
                y_train_file = path + "y_train_sat6.csv"
                x_test_file = path + "X_test_sat6.csv"
                y_test_file = path + "y_test_sat6.csv"
                
                X1 = pd.read_csv(x_train_file, header=None, dtype=np.float32).values
                y1 = pd.read_csv(y_train_file, header=None, dtype=np.int32).values
                X2 = pd.read_csv(x_test_file, header=None, dtype=np.float32).values
                y2 = pd.read_csv(y_test_file, header=None, dtype=np.int32).values

                np.save(path + "X_train_sat6.npy", X1)
                np.save(path + "y_train_sat6.npy", y1)
                np.save(path + "X_test_sat6.npy", X2)
                np.save(path + "y_test_sat6.npy", y2)

            x_train_file = path + "X_train_sat6.npy"
            y_train_file = path + "y_train_sat6.npy"
            x_test_file = path + "X_test_sat6.npy"
            y_test_file = path + "y_test_sat6.npy"

        else:
            raise ValueError("Invalid dataset. Accepted values are 'EuroSAT', 'DeepSat4' or 'DeepSat6'")
            
        train_samples = int(0.8 * limit)
        test_samples = int(0.2 * limit)

        train_dataset = DeepSatCSV(x_train_file, y_train_file, limit=train_samples, target_size=image_size)
        test_dataset = DeepSatCSV(x_test_file, y_test_file, limit=test_samples, target_size=image_size)

        return DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(test_dataset, batch_size=32, shuffle=False)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import random

class EuroSAT:
    def __init__(self, root, num_classes=None, image_size=64, batch_size=256,
                 test_size=0.2, random_state=42, examples_per_class=1000,
                 allowed_classes=None, output = 'np'):
        """
        Initializes the dataset loader for EuroSAT with specified parameters.
        
        Args:
            root (str): Root directory of the EuroSAT dataset.
            num_classes (int): Number of classes to include. If None, include all.
            image_size (int): Size to which images are resized.
            batch_size (int): Batch size for DataLoader.
            test_size (float): Proportion of data used for validation/testing.
            random_state (int): Random seed for reproducibility.
            examples_per_class (int): Max number of examples per class. -1 for no limit.
            allowed_classes (list): List of class names allowed. If None, all classes are included.
        """
        self.root = root
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
        
        # If allowed classes is specified, convert to index values
        self.allowed_classes = [self.class_dict[c] for c in allowed_classes] if allowed_classes else None

        
        self.labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

                
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        """Returns DataLoader objects for training and validation sets."""
        dataset = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform)
        '''
        if self.num_classes:
            self.allowed_classes = list(self.class_dict.values())[:self.num_classes]
        '''
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.allowed_classes)}
        
        # Apply allowed classes filter if specified
        if self.allowed_classes:
            dataset = self.filter_classes(dataset)
            
        # Limit the number of examples per class if specified
        if self.examples_per_class > 0:
            dataset = self.limit_examples_per_class(dataset)

        # Split dataset into training and validation sets
        train_set, val_set = self.split_dataset(dataset)

        # Create DataLoaders for both sets
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.print_class_counts(train_set, "train")
        self.print_class_counts(val_set, "validation")
        
        if self.output == "dl":
            # Retornar formato DataLoader
            return train_loader, val_loader
        elif self.output == "np":
            # Recolectar los datos en arrays de NumPy
            X_train, y_train = [], []
            for inputs, labels in train_loader:
                X_train.append(inputs.numpy())
                y_train.append(labels.numpy())
    
            X_test, y_test = [], []
            for inputs, labels in val_loader:
                X_test.append(inputs.numpy())
                y_test.append(labels.numpy())
    
            # Concatenar los lotes en un único array de NumPy
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            X_test = np.concatenate(X_test)
            y_test = np.concatenate(y_test)
    
            return X_train, y_train, X_test, y_test, index_mapping
        else:
            raise ValueError(f"Unsupported format: {output}. Use 'dl' for DataLoader or 'np' for NumPy.")


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

        # Limit the number of examples per class
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
        train_set = Subset(dataset.dataset, train_indices)
        val_set = Subset(dataset.dataset, val_indices)
        return train_set, val_set

    def print_class_counts(self, dataset, set_type):
        """Prints the class distribution in the given dataset."""
        class_counts = collections.Counter(dataset.dataset.targets[idx] for idx in dataset.indices)
        #print(f"Class distribution in the {set_type} set:")
        suma = 0
        for class_label, count in sorted(class_counts.items()):
            class_name = list(self.class_dict.keys())[list(self.class_dict.values()).index(class_label)]
            #print(f"\t{class_name} (Class {class_label}): {count} examples")
            suma += count
        #print(f"Number of examples: {suma}")

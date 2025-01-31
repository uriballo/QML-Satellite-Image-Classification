import pytest
import os
import shutil
import numpy as np
from src.utils.dataset import EuroSAT

@pytest.fixture(scope="module")
def dataset():
    test_root = "dataset/EuroSAT_RGB"
    ds = EuroSAT(root=test_root, examples_per_class=100)
    yield ds
    if os.path.exists("test_dataset"):
        shutil.rmtree("test_dataset")

def test_initialization(dataset):
    assert os.path.exists(dataset.root)
    assert dataset.batch_size == 256
    assert dataset.image_size == 64

def test_numpy_output(dataset):
    X_train, y_train, _, _, _ = dataset.get_loaders()
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert len(X_train.shape) == 4  # (batch, channels, height, width)
    assert X_train.shape[1] == 3    # RGB channels

def test_dataloader_output():
    dataset = EuroSAT(output='dl', examples_per_class=100)
    train_loader, val_loader = dataset.get_loaders()
    assert hasattr(train_loader, 'dataset')
    assert hasattr(val_loader, 'dataset')

def test_filtered_classes():
    allowed_classes = ['Forest', 'River']
    dataset = EuroSAT(allowed_classes=allowed_classes, examples_per_class=100)
    _, y_train, _, _, _ = dataset.get_loaders()
    unique_classes = np.unique(y_train)
    assert len(unique_classes) == len(allowed_classes)

def test_examples_per_class_limit():
    examples = 50
    dataset = EuroSAT(examples_per_class=examples)
    _, y_train, _, _, _ = dataset.get_loaders()
    class_counts = np.bincount(y_train)
    assert all(count <= examples for count in class_counts[class_counts > 0])

def test_invalid_output_format():
    dataset = EuroSAT(output='invalid')
    with pytest.raises(ValueError):
        dataset.get_loaders()
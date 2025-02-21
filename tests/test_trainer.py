import pytest
import torch
import numpy as np
from src.utils.training import Trainer
from src.utils.dataset import EuroSAT

class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

@pytest.fixture
def dummy_loader():
    # Create a minimal DataLoader, for instance with a small dummy dataset.
    from torch.utils.data import DataLoader, TensorDataset
    dummy_data = torch.randn(10, 3, 64, 64)
    dummy_labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    return DataLoader(dataset, batch_size=2)

@pytest.fixture
def trainer(dummy_loader):
    return Trainer(
        model=torch.nn.Linear(10, 2),  # A dummy model
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        log=False,
        plot=False,
        allowed_classes=[0, 1]
    )


@pytest.fixture
def dataset():
    return EuroSAT(
        allowed_classes=['AnnualCrop', 'Forest'],
        examples_per_class=10,
        batch_size=4
    )

@pytest.fixture
def model():
    return SimpleModel(num_classes=2)

def test_trainer_initialization(trainer):
    assert trainer.log == False
    assert trainer.plot == False
    assert len(trainer.labels) == 2
    assert trainer.labels == [0, 1]
    assert len(trainer.train_losses) == 0

def test_trainer_fit(trainer, dataset, model):
    trainer.fit()
    
    assert len(trainer.train_losses) == 2
    assert len(trainer.val_losses) == 2
    assert len(trainer.train_accuracies) == 2
    assert len(trainer.val_accuracies) == 2
    assert trainer.confusion_matrix_train is not None
    assert trainer.confusion_matrix_val is not None

def test_trainer_evaluate(trainer, dataset, model):
    loaders = dataset.get_loaders()
    val_loader = loaders[1]
    
    val_loss, val_acc, precision, recall, f1, conf_matrix = trainer._evaluate(val_loader, model)
    
    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
    assert conf_matrix.shape == (2, 2)
    
def test_trainer_metrics_initialization(trainer):
    assert isinstance(trainer.precision, list)
    assert isinstance(trainer.recall, list)
    assert isinstance(trainer.f1, list)
    assert trainer.confusion_matrix_train is None
    assert trainer.confusion_matrix_val is None

def test_trainer_device_selection(trainer):
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert trainer.device == expected_device
import schedulefree
import mlflow
import time
import torch
import schedulefree as sf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from src.utils.plotting import confusion_matrix_plot
from loguru import logger

class Trainer:
    """
    A versatile training pipeline for deep learning models, supporting classical, hybrid and pure quantum architectures.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of training epochs.
        early_stopping (bool): If True, use early stopping based on validation loss.
        patience (int): Number of epochs to wait for improvement before stopping.
        log_wandb (bool): If True, log metrics to Weights & Biases.
        wandb_project (str): W&B project name (used if log_wandb=True).
        wandb_run_name (str): W&B run name (used if log_wandb=True).
        use_quantum (bool): If True, enables quantum-based layers.
        plot (bool): If True, generates and displays confusion matrix.
        allowed_classes (list): List of class labels allowed for training and evaluation. If None, all classes are used.
        lr (float): Learning rate for the optimizer.
        use_schedulefree (bool): If True, enables schedulefree module.
    """
    def __init__(self,
                model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 10,
                early_stopping: bool = False,
                patience: int = 3,
                log: bool = True,
                mlflow_project: str = 'Prueba train',
                mlflow_run_name: str = '1',
                use_quantum: bool = True,
                plot: bool = True,
                allowed_classes: list = None,
                lr: float = 0.01,
                use_schedulefree: bool = False,
                mlflow_params: dict = None,
                ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.log = log
        self.mlflow_project = mlflow_project
        self.mlflow_run_name = mlflow_run_name
        self.use_quantum = use_quantum
        self.plot = plot
        self.lr = lr
        self.schedulefree = use_schedulefree
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.criterion = torch.nn.CrossEntropyLoss()
        self.plot = plot
        self.confusion_matrix_train = None
        self.confusion_matrix_val = None
        self.labels = allowed_classes
        self._best_val_loss = float('inf')
        self._early_stop_counter = 0
        self.mlflow_params = mlflow_params or {} 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self):
        model = self.model.to(self.device)
        train_loader, val_loader = self.train_loader, self.val_loader
        lr, epochs = self.lr, self.epochs
        
        criterion = self.criterion

        if self.schedulefree:
            self.optimizer = sf.AdamWScheduleFree(model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        
        all_labels = []
        all_preds = []        

        if self.log:
            mlflow.set_experiment(self.mlflow_project)
            mlflow.start_run(run_name=self.mlflow_run_name)
            
            # Combine passed parameters with extracted model attributes
            params_to_log = {
                # Start with any parameters passed to the constructor
                **self.mlflow_params,
                
                # Add basic training parameters
                "lr": self.lr,
                "epochs": self.epochs,
                "use_quantum": self.use_quantum,
                "use_schedulefree": self.schedulefree,
                "early_stopping": self.early_stopping,
                "patience": self.patience,
                "batch_size": next(iter(self.train_loader))[0].shape[0] if len(self.train_loader) > 0 else None,
                
                # Add model parameters
                "model_type": self.model.__class__.__name__,
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                
                # Extract model-specific attributes if they exist
                "qkernel_shape": getattr(self.model, 'qkernel_shape', None),
                "fc_hidden_dim": getattr(self.model, 'fc_hidden_dim', None),
                "n_filters_1": getattr(self.model, 'n_filters_1', None),
            }
            
            # Log model attributes that might be dictionaries or complex objects
            if hasattr(self.model, 'embedding'):
                params_to_log["embedding_func"] = self.model.embedding.get("func").__name__ if self.model.embedding.get("func") else None
                # Log embedding parameters if they exist
                if self.model.embedding.get("func_params"):
                    for key, value in self.model.embedding.get("func_params").items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            params_to_log[f"embedding_{key}"] = value
            
            if hasattr(self.model, 'variational'):
                params_to_log["circuit_func"] = self.model.variational.get("func").__name__ if self.model.variational.get("func") else None
                # Log circuit parameters if they exist
                if self.model.variational.get("func_params"):
                    for key, value in self.model.variational.get("func_params").items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            params_to_log[f"circuit_{key}"] = value
            
            if hasattr(self.model, 'measurement'):
                params_to_log["measurement_func"] = self.model.measurement.get("func").__name__ if self.model.measurement.get("func") else None
                # Log measurement parameters if they exist
                if self.model.measurement.get("func_params"):
                    for key, value in self.model.measurement.get("func_params").items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            params_to_log[f"measurement_{key}"] = value
            
            # Log additional model attributes
            if hasattr(self.model, 'dataset'):
                params_to_log["dataset"] = self.model.dataset
            
            if hasattr(self.model, 'image_size'):
                params_to_log["image_size"] = self.model.image_size
            
            # Log all parameters
            mlflow.log_params(params_to_log)

        for epoch in range(epochs):
            model.train()

            if self.schedulefree:
                self.optimizer.train()
            
            running_loss, correct, total = 0.0, 0, 0

            start_time = time.time()
                
            for (inputs, labels) in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)   

                self.optimizer.zero_grad()
                outputs = model(inputs).to(self.device)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()*inputs.size(0)
                
                _, preds = torch.max(outputs, dim = 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            self.confusion_matrix_train = confusion_matrix(all_labels, all_preds)
        

            train_loss = running_loss / total
            train_acc = 100 * correct / total
            
            elapsed_time = time.time() - start_time

            # VAL
            val_loss, val_acc, precision, recall, f1, confusion_matrix_val = self._evaluate(val_loader, model)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.precision.append(precision)
            self.recall.append(recall)
            self.f1.append(f1)
            
            # Early stopping check
            if self.early_stopping:
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._early_stop_counter = 0
                else:
                    self._early_stop_counter += 1
                    if self._early_stop_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            # Logging
            if self.log:
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("precision", precision, step=epoch)
                mlflow.log_metric("recall", recall, step=epoch)
                mlflow.log_metric("f1", f1, step=epoch)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
        
            logger.debug(f"Epoch [{epoch+1}/{epochs}]: "
                         f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
                         f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

        mlflow.end_run()

        if self.plot:
                confusion_matrix_plot(self.confusion_matrix_train, self.labels, title = 'Confusion matrix train')
                confusion_matrix_plot(confusion_matrix_val, self.labels, title = 'Confusion matrix val')

    def _evaluate(self, val_loader: DataLoader, model):
        """
        Evaluate the model on a given DataLoader. Returns average loss and accuracy.
        """
        
        model.eval()
        
        if self.schedulefree:
            self.optimizer.eval()
            
        val_loss, val_acc, running_loss, correct, total = 0.0, 0.0, 0.0, 0, 0

        criterion = self.criterion

        all_labels = []
        all_preds = []

        start_time = time.time()

        with torch.no_grad():
            for (inputs, labels) in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs).to(self.device)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, dim = 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss = running_loss/total
        val_acc = 100 * correct/total
        elapsed_time = time.time() - start_time

        precision = precision_score(all_labels, all_preds, average = 'macro', zero_division = 1)
        recall = recall_score(all_labels, all_preds, average = 'macro', zero_division = 1)
        f1 = f1_score(all_labels, all_preds, average = 'macro', zero_division = 1)

        self.confusion_matrix_val = confusion_matrix(all_labels, all_preds)
        
        
        return val_loss, val_acc, precision, recall, f1, self.confusion_matrix_val

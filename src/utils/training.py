import wandb
import time
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from src.utils.plotting import confusion_matrix_plot

class Trainer:
    def __init__(self,
                log_wandb: bool = True,
                wandb_project: str = 'Prueba train',
                wandb_run_name: str = '1',
                use_quantum: bool = True,
                plot: bool = True,
                allowed_class_idx = None,
                ):

        self.log_wandb = log_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.use_quantum = use_quantum
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
        
        labels = ['Annual\nCrop', 'Forest',
                  'Herbaceous\nVegetation',
                  'Highway', 'Industrial',
                  'Pasture', 'Permanent\nCrop',
                  'Residential', 'River',
                  'SeaLake']

        self.labels = labels if allowed_class_idx is None else [labels[i] for i in allowed_class_idx if i < len(labels)]
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self, 
            model,
            train_loader: DataLoader,  
            val_loader: DataLoader,
            learning_rate: float,
            epochs: int = 10,
           ):
        
        model = model.to(self.device)
        
        criterion = self.criterion
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        
        all_labels = []
        all_preds = []        

        if self.log_wandb:
            wandb.init(project = self.wandb_project, name = self.wandb_run_name)
            wandb.config.update({
                "lr": learning_rate,
                "epochs": epochs,
                #"batch_size": self.batch_size,
                #"qkernel_shape": self.qkernel_shape,
                "use_quantum": self.use_quantum,
            })
        
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            start_time = time.time()
                
            for (inputs, labels) in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)   
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
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
           
            # Logging
            if self.log_wandb:
                
                wandb.log({
                    #"epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
        
            print(f"Epoch [{epoch+1}/{epochs}]: "
                f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}%, "
                f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}%")
            
        
        
        wandb.finish()

        if self.plot:
                confusion_matrix_plot(self.confusion_matrix_train, self.labels, title = 'Confusion matrix train')
                confusion_matrix_plot(confusion_matrix_val, self.labels, title = 'Confusion matrix val')

    def _evaluate(self, val_loader: DataLoader, model):
        """
        Evaluate the model on a given DataLoader. Returns average loss and accuracy.
        """
        model.eval()
        val_loss, val_acc, running_loss, correct, total = 0.0, 0.0, 0.0, 0, 0

        criterion = self.criterion

        all_labels = []
        all_preds = []

        start_time = time.time()

        with torch.no_grad():
            for (inputs, labels) in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
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

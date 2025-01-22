import torch.nn as nn
import torch.nn.functional as F

from src.nn.qlayers import quantum_linear

class HQNN_Parallel(nn.Module):
    def __init__(self, q_circuit, weight_shapes, num_qubits=4, embedding="n", n_classes=10):
        super(HQNN_Parallel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
       
        self.qfc = QuantumLinear(256, 256, q_circuit, weight_shapes, num_qubits, embedding)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_classes)
        print("Number of trainable parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))

                
    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv2(x) 
        x = self.bn2(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv3(x) 
        x = self.bn3(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)

        x = self.conv4(x) 
        x = self.bn4(x)  # Batch norm before ReLU
        x = F.leaky_relu(x) 
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, 0.2)     

        x = x.view(x.shape[0], -1)
        x = F.normalize(x)
        x = self.qfc(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        
        return x
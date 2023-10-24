"""
This module is used to load and preprocess the MNIST dataset and define a PyTorch model.
"""

# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# torch is the main PyTorch library
import torch
# torch.nn provides classes for building neural networks
import torch.nn as nn
# torch.optim provides classes for implementing various optimization algorithms
import torch.optim as optim
# torchvision.datasets provides classes for loading and manipulating datasets
from torchvision import datasets, transforms
# torchvision.transforms provides classes for transforming images
# torch.utils.data provides utilities for loading and manipulating data
from torch.utils.data import DataLoader
# numpy is used for numerical operations
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# 'transform' is a sequence of transformations applied to the images in the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 'trainset' represents the MNIST dataset
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
# 'trainloader' is a data loader for the MNIST dataset
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    This class defines a simple feed-forward neural network for classifying MNIST images.
    The network has three fully connected layers and uses ReLU activation functions.
    The forward method takes an input tensor, reshapes it, and passes it through the network.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)

# Step 3: Train the Model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "mnist_model.pth")
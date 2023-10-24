"""
This script loads and preprocesses the MNIST dataset, defines a PyTorch model, and trains the model.
"""
# PIL is used for opening, manipulating, and saving many different image file formats
import PIL.Image
# torch is the main PyTorch library
import torch
# torch.nn provides classes for building neural networks
import torch.nn as nn
# torch.optim provides classes for implementing various optimization algorithms
import torch.optim as optim
# torchvision.datasets provides classes for loading and using various popular datasets
from torchvision import datasets
# torchvision.transforms provides classes for transforming images
from torchvision import transforms
# torch.utils.data provides classes for loading data in parallel
from torch.utils.data import DataLoader
# numpy is used for numerical operations
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# This is a sequence of preprocessing steps to be applied to the images in the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# This represents the MNIST training dataset
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
# This is a data loader for batching and shuffling the training data
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
# This class defines the architecture of the PyTorch model
class Net(nn.Module):
    """
    This class defines the architecture of the PyTorch model.
    """
    def __init__(self):
        """
        This method initializes the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        """
        This method defines the forward pass of the model.
        """
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
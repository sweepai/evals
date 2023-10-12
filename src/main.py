"""
This module defines and trains a PyTorch model for recognizing digits in images using the MNIST dataset.
"""

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# The transform converts the images to tensors and normalizes them to have a mean of 0.5 and a standard deviation of 0.5.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# The MNIST dataset is downloaded and loaded with the defined transform applied.
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    This class defines the architecture of the PyTorch model.
    """
    def __init__(self):
        super().__init__()
        # The first layer takes the flattened input image and outputs 128 features.
        self.fc1 = nn.Linear(28 * 28, 128)
        # The second layer takes the 128 features and outputs 64 features.
        self.fc2 = nn.Linear(128, 64)
        # The third layer takes the 64 features and outputs 10 classes, one for each digit.
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        """
        This method defines the forward pass of the model.
        """
        # The input tensor is reshaped to a flat tensor.
        x = x.view(-1, 28 * 28)
        # The first layer is applied with a ReLU activation function.
        x = nn.functional.relu(self.fc1(x))
        # The second layer is applied with a ReLU activation function.
        x = nn.functional.relu(self.fc2(x))
        # The third layer is applied without an activation function.
        x = self.fc3(x)
        # The output is passed through a log softmax function to get the log probabilities of each class.
        return nn.functional.log_softmax(x, dim=1)

# Step 3: Train the Model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    for images, labels in trainloader:
        # The gradients are zeroed to prevent accumulation from previous iterations.
        optimizer.zero_grad()
        # The output of the model is computed.
        output = model(images)
        # The loss is computed using the negative log likelihood loss function.
        loss = criterion(output, labels)
        # The gradients are computed via backpropagation.
        loss.backward()
        # The weights are updated using the computed gradients.
        optimizer.step()

# The state_dict of the model is saved instead of the model itself because it contains the parameters of the model and is more portable.
torch.save(model.state_dict(), "mnist_model.pth")
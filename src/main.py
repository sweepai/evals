"""
This script defines the data loading and preprocessing steps, as well as the PyTorch model for MNIST digit classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Step 1: Load MNIST Data and Preprocess
# The transformation pipeline consists of two steps:
# 1. transforms.ToTensor() - Converts the input image to PyTorch tensor.
# 2. transforms.Normalize((0.5,), (0.5,)) - Normalizes the tensor with mean 0.5 and standard deviation 0.5.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
"""
This class defines the PyTorch model for MNIST digit classification.
The model consists of three fully connected layers.
"""


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # First fully connected layer, takes input of size 28*28 and outputs size 128.
        self.fc1 = nn.Linear(28 * 28, 128)
        # Second fully connected layer, takes input of size 128 and outputs size 64.
        self.fc2 = nn.Linear(128, 64)
        # Third fully connected layer, takes input of size 64 and outputs size 10 (for 10 digit classes).
        self.fc3 = nn.Linear(64, 10)

    """
    This method defines the forward pass of the model.
    It applies the following transformations to the input:
    1. Reshapes the input to a 1D tensor.
    2. Applies the first fully connected layer followed by a ReLU activation function.
    3. Applies the second fully connected layer followed by a ReLU activation function.
    4. Applies the third fully connected layer.
    5. Applies a log softmax function to the output of the third layer.
    """

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

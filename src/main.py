from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
"""
This class defines the architecture of a neural network for digit recognition.
It consists of three fully connected layers.
"""
class Net(nn.Module):
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

# Training loop: This loop trains the neural network on the MNIST dataset.
epochs = 3
for epoch in range(epochs):
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "mnist_model.pth")
        # Zero the gradients before a new iteration
        # Forward propagation: Pass the images through the model to get the output
        # Compute the loss between the output and the actual labels
        # Backpropagation: Compute the gradients of the loss with respect to the model's parameters
        # Optimizer step: Update the model's parameters
    """
    Initialize the neural network with three fully connected layers.
    The first layer (fc1) has 128 neurons and takes as input the flattened 28x28 pixel MNIST images.
    The second layer (fc2) has 64 neurons.
    The third layer (fc3) has 10 neurons, corresponding to the 10 possible digits, and will output the network's log-probabilities.
    """
    """
    Defines the forward pass of the neural network.
    The input images are first flattened and then passed through the three layers with ReLU activation functions applied after the first and second layers.
    The output of the third layer is passed through a log softmax function to obtain the network's log-probabilities.
    """
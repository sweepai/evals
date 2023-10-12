"""
This module loads the MNIST dataset, preprocesses the data, defines a PyTorch model, trains the model on the data, and saves the trained model. 

The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The PyTorch model is a simple neural network with three fully connected layers. The model is trained using the stochastic gradient descent optimization algorithm and the negative log likelihood loss function.
"""

# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library based on the Torch library
import torch
# nn is a sublibrary of PyTorch, used for building neural networks
import torch.nn as nn
# optim is a sublibrary of PyTorch, used for implementing various optimization algorithms
import torch.optim as optim
# torchvision is a library that has datasets, model architectures and image transformation tools
from torchvision import datasets, transforms
# DataLoader is a PyTorch class for loading and iterating over datasets
from torch.utils.data import DataLoader
# numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# transforms.Compose is a simple way to build a transformation pipeline. Here, it is used to convert the images to PyTorch tensors and normalize them.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# datasets.MNIST is a PyTorch function for loading the MNIST dataset. Here, it is used to download the training set and apply the transformations defined above.
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
# DataLoader is a PyTorch function for creating an iterable over the dataset. Here, it is used to shuffle the training data and batch it.
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
# The Net class defines a simple neural network with three fully connected layers. The forward method defines the forward pass of the network.
class Net(nn.Module):
    """
    A simple neural network with three fully connected layers.

    The network takes as input a batch of grayscale images of size 28x28, flattens them into a vector of size 784, and passes them through three fully connected layers. The first two layers use the ReLU activation function, and the final layer uses the log softmax function for output.
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
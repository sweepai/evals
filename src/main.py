from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from cnn import CNN

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Instantiate the CNN Model
model = CNN()

# Step 3: Define the Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: Train the Model using the CNN's train method
epochs = 3
for epoch in range(epochs):
    model.train(trainloader, optimizer)

torch.save(model.state_dict(), "mnist_model.pth")
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
from cnn import CNN, train_cnn

# Step 2: Define the PyTorch Model
model = CNN()

# Step 3: Train the Model
epochs = 3
train_cnn(model, trainloader, epochs)

torch.save(model.state_dict(), "mnist_model.pth")
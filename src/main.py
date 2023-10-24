import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN, train_cnn

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the CNN Model
model = CNN()
print("CNN model defined.")

# Step 3: Train the CNN Model
epochs = 10
train_cnn(model, trainloader, epochs)
print("CNN model trained.")

torch.save(model.state_dict(), "mnist_model.pth")

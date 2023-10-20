import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
model = CNN()


# Step 3: Train the Model
def train(model, trainloader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    for epoch in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")


train(model, trainloader)
torch.save(model.state_dict(), "mnist_cnn_model.pth")

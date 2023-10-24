import pytest
import torch
import torch.nn as nn
import os
from main import Net
from unittest.mock import MagicMock

def test_net_initialization():
    model = Net()
    assert isinstance(model.fc1, nn.Linear)
    assert isinstance(model.fc2, nn.Linear)
    assert isinstance(model.fc3, nn.Linear)

def test_net_forward():
    model = Net()
    input_tensor = torch.randn(64, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (64, 10)

def test_training_loop():
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    trainloader = MagicMock()
    trainloader.return_value = [torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))]

    initial_loss = float('inf')
    epochs = 3
    for epoch in range(epochs):
        for images, labels in trainloader():
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        assert loss.item() < initial_loss
        initial_loss = loss.item()

def test_model_saving():
    assert os.path.isfile("mnist_model.pth")

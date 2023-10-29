import pytest
from unittest.mock import Mock
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import Net

def test_Net():
    model = Net()
    mock_input = torch.randn(64, 1, 28, 28)
    output = model(mock_input)
    assert output.shape == (64, 10)

def test_data_loading():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    assert len(trainloader) == len(trainset) // 64

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    assert images.shape == (64, 1, 28, 28)
    assert labels.shape == (64,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

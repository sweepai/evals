import pytest
from unittest.mock import Mock, call
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import Net, transform, trainset, trainloader, model, optimizer, criterion

def test_data_loading_and_preprocessing(mocker):
    """
    Test data loading and preprocessing.
    """
    mock_transforms = mocker.patch('torchvision.transforms.Compose')
    mock_datasets = mocker.patch('torchvision.datasets.MNIST')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
    
    mock_transforms.assert_called_once_with([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mock_datasets.assert_called_once_with('.', download=True, train=True, transform=transform)

def test_model_definition(mocker):
    """
    Test model definition.
    """
    mock_linear = mocker.patch('torch.nn.Linear')
    mock_relu = mocker.patch('torch.nn.functional.relu')
    
    model = Net()
    
    assert isinstance(model, Net)
    assert model.fc1.out_features == 128
    assert model.fc2.out_features == 64
    assert model.fc3.out_features == 10
    
    mock_linear.assert_has_calls([call(28 * 28, 128), call(128, 64), call(64, 10)])
    mock_relu.assert_called()

def test_training_process(mocker):
    """
    Test training process.
    """
    mock_dataloader = mocker.patch('torch.utils.data.DataLoader')
    mock_optimizer = mocker.patch('torch.optim.SGD')
    mock_loss = mocker.patch('torch.nn.NLLLoss')
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    
    mock_dataloader.assert_called_once_with(trainset, batch_size=64, shuffle=True)
    mock_optimizer.assert_called_once_with(model.parameters(), lr=0.01)
    mock_loss.assert_called_once()

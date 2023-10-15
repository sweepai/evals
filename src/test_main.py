import pytest
from pytest_mock import MockerFixture
from main import transform, trainset, trainloader, Net, model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def test_data_loading_and_preprocessing(mocker: MockerFixture):
    mock_mnist = mocker.patch.object(datasets, 'MNIST')
    mock_dataloader = mocker.patch.object(DataLoader, '__init__')
    
    mock_mnist.assert_called_once_with('.', download=True, train=True, transform=transform)
    mock_dataloader.assert_called_once_with(trainset, batch_size=64, shuffle=True)

def test_model_definition(mocker: MockerFixture):
    mock_model = mocker.patch.object(Net, '__init__')
    
    mock_model.assert_called_once()

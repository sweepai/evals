import pytest
from pytest_mock import MockerFixture
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import Net

def test_data_loading_and_preprocessing(mocker: MockerFixture):
    """
    Test the data loading and preprocessing steps.
    """
    mock_mnist = mocker.patch.object(datasets, 'MNIST')
    mock_dataloader = mocker.patch.object(DataLoader, '__init__')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    mock_mnist.assert_called_once_with('.', download=True, train=True, transform=transform)
    mock_dataloader.assert_called_once_with(trainset, batch_size=64, shuffle=True)

def test_model_definition():
    """
    Test the model definition.
    """
    model = Net()

    self.assertEqual(isinstance(model.fc1, nn.Linear), True)
    self.assertEqual(isinstance(model.fc2, nn.Linear), True)
    self.assertEqual(isinstance(model.fc3, nn.Linear), True)

    self.assertEqual(model.fc1.in_features, 28 * 28)
    self.assertEqual(model.fc1.out_features, 128)

    self.assertEqual(model.fc2.in_features, 128)
    self.assertEqual(model.fc2.out_features, 64)

    self.assertEqual(model.fc3.in_features, 64)
    self.assertEqual(model.fc3.out_features, 10)

    self.assertEqual(output.shape, (1, 10))

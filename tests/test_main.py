import pytest
import subprocess
from pytest_mock import MockerFixture
from torchvision import datasets
from torch.utils.data import DataLoader
from src import main

def setup_module(module):
    """Setup for the test module."""
    try:
        subprocess.run(["poetry", "--version"], shell=True, check=True)
    except subprocess.CalledProcessError:
        subprocess.run("curl -sSL https://install.python-poetry.org | python3 -", shell=True, check=True)

def test_data_loading_and_preprocessing(mocker: MockerFixture):
    """Test the data loading and preprocessing steps."""
    mock_mnist = mocker.patch.object(datasets, 'MNIST')
    mock_dataloader = mocker.patch.object(DataLoader, '__init__')

    main.load_and_preprocess_data()

    mock_mnist.assert_called_once_with('.', download=True, train=True, transform=main.transform)
    mock_dataloader.assert_called_once_with(mock_mnist.return_value, batch_size=64, shuffle=True)

def test_model_definition():
    """Test the model definition."""
    model = main.Net()

    assert isinstance(model, main.Net)
    assert isinstance(model.fc1, nn.Linear)
    assert model.fc1.in_features == 784
    assert model.fc1.out_features == 128
    assert isinstance(model.fc2, nn.Linear)
    assert model.fc2.in_features == 128
    assert model.fc2.out_features == 64
    assert isinstance(model.fc3, nn.Linear)
    assert model.fc3.in_features == 64
    assert model.fc3.out_features == 10

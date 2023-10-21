from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from main import Net, trainloader, transform


def test_data_loading_and_preprocessing():
    """
    Test the data loading and preprocessing steps.
    """
    with mock.patch.object(datasets, "MNIST") as mock_mnist:
        with mock.patch.object(DataLoader, "__init__") as mock_dataloader:
            mock_mnist.return_value = None
            mock_dataloader.return_value = None
            pytest.assert_(trainloader is not None, "trainloader should not be None")
            mock_mnist.assert_called_once_with(
                ".", download=True, train=True, transform=transform
            )
            mock_dataloader.assert_called_once_with(None, batch_size=64, shuffle=True)


def test_model_definition():
    """
    Test the model definition.
    """
    with mock.patch.object(torch.nn, "Module") as mock_module:
        model = Net()
        pytest.assert_(model is not None, "model should not be None")
        mock_net.assert_called()
        assert model.fc1.in_features == 28 * 28, "Input features of the first layer should be 28 * 28"
        assert model.fc1.out_features == 128, "Output features of the first layer should be 128"
        pytest.assert_(model.fc2.in_features == 128, "Input features of the second layer should be 128")
        pytest.assert_(model.fc2.out_features == 64, "Output features of the second layer should be 64")
        assert model.fc3.in_features == 64, "Input features of the third layer should be 64"
        pytest.assert_(model.fc3.out_features == 10, "Output features of the third layer should be 10")

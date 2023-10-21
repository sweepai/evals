from unittest import mock

import pytest
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
            pytest.assert_not_none(trainloader)
            mock_mnist.assert_called_once_with(
                ".", download=True, train=True, transform=transform
            )
            mock_dataloader.assert_called_once_with(None, batch_size=64, shuffle=True)


def test_model_definition():
    """
    Test the model definition.
    """
    with mock.patch.object(Net, "__init__", return_value=None) as mock_net:
        model = Net()
        pytest.assert_not_none(model)
        mock_net.assert_called()

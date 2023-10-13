import pytest
from unittest.mock import Mock, call
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from src import main

def test_data_loading(mocker):
    """
    Test the data loading step in main.py.
    This test checks that the datasets.MNIST and DataLoader functions are called with the correct arguments.
    """
    mock_mnist = mocker.patch('torchvision.datasets.MNIST', return_value=[])
    mock_dataloader = mocker.patch('torch.utils.data.DataLoader', return_value=[])

    main.load_data()

    mock_mnist.assert_called_once_with('.', download=True, train=True, transform=main.transform)
    mock_dataloader.assert_called_once_with(mock_mnist.return_value, batch_size=64, shuffle=True)

def test_data_preprocessing(mocker):
    """
    Test the data preprocessing step in main.py.
    This test checks that the transforms.Compose function is called with the correct arguments.
    """
    mock_compose = mocker.patch('torchvision.transforms.Compose', return_value=[])

    main.preprocess_data()

    mock_compose.assert_called_once_with([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

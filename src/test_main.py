import pytest
from pytest_mock import MockerFixture
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from main import transform, Net

def test_transform(mocker: MockerFixture):
    mock_image = mocker.Mock(spec=Image.Image)
    mock_image.size = (28, 28)
    transformed_image = transform(mock_image)
    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (1, 28, 28)

def test_dataloader(mocker: MockerFixture):
    mock_dataset = mocker.Mock()
    mock_dataset.__len__.return_value = 64
    dataloader = DataLoader(mock_dataset, batch_size=64, shuffle=True)
    assert len(dataloader) == 1

def test_model(mocker: MockerFixture):
    model = Net()
    mock_input = mocker.Mock(spec=torch.Tensor)
    mock_input.shape = (1, 28, 28)
    output = model(mock_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 10)

import pytest
from pytest_mock import MockerFixture
from main import Net, transform
from api import app
from fastapi.testclient import TestClient
from PIL import Image
import torch
import torch.nn as nn
import io

def test_net_init(mocker: MockerFixture):
    mock_super_init = mocker.patch('torch.nn.Module.__init__')
    net = Net()
    assert mock_super_init.assert_called_once()

def test_net_forward(mocker: MockerFixture):
    mock_input = mocker.patch('torch.Tensor')
    mock_relu = mocker.patch('torch.nn.functional.relu')
    mock_log_softmax = mocker.patch('torch.nn.functional.log_softmax')
    net = Net()
    net.forward(mock_input)
    assert mock_relu.assert_any_call(net.fc1(mock_input.view(-1, 28 * 28)))
    assert mock_relu.assert_any_call(net.fc2(mock_relu.return_value))
    assert mock_log_softmax.assert_called_once_with(net.fc3(mock_relu.return_value), dim=1)

def test_predict(mocker: MockerFixture):
    mock_file = mocker.patch('fastapi.UploadFile')
    mock_image_open = mocker.patch('PIL.Image.open')
    mock_image_open.return_value.convert.return_value = Image.new('L', (28, 28))
    client = TestClient(app)
    response = client.post("/predict/", files={"file": ("filename", io.BytesIO(), "image/png")})
    assert response.status_code == 200, "Expected status code 200"
    assert 'prediction' in response.json(), "Expected 'prediction' in response"

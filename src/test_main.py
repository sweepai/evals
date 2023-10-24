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
    assert mock_super_init.assert_called_once(), "Expected super init to be called once"

def test_net_forward(mocker: MockerFixture):
    mock_input = mocker.patch('torch.Tensor')
    mock_relu = mocker.patch('torch.nn.functional.relu')
    mock_log_softmax = mocker.patch('torch.nn.functional.log_softmax')
    net = Net()
    net.forward(mock_input)
    assert mock_relu.called, "Expected relu to be called"
    assert mock_log_softmax.called, "Expected log softmax to be called"

def test_predict(mocker: MockerFixture):
    mock_file = mocker.patch('fastapi.UploadFile')
    mock_image_open = mocker.patch('PIL.Image.open')
    mock_image_open.return_value.convert.return_value = Image.new('L', (28, 28))
    client = TestClient(app)
    response = client.post("/predict/", files={"file": ("filename", io.BytesIO(), "image/png")})
    assert response.status_code == 200, "Expected status code 200"
    assert 'prediction' in response.json(), "Expected 'prediction' in response"

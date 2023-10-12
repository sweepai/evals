# evals

This project provides a Python implementation for training and evaluating a simple neural network on the MNIST dataset using PyTorch.

## MNISTTrainer class

The `MNISTTrainer` class is used to train and evaluate the model. It is defined in `src/main.py`.

### Usage

First, create an instance of the `MNISTTrainer` class:

```python
trainer = MNISTTrainer()
```

You can then train the model using the `train` method:

```python
trainer.train()
```

The trained model's parameters are automatically saved to a file named "mnist_model.pth".

To load the model parameters from this file, use the following code:

```python
trainer.model.load_state_dict(torch.load("mnist_model.pth"))
```

To evaluate the model, you can use the `predict` method in `src/api.py`. This method takes an image file as input and returns the model's prediction.

## Dependencies

This project requires the following Python libraries:

- PyTorch
- torchvision
- numpy
- PIL
- FastAPI

You can install these dependencies using pip:

```bash
pip install torch torchvision numpy pillow fastapi
```

## Running the project

To run the project, first start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

You can then send a POST request to the `/predict` endpoint with an image file to get the model's prediction.
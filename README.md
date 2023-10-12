# evals

This project contains a PyTorch implementation for training and evaluating a simple neural network on the MNIST dataset.

## ModelTrainer class

The `ModelTrainer` class is used to train and evaluate the model. Here is how you can use it:

```python
from main import ModelTrainer

# Create an instance of ModelTrainer
trainer = ModelTrainer()

# Run the training and evaluation process
trainer.run()
```

The `ModelTrainer` class has the following methods:

- `load_data()`: Loads the MNIST dataset and applies the necessary transformations.
- `define_model()`: Defines the model architecture.
- `train_model()`: Trains the model on the MNIST dataset.
- `evaluate()`: Evaluates the model performance on the test dataset.
- `run()`: Runs the entire process (data loading, model definition, training, and evaluation).

## Saving and Loading the Model

After training, the model's parameters are saved to a file named "mnist_model.pth". You can load the model parameters using the `torch.load()` function:

```python
model.load_state_dict(torch.load("mnist_model.pth"))
```
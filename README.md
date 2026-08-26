# Custom Deep Learning Framework

A from-scratch, pure-NumPy deep learning framework supporting Neural Networks (MLP), Convolutional Neural Networks (CNNs), and regression models.

This document provides a quickstart guide and API reference for building, training, and evaluating models using the framework.

## Installation / Setup
Ensure you have the required dependencies installed (NumPy, Pandas, Pillow, etc.).
```bash
uv init
uv add -r requirements.txt
```

## Quickstart API Guide

### 1. Imports
Import the core components from the `deep_learning` package:
```python
from deep_learning import (
    Network, 
    InputLayer, Dense, Conv2D, MaxPool2D, Flatten,
    ActivationFunction, LossFunction
)
```

### 2. Prepare Data
Ensure your inputs and targets are NumPy arrays:
- **Classification Targets**: One-hot encoded (e.g., shape `(N, classes)`).
- **CNN Inputs**: Shaped as `(N, Channels, Height, Width)`.
- **Dense/MLP Inputs**: Shaped as `(N, Features)`.

### 3. Define the Architecture
Construct your model as a standard Python list of layers. The first layer must always be an `InputLayer` initialized with the training inputs.

**Example CNN Architecture:**
```python
act = ActivationFunction

layers = [
    InputLayer(x_train),
    Conv2D(16, kernel_size=3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
    MaxPool2D(pool_size=2, stride=2),
    Conv2D(32, kernel_size=3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
    MaxPool2D(pool_size=2, stride=2),
    Flatten(),
    Dense(256, act_func=act.ReLU, use_dropout=True, drop_rate=0.3),
    Dense(10, act_func=act.softmax)
]
```

### 4. Initialize the Network
Pass the layers and hyperparameters to the `Network` class:
```python
model = Network(
    layers=layers,
    training_set=(x_train, y_train), # Tuple of (inputs, targets)
    test_set=(x_test, y_test),       # Optional evaluation set
    loss_func=LossFunction.cc_loss,  # cc_loss for classification, MSE for regression
    batch=32,                        # Batch size (None for full-batch training)
    learning_rate=0.01,
    epoch_limit=10,
    iteration_event_trigger=1        # How often to print training progress logs
)
```

### 5. Training
Trigger the training loop using `.fit_model()`. The model will iteratively perform forward propagation, backpropagation, and weight updates.
```python
model.fit_model()
```

### 6. Evaluation
Evaluate the model against the `test_set` provided during initialization. 
- For classification (`cc_loss`), it returns the accuracy percentage.
- For regression (`MSE`), it returns the R-squared percentage.
```python
accuracy = model.evaluate()
print(f"Test Accuracy: {accuracy:.2f}%")
```

### 7. Inference / Prediction
Run inference on new data using `.predict()`.
```python
predictions = model.predict(input=new_data_array)
predicted_classes = np.argmax(predictions, axis=1)
```

### 8. Save and Load Weights
You can persist trained weights and biases to a `.npz` file and reload them later to skip training.
```python
# Save weights
model.save_weights("my_model_weights.npz")

# Load weights
model.load_weights("my_model_weights.npz")
```

## Available Layers
- `InputLayer(inputs)`: Placeholder for the input shape and data.
- `Dense(n_neurons, act_func, use_dropout=False, drop_rate=0.0)`: Fully connected layer.
- `Conv2D(n_kernels, kernel_size, act_func, stride, padding, use_bn=False)`: 2D Convolutional layer.
- `MaxPool2D(pool_size, stride)`: 2D Max pooling layer.
- `Flatten()`: Flattens multi-dimensional inputs into a 1D vector (often used before Dense layers).

## Available Activation & Loss Functions
- **Activations (`ActivationFunction`)**: `ReLU`, `softmax`, `sigmoid`, `linear`
- **Losses (`LossFunction`)**: `cc_loss` (Categorical Cross-Entropy), `MSE` (Mean Squared Error)
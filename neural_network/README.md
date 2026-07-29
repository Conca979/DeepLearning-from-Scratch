# Neural Network from Scratch

This directory contains a flexible, purely NumPy-based implementation of a fully connected Feedforward Neural Network. It demonstrates core deep learning concepts including forward propagation, backpropagation, and mini-batch training.

## Algorithms Explained

### 1. Forward Propagation
Data passes through the network layers sequentially. For each hidden layer, the output is computed as:
$Z^{[l]} = A^{[l-1]} W^{[l]T} + b^{[l]}$
$A^{[l]} = g^{[l]}(Z^{[l]})$
Where:
- $W^{[l]}$ and $b^{[l]}$ are the weights and biases of layer $l$.
- $g^{[l]}$ is the activation function (e.g., ReLU, Sigmoid).
- $A^{[l-1]}$ is the output from the previous layer.

### 2. Loss Functions
The network evaluates its predictions against target values using a loss function $L(Y, \hat{Y})$.
- **Mean Squared Error (MSE):** Used for regression tasks.
- **Categorical Cross-Entropy:** Used with Softmax for multi-class classification tasks.

### 3. Backpropagation (Gradient Descent)
The network learns by calculating the gradient of the loss function with respect to its weights and biases, propagating the error backwards using the chain rule.

**Delta Term Calculation:**
For the output layer:
$\delta^{[L]} = \frac{\partial L}{\partial Z^{[L]}} = \frac{\partial L}{\partial A^{[L]}} \odot g'^{[L]}(Z^{[L]})$

For hidden layers:
$\delta^{[l]} = (\delta^{[l+1]} W^{[l+1]}) \odot g'^{[l]}(Z^{[l]})$

**Weight & Bias Updates:**
$W^{[l]} := W^{[l]} - \alpha (\delta^{[l]T} A^{[l-1]})$
$b^{[l]} := b^{[l]} - \alpha \sum \delta^{[l]}$
Where $\alpha$ is the learning rate.

### 4. Weight Initialization

- **He Initialization:** Scaled by $\sqrt{\frac{2}{\text{numberNeuronInPreviousLayer}}} $, optimized for ReLU activations to prevent vanishing gradients.
- **Xavier Initialization:** Scaled by $\sqrt{\frac{1}{\text{numberNeuronInPreviousLayer}}} $, used for Sigmoid and Tanh activations.

## Step-by-Step Setup & Training

### 1. Data Preparation
Separate your features (`X`) and targets (`Y`). For classification, ensure your targets are one-hot encoded.
```python
import numpy as np

# Example: loading regression data
data = np.loadtxt('../data/Advertising.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3, 4))
split = int(0.8 * len(data))

x_train, y_train = data[:split, :-1], data[:split, -1:]
x_test, y_test = data[split:, :-1], data[split:, -1:]
```

### 2. Network Configuration
Define the architecture using a list of neuron counts and a corresponding list of activation functions for each layer.
```python
import neural_network as nn

act = nn.ActivationFunction
# --- How Many Neurons per Layer? ---
# The "In-Between" Rule: A number between the size of the input layer and output layer.
# The 2/3 Rule: (Number of Inputs * 2/3) + Number of Outputs
# The Funnel Architecture: Decrease the size for subsequent layers.
model_inits = [[512, 256, 10], 
               [act.ReLU, act.ReLU, act.softmax]]
loss_func = nn.LossFunction.cc_loss

model = nn.BasicNeuralNetwork(
    layers_init=model_inits,
    loss_func=loss_func,
    training_set=(x_train, y_train),
    test_set=(x_test, y_test),
    batch=128,                   # Mini-batch size
    leanring_rate=0.01,
    epsilon=0.00001,             # Convergence threshold based on loss smoothing
    epoch_limit=20,              # Max epochs
    iteration_event_trigger=100  # Print progress every 100 iterations
)
```

### 3. Training the Model
Train the network using mini-batch gradient descent. The training stops when the loss stabilizes (based on exponentially weighted average loss and `epsilon`) or when `epoch_limit` is reached.
```python
model.fit_model()
```

### 4. Evaluation and Visualization
Evaluate the network's performance on the test set.
```python
# For MSE loss, returns R-squared value. For classification, returns accuracy/precision.
score = model.evaluate()
print(f"Model Score: {score}%")

# Visualize predictions vs actual targets
model.predict_vs_target()
```

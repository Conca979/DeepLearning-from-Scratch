# Machine Learning from Scratch

A pure-NumPy implementation of core machine learning algorithms — Linear Regression, Logistic Regression, and a fully-connected Neural Network — built from the ground up without scikit-learn or any ML framework.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Datasets](#datasets)
- [Running the Models](#running-the-models)
  - [Linear Regression](#linear-regression)
  - [Logistic Regression](#logistic-regression)
  - [Neural Network](#neural-network)
- [Model Configuration](#model-configuration)
  - [Regression Hyperparameters](#regression-hyperparameters)
  - [Neural Network Hyperparameters](#neural-network-hyperparameters)
- [Key Concepts Implemented](#key-concepts-implemented)

---

## Overview

This project implements machine learning models from scratch using only NumPy, Matplotlib, and Pandas. The goal is educational — to understand exactly what happens under the hood during training, rather than relying on black-box libraries.

**For detailed algorithmic explanations and mathematical breakdowns of the implementations, please refer to the specific module documentation:**
- [Regression (Linear & Logistic) Documentation](./regression/README.md)
- [Neural Network Documentation](./neural_network/README.md)

**What's implemented:**

| Model | Algorithm | Use Case |
|---|---|---|
| `BasicLinearRegression` | Gradient Descent | Continuous value prediction (regression) |
| `BasicLogisticRegression` | Gradient Ascent (log-likelihood) | Binary classification |
| `BasicNeuralNetwork` | Backpropagation | Regression & multi-class classification |

---

## Project Structure

```
MachineLeaning/
├── regression.py          # Linear & Logistic Regression classes
├── neural_network.py      # Neural Network, Activation & Loss functions
├── run.py                 # Entry point for regression models
├── nn_run.py              # Entry point for the neural network
└── data/
    ├── Advertising.csv    # TV/Radio/Newspaper ad spend vs. sales
    ├── Cellphone.csv      # Phone specs vs. price
    ├── Diabetes.csv       # Medical indicators vs. diabetes progression
    └── Iris.csv           # Flower measurements vs. species (classification)
```

> **Note:** `run.py` and `nn_run.py` reference `data/StudentScore.csv` by default, which is excluded from the repository (listed in `.gitignore`). Swap this path to any compatible dataset before running.

---

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- Seaborn
- Pandas

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/Conca979/MachineLearning.git
cd MachineLearning
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install numpy matplotlib seaborn pandas
```

---

## Datasets

The `data/` folder contains four ready-to-use datasets:

| File | Features | Target | Task |
|---|---|---|---|
| `Advertising.csv` | TV, Radio, Newspaper spend | Sales | Linear Regression |
| `Cellphone.csv` | Phone specs (CPU, RAM, camera, etc.) | Price | Linear Regression |
| `Diabetes.csv` | BMI, BP, Cholesterol, LDL | Disease progression score | Linear Regression |
| `Iris.csv` | Sepal/petal length & width | Species | Classification (Neural Network) |

---

## Running the Models

### Linear Regression

Edit `run.py` to point to your dataset and configure hyperparameters, then run:

```bash
python run.py
```

A minimal example using `Advertising.csv`:

```python
from regression import BasicLinearRegression
import numpy as np

data = np.loadtxt('data/Advertising.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4))

model = BasicLinearRegression(
    trainingSet=data,
    testSet=None,
    epsilon=1e-7,
    learningRate=0.0001,
    modelDegree=1,
    iterationLogTrigger=10000
)

model.fitModel()
print(f"R²: {model._modelEvaluaion()[2]:.4f}")
print(f"Iterations: {model.iterationCount}")
model.showModel()          # Scatter plot: predicted vs actual values
model.showModel(graph=1)   # Cost vs. iteration curve
```

---

### Logistic Regression

```python
from regression import BasicLogisticRegression
import numpy as np

# Expects last column to be a binary label (0 or 1)
data = np.loadtxt('data/your_binary_dataset.csv', delimiter=',', skiprows=1)
split = int(len(data) * 0.8)

model = BasicLogisticRegression(
    trainingSet=data[:split],
    testSet=data[split:],
    epsilon=1e-6,
    learningRate=0.001,
    modelDegree=1,
    logEventTrigger=10000
)

model.fitModel()
model.showConfusionMatrix()   # Heatmap with F1 score
model.showCostTrend()         # Log-likelihood over iterations
```

---

### Neural Network

Edit `nn_run.py` to point to your dataset, then run:

```bash
python nn_run.py
```

A minimal example using `Advertising.csv` for regression:

```python
import neural_network as nn
import numpy as np

data = np.loadtxt('data/Advertising.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4))
split = int(0.8 * len(data))

x_train, y_train = data[:split, :-1], data[:split, -1:]
x_test,  y_test  = data[split:, :-1], data[split:, -1:]

act = nn.ActivationFunction
model = nn.BasicNeuralNetwork(
    layers_init=[[8, 1], [act.ReLU, act.identity]],
    loss_func=nn.LossFunction.MSE,
    training_set=(x_train, y_train),
    test_set=(x_test, y_test),
    leanring_rate=0.01,
    epsilon=0.00001,
    iteration_event_trigger=10000
)

model.fit_model()
print(f"Goodness of fit: {model.evaluate()}")
model.predict_vs_target()
```

For **multi-class classification** with the Iris dataset, use `softmax` activation on the output layer and `cc_loss` (categorical cross-entropy) as the loss function. See the commented-out example at the bottom of `nn_run.py`.

---

## Model Configuration

### Regression Hyperparameters

| Parameter | Description | Typical Value |
|---|---|---|
| `learningRate` | Step size for gradient update | `0.0001` – `0.01` |
| `epsilon` | Convergence threshold (relative cost change) | `1e-6` – `1e-7` |
| `modelDegree` | Polynomial degree for feature expansion | `1` (linear) – `5` |
| `iterationLogTrigger` | Print progress every N iterations (`-1` to disable) | `10000` |
| `initWeights` | Custom starting weights; `None` initializes to zeros | `None` |

### Neural Network Hyperparameters

| Parameter | Description | Example |
|---|---|---|
| `layers_init` | `[[neurons_per_layer], [activations]]` | `[[8, 1], [ReLU, identity]]` |
| `loss_func` | `LossFunction.MSE` or `LossFunction.cc_loss` | `LossFunction.MSE` |
| `leanring_rate` | Gradient descent step size | `0.01` |
| `epsilon` | Convergence threshold | `0.00001` |
| `iteration_event_trigger` | Log training progress every N iterations | `10000` |

**Available activation functions:** `identity`, `ReLU`, `sigmoid`, `softmax`, `Tanh`

**Layer sizing heuristics (from code comments):**
- *In-between rule:* choose neuron count between input and output size
- *2/3 rule:* `(n_inputs × 2/3) + n_outputs`
- *Funnel architecture:* decrease neuron count in successive hidden layers

---

## Key Concepts Implemented

- **Z-score standardization** — applied to both features and output to prevent overflow during gradient computation
- **Polynomial feature expansion** — combinatorial feature interactions up to degree `n` via `PolyFeatureTransform`
- **Gradient descent / ascent** — weight updates driven by MSE (linear) or log-likelihood (logistic)
- **Backpropagation** — full forward and backward pass with support for arbitrary layer depth
- **Weight initialization** — He (Kaiming) initialization for ReLU layers; Xavier for others
- **Model evaluation** — R² for regression; confusion matrix, precision, recall, F1, accuracy for classification
- **Model persistence** — `saveModel()` serializes trained weights and hyperparameters to JSON
- **K-fold cross-validation** — scaffolding present in `run.py` (commented out, ready to enable)
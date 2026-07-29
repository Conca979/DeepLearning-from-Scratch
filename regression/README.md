# Regression Algorithms from Scratch

This directory contains pure-NumPy implementations of Linear and Logistic Regression, built from the ground up to demonstrate the mathematical foundations of machine learning.

## Algorithms Explained

### 1. Linear Regression (Gradient Descent)
Linear Regression predicts a continuous target value by fitting a linear equation to the data. 

**Hypothesis:**
$h_\theta(x) = \theta^T x$
Where $\theta$ represents the weights and $x$ represents the features.

**Cost Function (Mean Squared Error - MSE):**
$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$

**Gradient Descent Update Rule:**
To minimize the cost function, we iteratively update the weights:
$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$
$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$
Where $\alpha$ is the learning rate.

### 2. Logistic Regression (Gradient Ascent)
Logistic Regression predicts binary categories (0 or 1) by estimating probabilities using a logistic function.

**Hypothesis (Sigmoid Function):**
$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$

**Cost Function (Log-Likelihood):**
We maximize the log-likelihood (equivalent to minimizing cross-entropy):
$L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$

**Gradient Ascent Update Rule:**
$\theta_j := \theta_j + \alpha \frac{\partial L(\theta)}{\partial \theta_j}$
$\frac{\partial L(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$

### 3. Feature Processing
- **Z-Score Normalization:** Features are scaled to have a mean of 0 and a standard deviation of 1. This prevents overflow during computation and helps gradient descent converge faster.
- **Polynomial Transformation:** Features can be expanded combinatorially up to a specified degree to model non-linear relationships.

## Step-by-Step Setup & Training

### 1. Data Preparation
Your dataset should be a NumPy array where columns are features and the last column is the target variable.
```python
import numpy as np

# Load data (e.g., from CSV)
data = np.loadtxt('../data/Advertising.csv', delimiter=',', skiprows=1, usecols=(1, 2, 3, 4))
```

### 2. Model Initialization
Instantiate the regression model by specifying the dataset and hyperparameters.
```python
from regression import BasicLinearRegression

model = BasicLinearRegression(
    trainingSet=data,
    testSet=None,           # Optional: separate test set array
    epsilon=1e-7,           # Convergence threshold
    learningRate=0.001,     # Step size for gradient updates
    modelDegree=1,          # Set >1 for polynomial regression
    iterationLogTrigger=1000 # Print cost every 1000 iterations
)
```

### 3. Training the Model
Call the `fitModel` method. This will run the gradient descent algorithm until the cost shifts by less than the `epsilon` threshold.
```python
# log argument saves cost history to a file
model.fitModel(log='training_log.txt')
```

### 4. Evaluation and Visualization
After training, evaluate the model's performance and visualize the results.
```python
# For Linear Regression
print(f"R² (Goodness of Fit): {model.goodnessOfFit}")
model.showModel() # Scatter plot of actual vs predicted

# For Logistic Regression
# model.showConfusionMatrix()
# model.showCostTrend()
```

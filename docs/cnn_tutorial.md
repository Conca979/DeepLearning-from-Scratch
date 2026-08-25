# Pure-NumPy CNN — Complete Technical Tutorial (Expanded Edition)

> This document is a fully self-contained tutorial and technical reference for the CNN implemented in [`cnn.py`](file:///c:/Users/admin/projects/deep-learning/neural_network/cnn.py). It begins with beginner-friendly intuition and API reference, followed by every mathematical equation, vectorization trick, and backward-pass derivation proved from first principles.

---

## Table of Contents

1. [Beginner's Introduction to CNNs](#1-beginners-introduction-to-cnns)
2. [API Reference & Code Structure](#2-api-reference--code-structure)
3. [Notation & Conventions](#3-notation--conventions)
4. [Architecture Overview](#4-architecture-overview)
5. [The Training Loop](#5-the-training-loop)
6. [Deep Dive / Technical Details](#6-deep-dive--technical-details)
   - 6.1 [Layer 0 — InputLayer](#61-layer-0--inputlayer)
   - 6.2 [Layer 1 — ConvLayer](#62-layer-1--convlayer)
   - 6.3 [Layer 2 — MaxPoolLayer](#63-layer-2--maxpoollayer)
   - 6.4 [Layer 3 — FlattenLayer](#64-layer-3--flattenlayer)
   - 6.5 [Layer 4 — DenseLayer](#65-layer-4--denselayer)
7. [Activation Functions & Loss Functions](#7-activation-functions--loss-functions)
8. [Dynamic Learning Rate Decay](#8-dynamic-learning-rate-decay)
9. [Full Worked Example](#9-full-worked-example)

---

## 1. Beginner's Introduction to CNNs

Welcome to Convolutional Neural Networks (CNNs)! If you are learning about CNNs from scratch, this section is for you.

At a high level, a CNN is a special type of neural network designed to process data with a grid-like topology, such as an image. Instead of looking at an image pixel by pixel in isolation, a CNN looks at small patches of the image to find patterns, like edges, textures, or shapes.

### The Intuition behind Convolution

Imagine you are looking at a large painting with a small magnifying glass. You can only see a small section of the painting at a time. You slide this magnifying glass across the entire painting, left to right, top to bottom. 

In a CNN, the "magnifying glass" is called a **filter** (or **kernel**). A filter is a small matrix of numbers. As you slide it over the image, you perform a mathematical operation (a dot product) between the filter and the patch of the image it's currently looking at. 

- **Why do this?** Different filters act as feature detectors. One filter might light up (produce a high value) when it sees a vertical edge. Another might light up for a horizontal edge. As we go deeper into the network, filters combine these simple edges to detect complex shapes like eyes or wheels.
- **Key Concepts:**
  - **Stride:** How many pixels you move the magnifying glass each time you slide it.
  - **Padding:** Adding a border of zeros around the original image so the magnifying glass can properly look at the edges of the image.

### The Intuition behind Pooling

After a Convolutional layer, we often apply a **Pooling** layer (specifically Max Pooling). 

Imagine taking a high-resolution photo and scaling it down to a lower resolution. You lose some exact details, but you still know what's in the photo.

Max Pooling takes a small window (e.g., $2 \times 2$ pixels) and only keeps the maximum value from that window, discarding the rest.
- **Why do this?** It reduces the size of the data, making computation faster. More importantly, it creates **spatial invariance**. If a feature (like an eye) moves a pixel to the left or right, the pooling layer will still capture it. It cares *that* the feature exists, not exactly *where* it is.

### The Intuition behind Dense (Fully Connected) Layers

After alternating Convolution and Pooling layers, the network has learned high-level features. We then **flatten** the 2D grids into a 1D list and pass it into a standard neural network layer, called a **Dense** layer.

Here, every neuron connects to every neuron in the previous layer. This layer takes all the detected features ("I see an ear," "I see a nose," "I see fur") and combines them to make the final decision ("It's a cat!").

---

## 2. API Reference & Code Structure

The CNN is implemented entirely in [`cnn.py`](file:///c:/Users/admin/projects/deep-learning/neural_network/cnn.py) using pure NumPy. The design is object-oriented, structured as a linked list of layer objects managed by a central `ConvolutionalNetwork` class.

### `ConvolutionalNetwork`

This is the main class that orchestrates training and inference.

**Initialization:**
```python
model = ConvolutionalNetwork(
    layers_spec: list,          # Defines the architecture (see below)
    training_set: tuple,        # (x_train, y_train) arrays
    loss_func: Callable,        # Loss function (e.g., cc_loss, MSE)
    batch: int,                 # Mini-batch size
    learning_rate: float,       # Initial learning rate
    lr_decay_rate: float,       # Decay factor after each epoch (default: 1.0)
    epsilon: float,             # Convergence threshold (default: 1e-6)
    epoch_limit: int,           # Maximum epochs to train
    test_set: tuple,            # (x_test, y_test) for evaluation
    iteration_event_trigger: int # Logging frequency
)
```

**Core Methods:**
- `fit_model()`: Starts the training loop. It handles forward propagation, backpropagation, weight updates, and early stopping via EMA convergence.
- `predict(x)`: Takes input `x` of shape `(B, C, H, W)` or `(B, H \cdot W)` and returns the network's predictions.
- `evaluate()`: Calculates classification accuracy on the `test_set`.
- `save_weights(path)` / `load_weights(path)`: Serializes/deserializes model weights and Batch Normalization statistics using `.npz` files.

### Layer Specifications (`layers_spec`)

The architecture is defined using a list of tuples passed to `ConvolutionalNetwork`. 

**1. Convolutional Layer**
```python
('conv', n_filters, kernel_size, stride, padding, act_func, use_bn)
```
- `n_filters` (int): Number of feature detectors.
- `kernel_size` (int): Size of the square kernel ($k \times k$).
- `stride` (int): Sliding step size.
- `padding` (int): Zero-padding around the spatial dimensions.
- `act_func` (Callable): Activation function (e.g., ReLU).
- `use_bn` (bool): Whether to apply Batch Normalization before activation.

**2. Max Pooling Layer**
```python
('maxpool', pool_size, stride)
```
- `pool_size` (int): Size of the pooling window ($p \times p$).
- `stride` (int): Step size for pooling.

**3. Flatten Layer**
```python
('flatten',)
```
- No parameters. Reshapes `(B, C, H, W)` into `(B, C \cdot H \cdot W)`.

**4. Dense Layer**
```python
('dense', n_neurons, act_func, use_dropout, drop_rate)
```
- `n_neurons` (int): Number of output neurons.
- `act_func` (Callable): Activation function (e.g., ReLU, softmax).
- `use_dropout` (bool): Whether to use inverted dropout.
- `drop_rate` (float): Fraction of neurons to drop (default 0.5).

---

## 3. Notation & Conventions

| Symbol | Meaning |
|---|---|
| $B$ | Batch size (number of samples in one mini-batch) |
| $N$ | Usually $B \times H \times W$ — total scalar elements per channel in BN |
| $C_{in}$ | Number of input channels (depth) |
| $C_{out}$ / $n_f$ | Number of output channels = number of convolutional filters |
| $H, W$ | Spatial height and width of the input feature map |
| $kH, kW$ | Kernel height and width (square: $kH = kW = k$) |
| $p$ | Zero-padding applied symmetrically to H and W dimensions |
| $s$ | Stride — step size for the sliding kernel/pool window |
| $out\_H, out\_W$ | Output spatial dimensions after convolution or pooling |
| $K$ | Kernel tensor, shape $(n_f, C_{in}, kH, kW)$ |
| $W_{col}$ | Kernel reshaped to $(n_f, C_{in} \cdot kH \cdot kW)$ for matmul |
| $col$ | im2col matrix, shape $(B, C_{in} \cdot kH \cdot kW, out\_H \cdot out\_W)$ |
| $W$ | Dense weight matrix, shape $(n_{neuron}, n_{in})$ |
| $b$ | Bias vector |
| $Z$ | Pre-activation values (linear output before $act\_func$) |
| $A$ | Post-activation output = $act\_func(Z)$ |
| $L$ | Scalar loss value |
| $\frac{\partial L}{\partial X}$ | Partial derivative of loss w.r.t. any tensor $X$ (gradient) |
| $\odot$ | Element-wise (Hadamard) multiplication |
| $@$ | Matrix multiplication (batched where applicable) |
| $X^T$ | Transpose |
| $\mu_c, \sigma^2_c$ | Per-channel batch mean and variance in BN |
| $\hat{x}$ | Standardized value inside BN |
| $\gamma, \beta$ | Learnable BN scale and shift — shape $(1, C, 1, 1)$ |
| $\epsilon_{bn}$ | BN numerical stability constant ($10^{-5}$) |
| $\epsilon_{conv}$ | EWA convergence threshold (early stopping) |
| $\delta_{ij}$ | Kronecker delta: $1$ if $i=j$, else $0$ |

**Tensor index notation:**
- $X[b, c, h, w]$ — element at batch $b$, channel $c$, row $h$, column $w$.
- $X[b, f, n]$ — element at batch $b$, filter $f$, spatial position $n$ (flattened).

---

## 4. Architecture Overview

The LeNet-style MNIST network configures layers as follows:

```
Layer          Type          Config                      Output Shape
─────────────────────────────────────────────────────────────────────────
InputLayer     —             —                           (B,  1, 28, 28)
ConvLayer 1    conv          16 filters, 3×3, p=1, s=1   (B, 16, 28, 28)
               + BN + ReLU
MaxPoolLayer 1 pool          2×2, s=2                    (B, 16, 14, 14)
ConvLayer 2    conv          32 filters, 3×3, p=1, s=1   (B, 32, 14, 14)
               + BN + ReLU
MaxPoolLayer 2 pool          2×2, s=2                    (B, 32,  7,  7)
FlattenLayer   —             32×7×7 = 1568               (B, 1568)
DenseLayer 1   dense         256 neurons, ReLU, drop=0.3 (B, 256)
DenseLayer 2   dense         10 neurons, softmax         (B, 10)
─────────────────────────────────────────────────────────────────────────
Output         ŷ (probs)     10 classes                  (B, 10)
```

**OOP Linked-List Design:**
- Each layer is a separate Python object living inside `ConvolutionalNetwork`.
- `prv_layer` / `next_layer` pointers form a doubly-linked list.
- Forward and backward passes iterate this list.

> [!IMPORTANT]
> **`layer_delta_term` convention:** Every layer writes $\frac{\partial L}{\partial (\text{its own input})}$ into `self.layer_delta_term`. The previous layer reads this to continue the chain rule backward.

---

## 5. The Training Loop

```python
# fit_model() — full pseudocode
for each epoch (1 to epoch_limit):
    shuffle training indices randomly
    for each mini-batch (size B):
        ① _forward_propagation(x_batch)    # compute ŷ
        ② compute_loss(y_batch)            # scalar L
        ③ _compute_delta_term(y_batch)     # backward: all layer_delta_term
        ④ _backward_propagation()          # SGD step: all update_weights()
    lr *= lr_decay_rate                    # epoch-level exponential LR decay
    check EWA convergence → early stop if change ratio < ε_conv
```

**EWA Convergence Check:**
$$smoothed\_loss_t = \beta \cdot smoothed\_loss_{t-1} + (1-\beta) \cdot loss_t \quad [\beta = 0.2]$$
$$ratio = \frac{|smoothed_{t-1} - smoothed_t|}{smoothed_t + 10^{-12}}$$
If $ratio < \epsilon_{conv}$, training stops.

---

## 6. Deep Dive / Technical Details

This section contains the rigorous mathematical proofs and NumPy implementations for all operations inside the CNN layers.

### 6.1 Layer 0 — InputLayer

The InputLayer is a pure data holder. Its only job is to expose `layer_output` in the same interface that all downstream layers expect. No gradients flow into it.

### 6.2 Layer 1 — ConvLayer

The forward chain:
$$x \rightarrow \text{pad} \rightarrow \text{im2col} \rightarrow W_{col} @ col \rightarrow \text{bias} \rightarrow [\text{BN}] \rightarrow act\_func \rightarrow A$$

#### 6.2.1 Kernel Initialization (He)

**He Initialization** sets:
$$W \sim \mathcal{N}(0, \sigma^2) \quad \text{where} \quad \sigma = \sqrt{\frac{2}{\text{fan\_in}}}$$
Where $\text{fan\_in} = C_{in} \cdot kH \cdot kW$. This preserves activation variance through ReLU layers.

#### 6.2.2 im2col — The Full Derivation

A single convolutional output at position $(b, f, oh, ow)$ is:
$$Z[b, f, oh, ow] = \sum_c \sum_r \sum_q K[f, c, r, q] \cdot x_{pad}[b, c, oh \cdot s + r, ow \cdot s + q] + b[f]$$

Naively computing this requires 5 nested loops. **im2col** converts this into a single matrix multiplication. All receptive fields of $x_{pad}$ are stacked as columns of a matrix `col`. 

The equation becomes:
$$Z[b, f, n] = \sum_p K_{flat}[f, p] \cdot col[b, p, n] + b[f]$$
Where $p$ ranges over $C_{in} \cdot kH \cdot kW$ and $n$ over $out\_H \cdot out\_W$. In batched matrix form:
$$Z_{col}[b] = W_{col} @ col[b]$$
Using einsum:
```python
Z_col = np.einsum('fc, bcn -> bfn', W_col, col, optimize=True)
```

#### 6.2.3 Forward Pass — Step by Step

1. **Spatial zero-padding:** Pad `x` with $p$ zeros.
2. **im2col:** Resulting `col` has shape $(B, C_{in} \cdot kH \cdot kW, out\_H \cdot out\_W)$.
3. **Reshape kernel & multiply:**
   $$Z_{col} = W_{col} @ col$$
   Output spatial size:
   $$out\_H = \left\lfloor \frac{H + 2p - kH}{s} \right\rfloor + 1$$
   $$out\_W = \left\lfloor \frac{W + 2p - kW}{s} \right\rfloor + 1$$

#### 6.2.4 Batch Normalization — Forward

Let $N = B \times out\_H \times out\_W$.

1. **Batch statistics per channel c:**
   $$\mu_c = \frac{1}{N} \sum_{b, h, w} Z[b, c, h, w]$$
   $$\sigma^2_c = \frac{1}{N} \sum_{b, h, w} (Z[b, c, h, w] - \mu_c)^2$$
2. **Standardize:**
   $$\hat{x}[b, c, h, w] = \frac{Z[b, c, h, w] - \mu_c}{\sqrt{\sigma^2_c + \epsilon_{bn}}}$$
3. **Scale and shift:**
   $$BN(Z)[b, c, h, w] = \gamma_c \cdot \hat{x}[b, c, h, w] + \beta_c$$

#### 6.2.5 Backward Pass — Conv: Full Chain Rule

Incoming gradient: $incoming = \frac{\partial L}{\partial A}$.

**Step 1 — Through Activation:**
$$\frac{\partial L}{\partial Z}[b, f, h, w] = \frac{\partial L}{\partial A}[b, f, h, w] \cdot act\_func'(Z[b, f, h, w])$$

**Step 2 — Kernel Gradient $dK$:**
$$\frac{\partial L}{\partial W_{col}}[f, c] = \sum_b \sum_n \left( \frac{\partial L}{\partial Z_{col}}[b, f, n] \right) \cdot col[b, c, n]$$
$$dW_{col} = dZ_{col} @ col^T$$

**Step 3 — Gradient w.r.t. Input ($col2im$):**
$$\frac{\partial L}{\partial col}[b, :, n] = W_{col}^T \cdot \frac{\partial L}{\partial Z_{col}}[b, :, n]$$
Scattered back to the input grid using `np.add.at` to properly accumulate overlapping gradients.

#### 6.2.6 Batch Normalization — Backward

Given $d\_out = \frac{\partial L}{\partial BN_{out}}$:
$$\frac{\partial L}{\partial \gamma_c} = \sum_{b,h,w} d\_out \cdot \hat{x}$$
$$\frac{\partial L}{\partial \beta_c} = \sum_{b,h,w} d\_out$$

For the input $Z$:
$$d\hat{x} = d\_out \cdot \gamma$$
$$d\sigma^2 = \sum d\hat{x} \cdot (Z - \mu) \cdot \left(-\frac{1}{2}\right) \cdot (\sigma^2+\epsilon_{bn})^{-3/2}$$
$$d\mu = \sum d\hat{x} \cdot \frac{-1}{\sqrt{\sigma^2+\epsilon_{bn}}} + d\sigma^2 \cdot \frac{-2}{N} \sum (Z - \mu)$$
$$dZ = d\hat{x} \cdot \frac{1}{\sqrt{\sigma^2+\epsilon_{bn}}} + d\sigma^2 \cdot \frac{2(Z - \mu)}{N} + \frac{d\mu}{N}$$

### 6.3 Layer 2 — MaxPoolLayer

- **Forward:** Creates a 6D strided view of shape $(B, C, out\_H, out\_W, p, p)$ using `np.lib.stride_tricks.as_strided`, allowing zero-copy max computation.
- **Backward:** Routes gradient only to the positions that produced the max value (stored via a boolean mask).

### 6.4 Layer 3 — FlattenLayer

- **Forward:** `x.reshape(x.shape[0], -1)`
- **Backward:** Reverses reshape `next_layer.layer_delta_term.reshape(self.conv_shape)`

### 6.5 Layer 4 — DenseLayer

#### 6.5.1 Forward Pass & Inverted Dropout

$$Z = x @ W^T + b$$

**Inverted Dropout:** To keep expected values consistent between training and testing, surviving neurons are divided by $(1 - drop\_rate)$ during training.
$$\mathbb{E}[A_{masked}] = \frac{A_j \cdot (1-p)}{1-p} = A_j$$

#### 6.5.2 Backward Pass — Output Layer (Softmax + CCE)

For one-hot target $y$:
$$L = -\frac{1}{B} \sum_b \log(\hat{y}[b, \text{true\_class\_b}])$$
$$\frac{\partial L}{\partial \hat{y}[b,c]} = -\frac{1}{B} \frac{y[b,c]}{\hat{y}[b,c]}$$

Softmax Jacobian $J$:
$$\frac{\partial S_i}{\partial Z_j} = S_i \cdot (\delta_{ij} - S_j)$$

$$\delta = \frac{\partial L}{\partial Z} = \frac{\partial L}{\partial \hat{y}} @ J$$
$$dW = \delta^T @ x$$
$$db = \sum_b \delta$$
$$layer\_delta\_term = \delta @ W$$

#### 6.5.3 Backward Pass — Hidden Dense Layer

Apply the dropout mask to incoming gradients (since dropped neurons shouldn't learn).
$$\delta = act\_func'(Z) \odot (incoming \odot mask)$$
$$dW = \delta^T @ x_{input}$$
$$layer\_delta\_term = \delta @ W$$

---

## 7. Activation Functions & Loss Functions

- **ReLU:** $f(z) = \max(0, z)$. Derivative is $1$ for $z>0$, else $0$.
- **Softmax:** $S_i = \frac{\exp(Z_i - \max(Z))}{\sum_j \exp(Z_j - \max(Z))}$. (Subtracted max for numerical stability).
- **Categorical Cross Entropy:** $L = -\frac{1}{B} \sum_{b,c} y[b,c] \log(\hat{y}[b,c] + 10^{-15})$.

---

## 8. Dynamic Learning Rate Decay

$$lr_{epoch} = lr_0 \cdot r^{epoch}$$
Where $r$ is `lr_decay_rate`. This exponential decay allows coarse exploration early and fine-grained convergence later.

---

## 9. Full Worked Example

We trace **one complete forward + backward pass** for a micro-network:

Setup:
- $B = 1$
- $C_{in} = 1$
- $H = W = 4$
- $n_f = 1$
- $kH = kW = 2$
- $stride = 1, padding = 0$
- No BN, ReLU, Dense output (1 neuron) with MSE loss.

Output size: $out\_H = out\_W = (4-2)/1 + 1 = 3$.

### A. Forward Pass

**Input x, shape $(1, 1, 4, 4)$:**
```
x[0,0] =
  1  2  3  4
  5  6  7  8
  9 10 11 12
 13 14 15 16
```
**im2col - col, shape $(1, 4, 9)$:**
```
col[0] (shape 4×9):
           (0,0)(0,1)(0,2)(1,0)(1,1)(1,2)(2,0)(2,1)(2,2)
p=0:        1   2   3   5   6   7   9  10  11
p=1:        2   3   4   6   7   8  10  11  12
p=2:        5   6   7   9  10  11  13  14  15
p=3:        6   7   8  10  11  12  14  15  16
```
**Kernel K (shape $1, 1, 2, 2$) & $W_{col}$ (shape $1, 4$):**
$$K[0,0] = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix} \implies W_{col} = [1, -1, 0, 1]$$

**Convolution via matmul:**
$$Z_{col}[0, 0, :] = W_{col} @ col[0] = [5, 6, 7, 9, 10, 11, 13, 14, 15]$$
$$Z = \begin{bmatrix} 5 & 6 & 7 \\ 9 & 10 & 11 \\ 13 & 14 & 15 \end{bmatrix}$$
After ReLU ($A = Z$) and Flatten ($A_{flat} = [5, 6, 7, 9, 10, 11, 13, 14, 15]$).

**DenseLayer (9 inputs -> 1 output):**
$$W_{dense} = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], \quad b_{dense} = 0$$
$$\hat{y} = Z_{dense} = A_{flat} @ W_{dense}^T + b = 9.0$$

**Loss (MSE), target $y = 10.0$:**
$$L = \frac{1}{2 \cdot 1} \cdot (9.0 - 10.0)^2 = 0.5$$

### B. Backward Pass

**MSE derivative:**
$$\frac{\partial L}{\partial \hat{y}} = \frac{1}{1} \cdot (\hat{y} - y) = 9.0 - 10.0 = -1.0$$

**DenseLayer backward:**
$$delta_{dense} = \frac{\partial L}{\partial \hat{y}} \cdot f'(Z) = -1.0 \cdot 1 = -1.0$$
$$dW_{dense} = delta_{dense}^T @ A_{flat} = [-5, -6, -7, -9, -10, -11, -13, -14, -15]$$
$$layer\_delta\_term_{dense} = delta_{dense} @ W_{dense} = [-0.1, -0.1, \dots, -0.1]$$

**FlattenLayer backward:** Reshape $(1, 9) \rightarrow (1, 1, 3, 3)$.
$$d\_conv = \begin{bmatrix} -0.1 & -0.1 & -0.1 \\ -0.1 & -0.1 & -0.1 \\ -0.1 & -0.1 & -0.1 \end{bmatrix}$$

**ConvLayer backward (through ReLU):**
$$dZ = (Z > 0) \odot d\_conv = 1 \odot d\_conv = d\_conv$$

**Kernel gradient:**
$$dZ_{col} = [[-0.1, \dots, -0.1]]$$
$$dW_{col} = dZ_{col} @ col^T = [-5.4, -6.3, -9.0, -9.9]$$
$$dK = \begin{bmatrix} -5.4 & -6.3 \\ -9.0 & -9.9 \end{bmatrix}$$

**Gradient to input via col2im:**
$$d\_col = W_{col}^T @ dZ_{col}$$
Each row $p$ of $d\_col[0]$ is $W_{col}[0,p] \cdot (-0.1)$. 
For example, pixel $(0,0)$ only appears in window $(0,0)$, so $dx[0,0,0,0] = -0.1$.
Pixel $(0,1)$ appears in windows $(0,0)$ and $(0,1)$, so $dx[0,0,0,1] = 0.1 - 0.1 = 0.0$.
This properly accumulates gradients via `np.add.at`.

---

*End of tutorial. Every formula maps 1-to-1 with source code in [`cnn.py`](file:///c:/Users/admin/projects/deep-learning/neural_network/cnn.py).*

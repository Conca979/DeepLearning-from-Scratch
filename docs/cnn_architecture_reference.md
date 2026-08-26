# CNN Architecture — Complete Reference Guide

> **Codebase**: `src/deep_learning/` (pure-NumPy implementation)  
> **Entry point**: `from deep_learning import Network, Conv2D, MaxPool2D, Flatten, Dense, InputLayer, ActivationFunction, LossFunction`

---

## Table of Contents

1. [Shape Notation](#1-shape-notation)
2. [Module — Base Class](#2-module--base-class)
3. [InputLayer (CNN)](#3-inputlayer-cnn)
4. [Conv2D](#4-conv2d)
   - [im2col helpers](#41-im2col-helpers)
   - [Batch Normalization](#42-batch-normalization)
5. [MaxPool2D](#5-maxpool2d)
6. [Flatten](#6-flatten)
7. [Dense](#7-dense)
8. [InputLayer (MLP)](#8-inputlayer-mlp)
9. [ActivationFunction](#9-activationfunction)
10. [LossFunction](#10-lossfunction)
11. [Network](#11-network)
12. [End-to-End Data Flow](#12-end-to-end-data-flow)
13. [Full Example — LeNet on MNIST](#13-full-example--lenet-on-mnist)

---

## 1. Shape Notation

| Symbol | Meaning |
|--------|---------|
| `N` / `batch` | Number of samples in one mini-batch |
| `C_in` | Number of input channels |
| `C_out` / `n_filters` | Number of output channels (filters) |
| `H`, `W` | Spatial height and width of a feature map |
| `kH`, `kW` | Kernel (filter) height and width |
| `p` | Padding (pixels added on each side) |
| `s` | Stride |
| `out_H`, `out_W` | Spatial size after convolution / pooling |
| `F` | Number of features in a flattened vector |

**Output spatial size formula** (same for Conv2D and MaxPool2D):

```
out_H = floor((H + 2p - kH) / s) + 1
out_W = floor((W + 2p - kW) / s) + 1
```

---

## 2. `Module` — Base Class

**File**: `src/deep_learning/nn/core.py`

The abstract base for every trainable layer. All layers inherit from this class and must override `forward` and `compute_delta_term`.

### Attributes

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `prv_layer` | `Module \| None` | — | Reference to the previous layer in the graph |
| `next_layer` | `Module \| None` | — | Reference to the next layer in the graph |
| `layer_output` | `np.ndarray \| None` | `(batch, ...)` | Activation output produced by `forward()` |
| `layer_delta_term` | `np.ndarray \| None` | `(batch, ...)` | `dL/d(layer input)` — gradient flowing **into** this layer, computed by `compute_delta_term()` |
| `weights` | `np.ndarray \| None` | varies per layer | Learnable weight matrix |
| `biases` | `np.ndarray \| None` | varies per layer | Learnable bias vector |

### Methods

#### `__init__(self)`
Initialises every attribute to `None`. Subclasses call `super().__init__()`.

---

#### `init(self) -> None`
Called **once** after layers are linked by `Network._init_network()`.  
Override to allocate weights, biases, and cached index arrays.

- **Input**: None (reads `self.prv_layer.out_shape` internally)
- **Output**: None (sets `self.weights`, `self.biases`, etc.)

---

#### `forward(self, predict_input=None) -> np.ndarray`
Computes the layer's output and stores it in `self.layer_output`.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `predict_input` | `np.ndarray \| None` | `(batch, ...)` | Only used by the first layer; all others read `self.prv_layer.layer_output` |

- **Returns**: `np.ndarray` — same as `self.layer_output`
- **Raises**: `NotImplementedError` if not overridden

---

#### `compute_delta_term(self, network, targets) -> None`
Backward pass. Computes `self.layer_delta_term` = `dL/d(layer input)`, and typically also computes weight gradients `dW`, `db`.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `network` | `Network` | — | Reference to the training network (to call `compute_loss`) |
| `targets` | `np.ndarray` | `(batch, n_classes)` | Ground-truth one-hot labels for the current batch |

- **Returns**: `None`
- **Raises**: `NotImplementedError` if not overridden

---

#### `update_weights(self, lr) -> None`
Applies the gradient descent update: `param -= lr * grad`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lr` | `float` | Learning rate |

- **Default**: `pass` (no-op for layers with no parameters)

---

## 3. `InputLayer` (CNN)

**File**: `src/deep_learning/nn/modules/conv.py`

Feeds a 4-D image tensor into the convolutional network. This is the **entry node** of any CNN built with this library.

### `__init__(self, x_train)`

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `x_train` | `np.ndarray` | `(N, C, H, W)` | Full training set; used to infer `out_shape` |

### Attributes

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `layer_input` | `np.ndarray` | `(N, C, H, W)` | Reference to the training data |
| `out_shape` | `tuple` | `(C, H, W)` | Spatial dimensions (without batch), read by the next Conv2D during `init()` |
| `layer_output` | `np.ndarray \| None` | `(batch, C, H, W)` | Active mini-batch being processed |
| `next_layer` | `Module \| None` | — | Next layer in chain |

### Methods

#### `forward(self, x=None) -> None`

| Input | Shape | Description |
|-------|-------|-------------|
| `x` | `(batch, C, H, W)` or `None` | When given, stores `x` as output; otherwise uses `self.layer_input` |

- **Output** (`self.layer_output`): `(batch, C, H, W)` — the raw pixel tensor for the current batch.

---

## 4. `Conv2D`

**File**: `src/deep_learning/nn/modules/conv.py`

2-D convolutional layer implementing the pipeline:

```
input → pad → im2col → matmul → [BatchNorm] → Activation → output
```

### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_filters` | `int` | required | Number of convolutional filters = `C_out` |
| `kernel_size` | `int` | required | Square kernel side length `kH = kW = kernel_size` |
| `act_func` | `Callable` | required | Activation function (e.g. `ActivationFunction.ReLU`) |
| `stride` | `int` | `1` | Step size when sliding the kernel |
| `padding` | `int` | `0` | Zero-padding added to each spatial side |
| `use_bn` | `bool` | `True` | Whether to apply Batch Normalization after convolution |
| `bn_momentum` | `float` | `0.9` | EMA momentum for running mean/variance during inference |
| `prv_layer` | `Module \| None` | `None` | Injected automatically by `Network._init_network()` |
| `next_layer` | `Module \| None` | `None` | Injected automatically by `Network._init_network()` |

### Attributes (set by `init()`)

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `kernels` | `np.ndarray` | `(n_filters, C_in, kH, kW)` | Learnable filter weights, He-initialised |
| `biases` | `np.ndarray` | `(n_filters,)` | Per-filter bias terms, zero-initialised |
| `out_H`, `out_W` | `int` | scalar | Spatial dimensions of the output feature maps |
| `out_shape` | `tuple` | `(n_filters, out_H, out_W)` | Read by the next layer during its `init()` |
| `_k` | `np.ndarray` | `(C_in*kH*kW,)` | Pre-computed channel index for im2col |
| `_i` | `np.ndarray` | `(C_in*kH*kW, out_H)` | Pre-computed row index for im2col |
| `_j` | `np.ndarray` | `(C_in*kH*kW, out_W)` | Pre-computed col index for im2col |
| `col` | `np.ndarray` | `(batch, C_in*kH*kW, out_H*out_W)` | im2col matrix cached during `forward()` for use in `compute_delta_term()` |
| `x_shape_cache` | `tuple` | `(batch, C_in, H, W)` | Original input shape saved for `col2im` in backward |
| `Z` | `np.ndarray` | `(batch, n_filters, out_H, out_W)` | Pre-activation values (after BN, if used), saved for activation derivative |
| `layer_output` | `np.ndarray` | `(batch, n_filters, out_H, out_W)` | Post-activation feature maps |
| `layer_delta_term` | `np.ndarray` | `(batch, C_in, H, W)` | `dL/d(input)`, computed in backward pass |
| `dW` | `np.ndarray` | `(n_filters, C_in, kH, kW)` | Kernel gradient |
| `db` | `np.ndarray` | `(n_filters,)` | Bias gradient |

**Batch Norm attributes** (only when `use_bn=True`):

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `bn_gamma` | `np.ndarray` | `(1, n_filters, 1, 1)` | Learnable scale parameter γ |
| `bn_beta` | `np.ndarray` | `(1, n_filters, 1, 1)` | Learnable shift parameter β |
| `bn_run_mean` | `np.ndarray` | `(1, n_filters, 1, 1)` | Running mean (EMA), used at inference |
| `bn_run_var` | `np.ndarray` | `(1, n_filters, 1, 1)` | Running variance (EMA), used at inference |
| `Z_pre_bn` | `np.ndarray` | `(batch, n_filters, out_H, out_W)` | Raw conv output before BN (saved for backward) |
| `x_hat` | `np.ndarray` | `(batch, n_filters, out_H, out_W)` | Normalised values x̂ (saved for backward) |
| `bn_mean` | `np.ndarray` | `(1, n_filters, 1, 1)` | Batch mean computed during training |
| `bn_var` | `np.ndarray` | `(1, n_filters, 1, 1)` | Batch variance computed during training |
| `d_gamma` | `np.ndarray` | `(1, n_filters, 1, 1)` | Gradient w.r.t. γ |
| `d_beta` | `np.ndarray` | `(1, n_filters, 1, 1)` | Gradient w.r.t. β |

### Methods

#### `init(self) -> None`

Reads `self.prv_layer.out_shape` → `(C_in, H_in, W_in)`, computes output dimensions, allocates `kernels`, `biases`, pre-computes im2col indices, and allocates BN parameters.

**Weight initialisation** (He / Kaiming):
```
std = sqrt(2 / (C_in * kH * kW))
kernels ~ N(0, std²)     shape: (n_filters, C_in, kH, kW)
biases  = 0              shape: (n_filters,)
```

---

#### `forward(self) -> None`

**Full data flow**:

```
x           (batch, C_in, H, W)
  │  pad
x_pad       (batch, C_in, H+2p, W+2p)
  │  im2col
col         (batch, C_in*kH*kW, out_H*out_W)
  │  W_col @ col + b      W_col: (n_filters, C_in*kH*kW)
Z_col       (batch, n_filters, out_H*out_W)
  │  reshape
Z_raw       (batch, n_filters, out_H, out_W)
  │  [BatchNorm]
Z           (batch, n_filters, out_H, out_W)   ← saved as self.Z
  │  act_func(Z)
layer_output (batch, n_filters, out_H, out_W)
```

> **Key tensors and their shapes:**
>
> | Tensor | Shape | Meaning |
> |--------|-------|---------|
> | `x` | `(batch, C_in, H, W)` | Input feature maps |
> | `W_col` | `(n_filters, C_in*kH*kW)` | Kernels reshaped for matmul |
> | `col` | `(batch, C_in*kH*kW, out_H*out_W)` | im2col matrix |
> | `Z` | `(batch, n_filters, out_H, out_W)` | Pre-activation (after BN) |
> | `layer_output` | `(batch, n_filters, out_H, out_W)` | Post-activation |

---

#### `_bn_forward(self, Z) -> np.ndarray`

Spatial Batch Normalisation — normalises over axes `(batch, H, W)` **per channel**.

| Step | Formula | Shape |
|------|---------|-------|
| Compute mean | `μ = mean(Z, axes=[0,2,3])` | `(1, C, 1, 1)` |
| Compute var | `σ² = var(Z, axes=[0,2,3])` | `(1, C, 1, 1)` |
| Normalise | `x̂ = (Z - μ) / √(σ²+ε)` | `(batch, C, H, W)` |
| Scale & shift | `γ·x̂ + β` | `(batch, C, H, W)` |
| Update EMA | `run_mean = α·run_mean + (1-α)·μ` | — |

- **Input** `Z`: `(batch, n_filters, out_H, out_W)`
- **Output**: `(batch, n_filters, out_H, out_W)` — normalised feature maps

---

#### `compute_delta_term(self, network, targets) -> None`

Four-step backward pass:

| Step | Computes | Formula |
|------|----------|---------|
| ① Through activation | `dZ` | `act'(self.Z) ⊙ incoming` |
| ② Through BN | `dZ_raw` | `_bn_backward(dZ)` (if `use_bn`) |
| ③ Kernel grad | `dW`, `db` | `dW_col = einsum('bfn,bcn->fc', dZ_col, col)` |
| ④ Input grad | `layer_delta_term` | `col2im(W_colᵀ @ dZ_col)` |

> **Shape summary:**
>
> | Variable | Shape | Description |
> |----------|-------|-------------|
> | `incoming` | `(batch, n_filters, out_H, out_W)` | `dL/d(this layer's output)` from next layer |
> | `dZ` | `(batch, n_filters, out_H, out_W)` | `dL/dZ` — gradient through activation |
> | `dW` | `(n_filters, C_in, kH, kW)` | Kernel gradient |
> | `db` | `(n_filters,)` | Bias gradient |
> | `layer_delta_term` | `(batch, C_in, H, W)` | `dL/d(input)` — passed to prev layer |

---

#### `_bn_backward(self, d_out) -> np.ndarray`

Full derivation of the spatial BN backward pass. Let `N = batch * H * W`.

```
dx̂   = d_out · γ
dvar  = Σ[ dx̂ · (Z - μ) · (-½) · (σ²+ε)^{-3/2} ]
dμ    = Σ[ dx̂ · (-1/σ) ] + dvar · (-2/N) · Σ(Z - μ)
dZ    = dx̂/σ + dvar · 2(Z - μ)/N + dμ/N
```

| Input | Shape | Description |
|-------|-------|-------------|
| `d_out` | `(batch, C, H, W)` | `dL/d(BN output)` |

- **Output**: `(batch, C, H, W)` — `dL/d(conv output = BN input)`

Side effects: sets `d_gamma` `(1,C,1,1)` and `d_beta` `(1,C,1,1)`.

---

#### `update_weights(self, lr) -> None`

```python
kernels  -= lr * dW        # (n_filters, C_in, kH, kW)
biases   -= lr * db        # (n_filters,)
bn_gamma -= lr * d_gamma   # (1, n_filters, 1, 1)  [if use_bn]
bn_beta  -= lr * d_beta    # (1, n_filters, 1, 1)  [if use_bn]
```

---

### 4.1 im2col Helpers

These three module-level functions are **implementation details** used exclusively by `Conv2D`.

#### `_im2col_indices(C_in, kH, kW, out_H, out_W, stride)`

Pre-computes the three index arrays used to gather receptive-field patches from the input tensor. Called **once** in `Conv2D.init()`.

| Return | Shape | Meaning |
|--------|-------|---------|
| `k` | `(C_in*kH*kW,)` | Channel index for each kernel element |
| `i` | `(C_in*kH*kW, out_H)` | Row index into the padded input |
| `j` | `(C_in*kH*kW, out_W)` | Column index into the padded input |

---

#### `_im2col(x_pad, k, i, j, out_H, out_W) -> np.ndarray`

Gathers all receptive fields via advanced indexing — **zero extra loops**.

| Input | Shape | Description |
|-------|-------|-------------|
| `x_pad` | `(batch, C_in, H_pad, W_pad)` | Zero-padded input |
| `k, i, j` | pre-computed | Index arrays from `_im2col_indices` |

- **Output** `col`: `(batch, C_in*kH*kW, out_H*out_W)` — column matrix where each column is one flattened patch.

---

#### `_col2im(col, x_shape, k, i, j, out_H, out_W, padding) -> np.ndarray`

Transpose of im2col — scatters gradients back to the spatial input grid.  
Overlapping windows are accumulated with `np.add.at`.

| Input | Shape | Description |
|-------|-------|-------------|
| `col` | `(batch, C_in*kH*kW, out_H*out_W)` | Gradient in column form |
| `x_shape` | `(batch, C_in, H, W)` | Target shape (original un-padded) |

- **Output** `dx`: `(batch, C_in, H, W)` — gradient w.r.t. the un-padded input.

---

### 4.2 Batch Normalization

Batch Norm normalises each **channel** independently over the batch and spatial dimensions, then applies learnable scale (γ) and shift (β).

| Mode | Statistics used |
|------|----------------|
| Training (`network.training = True`) | Batch mean & variance; updates running stats via EMA |
| Inference (`network.training = False`) | Running mean & variance only (no data leakage) |

**Why Spatial BN?** For conv layers, feature maps are `(batch, C, H, W)`. Normalising over axes `(0, 2, 3)` treats every spatial position of the same channel as a sample — consistent with the standard "Batch Norm for CNNs" formulation.

---

## 5. `MaxPool2D`

**File**: `src/deep_learning/nn/modules/pooling.py`

Applies 2-D max-pooling with a non-overlapping or stride-based window. **No learnable parameters.**

### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pool_size` | `int` | required | Square pooling window side `p` |
| `stride` | `int` | required | Step between pooling windows |
| `prv_layer` | `Module \| None` | `None` | Injected by network |
| `next_layer` | `Module \| None` | `None` | Injected by network |

### Attributes (set by `init()`)

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `out_H`, `out_W` | `int` | scalar | Output spatial size `= (H - p) // s + 1` |
| `out_shape` | `tuple` | `(C, out_H, out_W)` | Read by the next layer |
| `mask` | `np.ndarray` | `(batch, C, out_H, out_W, p, p)` | Boolean mask — `True` where max occurred; saved during forward for backward |
| `x_shape` | `tuple` | `(batch, C, H, W)` | Cached original input shape for gradient allocation |
| `layer_output` | `np.ndarray` | `(batch, C, out_H, out_W)` | Max-pooled feature maps |
| `layer_delta_term` | `np.ndarray` | `(batch, C, H, W)` | `dL/d(input)` — only max positions receive gradient |

### Methods

#### `init(self) -> None`

Reads `(C, H, W)` from `prv_layer.out_shape`, computes `out_H`, `out_W`, sets `out_shape`.

---

#### `forward(self) -> None`

Uses `np.lib.stride_tricks.as_strided` to build a **zero-copy** view:

```
input       (batch, C, H, W)
  │  as_strided
windows     (batch, C, out_H, out_W, p, p)   — overlapping windows
  │  .max(axis=(4,5))
layer_output (batch, C, out_H, out_W)
```

`mask` records which position held the maximum (for tied values, all tied positions share the gradient equally).

---

#### `compute_delta_term(self, network, targets) -> None`

Scatters incoming gradient back to only the max positions:

```
incoming      (batch, C, out_H, out_W)
mask          (batch, C, out_H, out_W, p, p)   boolean
count         number of tied maxima per window
d_windows     incoming / count,  broadcast over (p, p)
layer_delta_term  (batch, C, H, W)   accumulated via sliding-window loop
```

---

#### `update_weights(self, lr) -> None`

No-op (`pass`) — no parameters to update.

---

## 6. `Flatten`

**File**: `src/deep_learning/nn/modules/pooling.py`

Converts a 4-D feature map tensor to a 2-D matrix so it can be fed into `Dense` layers. **No learnable parameters.**

### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prv_layer` | `Module \| None` | `None` | Injected by network |
| `next_layer` | `Module \| None` | `None` | Injected by network |

### Attributes

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `out_shape` | `tuple` | `(C*H*W,)` | Flat dimension size |
| `n_neuron` | `int` | `C*H*W` | Alias read by `Dense.init()` |
| `conv_shape` | `tuple` | `(batch, C, H, W)` | Saved during forward for reshaping in backward |
| `layer_output` | `np.ndarray` | `(batch, C*H*W)` | Flattened feature vector |
| `layer_delta_term` | `np.ndarray` | `(batch, C, H, W)` | `dL/d(input)` reshaped from Dense's delta |

### Methods

#### `init(self) -> None`

```python
flat_size = C * H * W
out_shape = (flat_size,)
n_neuron  = flat_size
```

---

#### `forward(self) -> None`

| Input | Shape | Output | Shape |
|-------|-------|--------|-------|
| `x` from `prv_layer` | `(batch, C, H, W)` | `layer_output` | `(batch, C*H*W)` |

Saves `conv_shape = x.shape` for use in `compute_delta_term`.

---

#### `compute_delta_term(self, network, targets) -> None`

Reshapes the incoming flat gradient back to the conv shape:

```python
layer_delta_term = next_layer.layer_delta_term.reshape(conv_shape)
```

| Input | Shape | Output | Shape |
|-------|-------|--------|-------|
| `next_layer.layer_delta_term` | `(batch, C*H*W)` | `layer_delta_term` | `(batch, C, H, W)` |

---

## 7. `Dense`

**File**: `src/deep_learning/nn/modules/linear.py`

Fully-connected (linear) layer with optional Dropout regularisation.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_neuron` | `int` | required | Number of output neurons |
| `act_func` | `Callable` | required | Activation function |
| `use_dropout` | `bool` | `False` | Whether to apply inverted dropout during training |
| `drop_rate` | `float` | `0.3` | Fraction of neurons dropped, in [0, 1) |
| `weights_initialization` | `str` | `"He"` | `"He"` → `std = √(2/n_in)`;  anything else → `std = √(1/n_in)` (Xavier-like) |

### Attributes (set by `init()`)

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `weights` | `np.ndarray` | `(n_neuron, n_in)` | Weight matrix; `n_in` inferred from previous layer |
| `biases` | `np.ndarray` | `(n_neuron,)` | Bias vector, zero-initialised |
| `Z` | `np.ndarray` | `(batch, n_neuron)` | Pre-activation values `= inp @ W^T + b` |
| `layer_output` | `np.ndarray` | `(batch, n_neuron)` | Post-activation (possibly with dropout applied) |
| `layer_delta_term` | `np.ndarray` | `(batch, n_neuron)` | `dL/dZ` for this layer |
| `drop_mask` | `np.ndarray` | `(batch, n_neuron)` | Bernoulli 0/1 mask (only when dropout active) |
| `dW` | `np.ndarray` | `(n_neuron, n_in)` | Weight gradient |
| `db` | `np.ndarray` | `(n_neuron,)` | Bias gradient |
| `out_shape` | `tuple` | `(n_neuron,)` | Read by downstream layers |

### Methods

#### `init(self) -> None`

Determines `n_in` from the previous layer:

| Previous layer has | `n_in` |
|--------------------|--------|
| `out_shape` attribute | `prod(out_shape)` |
| `n_neuron` attribute | `n_neuron` |
| else | `layer_output.shape[-1]` |

Weight init (He):
```
W ~ N(0, 2/n_in)    shape: (n_neuron, n_in)
b = 0               shape: (n_neuron,)
```

---

#### `forward(self, predict_input=None) -> np.ndarray`

```
inp           (batch, n_in)
  │  inp @ W^T + b
Z             (batch, n_neuron)
  │  act_func(Z)
out           (batch, n_neuron)
  │  [dropout: out * mask / (1 - drop_rate)]
layer_output  (batch, n_neuron)
```

**Inverted Dropout**: during training, neurons are randomly zeroed and remaining values are scaled up by `1/(1-drop_rate)` to preserve the expected magnitude at test time.

---

#### `compute_delta_term(self, network, targets) -> None`

Two cases depending on whether this is the **output layer** or a **hidden layer**:

**Output layer** (`self.next_layer is None`):

| Activation | Delta formula |
|------------|--------------|
| `softmax` | `(dL/dA) @ J_softmax` where `J` is the full Jacobian `(batch, C, C)` |
| other | `act'(Z) x dL/dA` |

**Hidden layer** (`self.next_layer` exists):

```
incoming  = next_layer.layer_delta_term @ next_layer.weights
delta     = act'(Z) x incoming           shape: (batch, n_neuron)
```

After computing `delta`:
```
dW = delta^T @ inp     shape: (n_neuron, n_in)
db = sum(delta, axis=0)  shape: (n_neuron,)
```

---

#### `update_weights(self, lr) -> None`

```python
weights -= lr * dW     # (n_neuron, n_in)
biases  -= lr * db     # (n_neuron,)
```

---

## 8. `InputLayer` (MLP)

**File**: `src/deep_learning/nn/modules/linear.py`

The entry node for **MLP-only** (non-CNN) networks. Works with 2-D data `(N, features)`.

### `__init__` Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `layer_input` | `np.ndarray` | `(N, F)` | Full training feature matrix |

### Attributes

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `layer_input` | `np.ndarray` | `(N, F)` | Training data reference |
| `out_shape` | `tuple` | `(F,)` or product of all dims | Spatial shape for downstream `init()` |
| `n_neuron` | `int` | `F` | Number of input features |
| `layer_output` | `np.ndarray` | `(batch, F)` | Current mini-batch |

---

## 9. `ActivationFunction`

**File**: `src/deep_learning/nn/functional.py`

All methods are `@staticmethod`. They accept a `derived=False` flag — when `True`, they return the derivative (used in backprop).

### Common Signature

```python
ActivationFunction.<name>(Z: np.ndarray, derived: bool = False) -> np.ndarray
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `Z` | `np.ndarray` | any | Pre-activation values |
| `derived` | `bool` | — | `False` → activation value; `True` → derivative |

### Available Functions

| Method | Formula (forward) | Derivative | Notes |
|--------|-------------------|------------|-------|
| `identity(Z)` | `Z` | `1` | Pass-through; used for regression output |
| `ReLU(Z)` | `max(0, Z)` | `1 if Z>0 else 0` | Recommended for hidden conv/dense layers |
| `sigmoid(Z)` | `1 / (1 + e^{-Z})` | `σ(Z)·(1−σ(Z))` | Clips Z to [-500, 500] to prevent overflow |
| `softmax(Z)` | `e^Z / Σe^Z` | Full Jacobian | Output shape `(batch, C, C)` when `derived=True` |
| `Tanh(Z)` | `tanh(Z)` | `1 - tanh²(Z)` | Symmetric, zero-centred |

**Softmax Jacobian** (when `derived=True`):

```
J[b, i, j] = s_i · (δ_{ij} - s_j)
```

- **Output shape**: `(batch, n_classes, n_classes)`

---

## 10. `LossFunction`

**File**: `src/deep_learning/nn/functional.py`

All methods are `@staticmethod`. The `derived=True` mode returns `dL/d(predicted_values)`.

### Common Signature

```python
LossFunction.<name>(predicted_values, targets, derived=False)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `predicted_values` | `np.ndarray` | `(batch, n_classes)` | Network's final output probabilities |
| `targets` | `np.ndarray` | `(batch, n_classes)` | One-hot ground-truth labels |
| `derived` | `bool` | — | `False` → scalar loss; `True` → gradient tensor |

### `cc_loss` — Categorical Cross-Entropy

| Mode | Formula | Output shape |
|------|---------|-------------|
| Forward | `-(1/N) · Σ log(y_hat_{class})` | scalar `float` |
| Backward | `-(1/N) · y / (y_hat + ε)` | `(batch, n_classes)` |

Used with `softmax` output. `ε = 1e-15` prevents `log(0)`.

### `MSE` — Mean Squared Error

| Mode | Formula | Output shape |
|------|---------|-------------|
| Forward | `(1/2N) · Σ(y_hat - y)²` | scalar `float` |
| Backward | `(1/N) · (y_hat - y)` | `(batch, n_out)` |

Used with `identity` or `sigmoid` output for regression tasks.

---

## 11. `Network`

**File**: `src/deep_learning/nn/network.py`

The training orchestrator. It links layers, runs forward/backward passes, manages epochs, and handles weight persistence.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layers` | `list[Module]` | required | Ordered list — first is `InputLayer`, last is the output `Dense` |
| `training_set` | `tuple[ndarray, ndarray]` | required | `(x_train, y_train)` |
| `loss_func` | `Callable` | required | `LossFunction.cc_loss` or `LossFunction.MSE` |
| `batch` | `int \| None` | `None` | Mini-batch size; `None` → full-batch GD |
| `learning_rate` | `float` | `0.01` | Global learning rate η |
| `epsilon` | `float` | `0.0001` | Early-stopping threshold on relative smoothed loss change |
| `epoch_limit` | `int` | `10` | Maximum number of full passes over training data |
| `test_set` | `tuple \| None` | `None` | `(x_test, y_test)` for evaluation |
| `iteration_event_trigger` | `int` | `-1` | Print loss every N weight updates (`-1` = never) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `network_layers` | `list[Module]` | All layers in order |
| `x_train`, `y_train` | `np.ndarray` | Training data |
| `x_test`, `y_test` | `np.ndarray` | Test data (if provided) |
| `network_loss` | `float` | Most recent batch loss |
| `iterations` | `int` | Total weight-update steps performed |
| `epoch` | `int` | Current epoch counter |
| `training` | `bool` | Training/inference flag consumed by BN and Dropout |

### Methods

#### `_init_network(self) -> None`

Links layers in a doubly-linked list, injects `network` reference into each layer, then calls `layer.init()` on every layer in order. Must be called before any forward pass.

---

#### `compute_loss(self, targets, derived=False) -> np.ndarray`

Calls `self.loss_func` on the last layer's `layer_output`.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `targets` | `np.ndarray` | `(batch, n_classes)` | Ground-truth for current batch |
| `derived` | `bool` | — | `False` → scalar loss; `True` → `dL/dy_hat` |

- **Returns**: scalar `float` (forward) or `np.ndarray (batch, n_classes)` (backward)

---

#### `_forward_propagation(self, predict_input) -> None`

Calls `layer.forward()` sequentially from input to output.

| Step | Layer | Action |
|------|-------|--------|
| 1 | `layers[0]` (InputLayer) | `forward(predict_input)` — loads batch |
| 2–N | `layers[1:]` | `forward()` — reads from `prv_layer.layer_output` |

---

#### `_backward_propagation(self, targets) -> None`

Two-phase backward:

1. **Delta computation** — iterate layers in **reverse** (output → input), calling `compute_delta_term`.
2. **Weight update** — iterate layers **forward** (skip InputLayer), calling `update_weights(lr)`.

---

#### `fit_model(self) -> None`

Main training loop:

```
for each epoch (until epoch_limit or early stop):
    shuffle training indices
    for each mini-batch:
        forward_propagation(batch_X)
        compute_loss(batch_Y)
        backward_propagation(batch_Y)
        update smoothed_loss  (exponential moving average, β=0.2)
        check early stopping: |Δ smoothed_loss| / smoothed_loss < epsilon
```

---

#### `predict(self, input) -> np.ndarray`

Sets `self.training = False`, runs forward pass, returns the last layer's output.

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `input` | `(batch, ...)` | Matches training data shape |

- **Returns**: `(batch, n_classes)` — raw probabilities (softmax) or regression values.

---

#### `evaluate(self) -> float`

| Loss function | Metric | Description |
|---------------|--------|-------------|
| `cc_loss` | Accuracy `%` | `mean(argmax(y_hat) == argmax(y)) × 100` |
| `MSE` | R² score `%` | `(1 - SSR/SST) × 100` |

---

#### `save_weights(self, path) -> None`

Saves all learnable parameters to a compressed `.npz` file:

| Key pattern | Content |
|-------------|---------|
| `l{idx}_kernels` | Conv layer kernels `(n_f, C, kH, kW)` |
| `l{idx}_weights` | Dense layer weights `(n_neuron, n_in)` |
| `l{idx}_biases` | Biases |
| `l{idx}_bn_gamma`, `l{idx}_bn_beta` | BN scale & shift |
| `l{idx}_bn_run_mean`, `l{idx}_bn_run_var` | BN running stats |

---

#### `load_weights(self, path) -> None`

Restores parameters from `.npz`. Auto-detects 0-based vs 1-based indexing (for backward compatibility between MLP and CNN saved files).

---

## 12. End-to-End Data Flow

### Forward Pass

```
x_train[batch]   (batch, C, H, W)
     │
InputLayer       (batch, C, H, W)     <- stores batch
     │
Conv2D           (batch, n_f1, out_H1, out_W1)
     │  pad -> im2col -> matmul -> BN -> ReLU
MaxPool2D        (batch, n_f1, out_H1//2, out_W1//2)
     │  stride window max
Conv2D           (batch, n_f2, out_H2, out_W2)
     │  pad -> im2col -> matmul -> BN -> ReLU
MaxPool2D        (batch, n_f2, out_H2//2, out_W2//2)
     │  stride window max
Flatten          (batch, n_f2 * out_H2//2 * out_W2//2)
     │  reshape
Dense            (batch, 256)
     │  W^T x + b -> ReLU -> Dropout
Dense            (batch, 10)
     │  W^T x + b -> softmax
     ↓
predictions      (batch, 10)   <- probabilities per class
```

### Backward Pass

```
Loss dL/dy_hat       (batch, 10)
     │  output Dense backward
Dense delta          (batch, 256)    -> dW, db
     │  hidden Dense backward
Dense delta (flat)   (batch, 1568)   -> dW, db
     │  Flatten backward (reshape)
Flatten delta        (batch, 32, 7, 7)
     │  MaxPool backward (scatter)
MaxPool delta        (batch, 32, 14, 14)
     │  Conv2D backward (BN + activation + col2im)
Conv2D delta         (batch, 16, 14, 14)  -> dW, db, d_gamma, d_beta
     │  MaxPool backward
MaxPool delta        (batch, 16, 28, 28)
     │  Conv2D backward
Conv2D delta         (batch, 1, 28, 28)   -> dW, db, d_gamma, d_beta
     │  (InputLayer -- ignored)
```

---

## 13. Full Example — LeNet on MNIST

```python
from deep_learning import (
    Network, InputLayer, Conv2D, MaxPool2D, Flatten, Dense,
    ActivationFunction as Act, LossFunction
)
import numpy as np

# x_train: (60000, 1, 28, 28)  float32 in [0, 1]
# y_train: (60000, 10)         one-hot

layers = [
    InputLayer(x_train),                                 # (N, 1, 28, 28)
    Conv2D(16, 3, act_func=Act.ReLU,                     # (N, 16, 28, 28)
           stride=1, padding=1, use_bn=True),
    MaxPool2D(pool_size=2, stride=2),                    # (N, 16, 14, 14)
    Conv2D(32, 3, act_func=Act.ReLU,
           stride=1, padding=1, use_bn=True),            # (N, 32, 14, 14)
    MaxPool2D(pool_size=2, stride=2),                    # (N, 32,  7,  7)
    Flatten(),                                           # (N, 1568)
    Dense(256, act_func=Act.ReLU,
          use_dropout=True, drop_rate=0.3),              # (N, 256)
    Dense(10,  act_func=Act.softmax),                    # (N, 10)
]

model = Network(
    layers=layers,
    training_set=(x_train, y_train),
    test_set=(x_test, y_test),
    loss_func=LossFunction.cc_loss,
    batch=64,
    learning_rate=0.1,
    epoch_limit=10,
    iteration_event_trigger=100,
)

model.fit_model()
accuracy = model.evaluate()          # R² or classification %
model.save_weights("weights/cnn")    # -> cnn.npz
```

### Learnable Parameters Count (example above)

| Layer | Parameters | Shape |
|-------|-----------|-------|
| Conv2D #1 kernels | 16×1×3×3 = **144** | `(16, 1, 3, 3)` |
| Conv2D #1 biases | **16** | `(16,)` |
| Conv2D #1 BN γ, β | 2×16 = **32** | `(1,16,1,1)` ×2 |
| Conv2D #2 kernels | 32×16×3×3 = **4 608** | `(32, 16, 3, 3)` |
| Conv2D #2 biases | **32** | `(32,)` |
| Conv2D #2 BN γ, β | 2×32 = **64** | `(1,32,1,1)` ×2 |
| Dense #1 weights | 256×1568 = **401 408** | `(256, 1568)` |
| Dense #1 biases | **256** | `(256,)` |
| Dense #2 weights | 10×256 = **2 560** | `(10, 256)` |
| Dense #2 biases | **10** | `(10,)` |
| **Total** | **≈ 409 130** | |

---

*Generated from source: `src/deep_learning/` — pure NumPy implementation.*

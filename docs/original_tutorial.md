# Pure-NumPy CNN — Complete Technical Tutorial (Expanded Edition)

> This document is a fully self-contained technical reference for the CNN implemented in [`cnn.py`](file:///c:/Users/admin/projects/Python/neural_network/cnn.py). Every equation, formula, vectorization trick, shape transformation, and backward-pass derivation is proved from first principles with numeric examples.

---

## Table of Contents

1. [Notation & Conventions](#1-notation--conventions)
2. [Architecture Overview](#2-architecture-overview)
3. [The Training Loop](#3-the-training-loop)
4. [Layer 0 — InputLayer](#4-layer-0--inputlayer)
5. [Layer 1 — ConvLayer](#5-layer-1--convlayer)
   - 5.1 [Kernel Initialization (He)](#51-kernel-initialization-he)
   - 5.2 [im2col — The Full Derivation](#52-im2col--the-full-derivation)
   - 5.3 [Forward Pass — Step by Step](#53-forward-pass--step-by-step)
   - 5.4 [Batch Normalization — Forward](#54-batch-normalization--forward)
   - 5.5 [Backward Pass — Conv: Full Chain Rule](#55-backward-pass--conv-full-chain-rule)
   - 5.6 [Batch Normalization — Backward: Full Computation Graph](#56-batch-normalization--backward-full-computation-graph)
6. [Layer 2 — MaxPoolLayer](#6-layer-2--maxpoollayer)
7. [Layer 3 — FlattenLayer](#7-layer-3--flattenlayer)
8. [Layer 4 — DenseLayer](#8-layer-4--denselayer)
   - 8.1 [Forward Pass & Inverted Dropout (Proof of Expectation)](#81-forward-pass--inverted-dropout-proof-of-expectation)
   - 8.2 [Backward Pass — Output Layer: Softmax Jacobian Proof & CCE Chain](#82-backward-pass--output-layer-softmax-jacobian-proof--cce-chain)
   - 8.3 [Backward Pass — Hidden Dense Layer: Full dW / db / delta Derivation](#83-backward-pass--hidden-dense-layer-full-dw--db--delta-derivation)
9. [Activation Functions & Loss Functions](#9-activation-functions--loss-functions)
10. [Dynamic Learning Rate Decay](#10-dynamic-learning-rate-decay)
11. [Full Worked Example](#11-full-worked-example)

---

## 1. Notation & Conventions

| Symbol | Meaning |
|---|---|
| `B` | Batch size (number of samples in one mini-batch) |
| `N` | Usually `B × H × W` — total scalar elements per channel in BN |
| `C_in` | Number of input channels (depth) |
| `C_out` / `n_f` | Number of output channels = number of convolutional filters |
| `H`, `W` | Spatial height and width of the input feature map |
| `kH`, `kW` | Kernel height and width (square: `kH = kW = k`) |
| `p` | Zero-padding applied symmetrically to H and W dimensions |
| `s` | Stride — step size for the sliding kernel/pool window |
| `out_H`, `out_W` | Output spatial dimensions after convolution or pooling |
| `K` | Kernel tensor, shape `(n_f, C_in, kH, kW)` |
| `W_col` | Kernel reshaped to `(n_f, C_in·kH·kW)` for matmul |
| `col` | im2col matrix, shape `(B, C_in·kH·kW, out_H·out_W)` |
| `W` | Dense weight matrix, shape `(n_neuron, n_in)` |
| `b` | Bias vector |
| `Z` | Pre-activation values (linear output before `act_func`) |
| `A` | Post-activation output = `act_func(Z)` |
| `L` | Scalar loss value |
| `∂L/∂X` | Partial derivative of loss w.r.t. any tensor `X` (gradient) |
| `⊙` | Element-wise (Hadamard) multiplication |
| `@` | Matrix multiplication (batched where applicable) |
| `Xᵀ` | Transpose |
| `μ_c`, `σ²_c` | Per-channel batch mean and variance in BN |
| `x̂` | Standardized value inside BN |
| `γ`, `β` | Learnable BN scale and shift — shape `(1, C, 1, 1)` |
| `ε_bn` | BN numerical stability constant (`1e-5`) |
| `ε_conv` | EWA convergence threshold (early stopping) |
| `δ_{ij}` | Kronecker delta: 1 if `i==j`, else 0 |
| `k, i, j` | Pre-computed im2col index arrays (see §5.2) |
| `p_size` | Pool window size in MaxPool |

**Tensor index notation:**
- `X[b, c, h, w]` — element at batch `b`, channel `c`, row `h`, column `w`.
- `X[b, f, n]` — element at batch `b`, filter `f`, spatial position `n` (flattened).

**Shape convention:** All shapes follow NumPy's C-order (row-major). Broadcasting rules apply across all operations.

---

## 2. Architecture Overview

The LeNet-style MNIST network in [`cnn_run.py`](file:///c:/Users/admin/projects/Python/neural_network/cnn_run.py):

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
DenseLayer 1   dense         256 neurons, ReLU, drop=0.3  (B, 256)
DenseLayer 2   dense         10 neurons, softmax          (B, 10)
─────────────────────────────────────────────────────────────────────────
Output         ŷ (probs)     10 classes                  (B, 10)
```

**OOP Linked-List Design:**
- Each layer is a separate Python object living inside `ConvolutionalNetwork` as a nested class.
- `prv_layer` / `next_layer` pointers form a doubly-linked list.
- No global computation graph is built. Forward and backward passes are achieved by simply iterating the list forward and backward respectively.

> [!IMPORTANT]
> **`layer_delta_term` convention:** Every layer writes `∂L / ∂(its own input)` into `self.layer_delta_term`. The previous layer reads this to continue the chain rule backward. This is the single consistent contract between all layers.

---

## 3. The Training Loop

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

**Why separate ③ and ④?**  
All `layer_delta_term` values are computed (backward sweep) *before* any weights are modified. If we interleaved delta computation with weight updates, a layer's gradient would be computed using already-updated weights from a *later* layer, corrupting the chain rule.

**EWA Convergence Check:**
```
smoothed_loss_t = β · smoothed_loss_{t−1} + (1−β) · loss_t    [β = 0.2]

ratio = |smoothed_{t−1} − smoothed_t| / (smoothed_t + 1e−12)

if ratio < ε_conv  →  stop
```
The exponential moving average smooths per-batch loss noise. `β = 0.2` gives relatively fast response (not oversmoothed). The convergence ratio measures *relative* change, so it is scale-invariant across different loss magnitudes.

---

## 4. Layer 0 — InputLayer

The InputLayer is a pure data holder. Its only job is to expose `layer_output` in the same interface that all downstream layers expect.

```python
def forward(self, x: np.ndarray = None) -> None:
    self.layer_output = x if x is not None else self.layer_input
```

- `layer_input` — the full `(N, C, H, W)` training array stored at construction time.
- `x` — a mini-batch slice `(B, C, H, W)` injected by `_forward_propagation`.

No gradient flows into the InputLayer (there are no parameters to update and no previous layer to propagate to). It has no `compute_delta_term`.

---

## 5. Layer 1 — ConvLayer

The ConvLayer is the mathematical heart of the CNN. Its forward chain is:
```
x  →  pad  →  im2col  →  W_col @ col  →  bias  →  [BN]  →  act_func  →  A
```
Its backward chain is the exact reverse, applying the chain rule at each step.

---

### 5.1 Kernel Initialization (He)

**Problem:** If we initialize weights from `Normal(0, 1)`, the variance of the layer's output explodes or vanishes as it passes through many layers, especially with ReLU.

**He Initialization** (He et al., 2015) sets:
```
W ~ Normal(0, σ²)    where  σ = √(2 / fan_in)
```

**Why `√(2/fan_in)` for ReLU?**

Suppose `Z_j = Σ_{i=1}^{n} W_{ji} · A_i`. If inputs `A_i` are i.i.d. with zero mean and variance `Var(A)`, and weights are i.i.d. with zero mean and variance `Var(W) = σ²`, then:

```
Var(Z_j) = n · σ² · Var(A)
```

ReLU zeroes out negative inputs, roughly halving the variance of the signal at each layer:
```
Var(A) ≈ (1/2) · Var(Z)      [because half the neurons are killed by ReLU]
```

For variance to be preserved layer-to-layer (`Var(A_out) = Var(A_in)`), we need:
```
n · σ² · (1/2) = 1   →   σ² = 2/n = 2/fan_in
```

In code: `fan_in = C_in * kH * kW` — the number of inputs to each neuron in the filter.

```python
fan_in  = C_in * kH * kW
K       = Normal(0, 1) * sqrt(2.0 / fan_in)   # shape: (n_f, C_in, kH, kW)
biases  = zeros(n_f)                            # shape: (n_f,)
```

---

### 5.2 im2col — The Full Derivation

#### Why im2col Exists

A single convolutional output at position `(b, f, oh, ow)` is defined as:

```
Z[b, f, oh, ow] = Σ_c Σ_r Σ_q  K[f, c, r, q] · x_pad[b, c, oh·s+r, ow·s+q]  +  b[f]
```

where `r ∈ [0, kH)`, `q ∈ [0, kW)`, `c ∈ [0, C_in)`.

Naively computing this for all `(b, f, oh, ow)` requires 5 nested Python loops — catastrophically slow. 

**The key insight:** All these sums are dot products between a flattened kernel `K[f, :]` and a flattened receptive field of `x_pad`. If we stack all receptive fields as columns of a matrix, one matmul handles the entire operation.

#### The Three Index Arrays

`_im2col_indices(C_in, kH, kW, out_H, out_W, stride)` pre-computes three 1D/2D index arrays that encode *exactly* which element of `x_pad` participates in which position of `col`.

**`k` — channel index, shape `(C_in·kH·kW,)`:**

The `col` matrix has `C_in·kH·kW` rows. Row `p` corresponds to one specific `(channel, kernel_row, kernel_col)` triplet. `k[p]` is the channel of that triplet.

```
k = repeat([0, 1, ..., C_in−1], repeats=kH*kW)
```

Example `C_in=2, kH=2, kW=2`:
```
k = [0, 0, 0, 0,  1, 1, 1, 1]
     channel-0 elems   channel-1 elems
```

**`row_off` — row offset within kernel, shape `(C_in·kH·kW,)`:**

For a `kH×kW` kernel, the row offsets for its elements are `[0,0,...,1,1,...,kH-1,...,kH-1]` each repeated `kW` times:

```
row_off_single = repeat(arange(kH), repeats=kW)   # e.g. [0,0,1,1] for kH=kW=2
row_off        = tile(row_off_single, C_in)        # repeat for each channel
```

**`row_out` — output row start positions, shape `(out_H,)`:**

When the kernel is at output row `oh`, it reads from input row `oh·s`:
```
row_out = stride * arange(out_H)   # [0, s, 2s, ..., (out_H−1)·s]
```

**`i` — full row index, shape `(C_in·kH·kW, out_H)`:**

`i[p, oh]` = row in `x_pad` of kernel element `p` when output position is row `oh`:
```
i = row_off[:, None] + row_out[None, :]
```

This outer-sum broadcast gives all combinations of kernel row offsets and output positions simultaneously.

**`j` — full column index, shape `(C_in·kH·kW, out_W)`** — constructed identically for columns.

#### `_im2col` — The Gather Operation

```python
def _im2col(x_pad, k, i, j, out_H, out_W):
    col = x_pad[:, k[:, None, None], i[:, :, None], j[:, None, :]]
    return col.reshape(x_pad.shape[0], -1, out_H * out_W)
```

The indexing `x_pad[:, k[...], i[...], j[...]]` is **NumPy advanced indexing** — all four index arrays are broadcast together to produce a single output:

| Axis | Slice | Broadcast shape |
|---|---|---|
| 0 (batch) | `:` → `(B,)` | `(B, 1, 1, 1)` |
| 1 (channel) | `k[:, None, None]` → `(M, 1, 1)` | `(1, M, 1, 1)` where `M = C_in·kH·kW` |
| 2 (row) | `i[:, :, None]` → `(M, out_H, 1)` | `(1, M, out_H, 1)` |
| 3 (col) | `j[:, None, :]` → `(M, 1, out_W)` | `(1, M, 1, out_W)` |

All four broadcast together → result shape `(B, M, out_H, out_W)`.

Reshape to `(B, M, out_H·out_W)` = `(B, C_in·kH·kW, out_H·out_W)`.

**Each column `col[b, :, n]` is the entire flattened `(C_in, kH, kW)` receptive field of image `b` at output position `n`.** This is why it's called `col` — each output position becomes one column.

> [!NOTE]
> **Memory cost:** A pixel at padded position `(r, c)` may appear in up to `kH × kW` different receptive fields (for stride=1). So `col` holds `kH × kW` copies of each pixel on average. For `k=3, s=1` this is 9× the raw input data volume.

#### Why This Converts Convolution to a Single Matmul

The convolution equation for all `f` and `n` (output position) at once:

```
Z[b, f, n] = Σ_p  K_flat[f, p] · col[b, p, n]  +  b[f]
```

where `p` ranges over `C_in·kH·kW` and `n` over `out_H·out_W`. This is exactly a batched matrix multiplication:

```
Z_col[b, :, :] = W_col @ col[b, :, :]
```

Compactly in einsum notation:
```python
Z_col = np.einsum('fc, bcn -> bfn', W_col, col, optimize=True)
```

Reading the einsum:
- `f` → filter index (rows of `W_col`, rows of `Z_col`)
- `c` → contracted index `C_in·kH·kW` (inner dot product)
- `b` → batch (passed through)
- `n` → spatial position (columns of `col` and `Z_col`)

This is equivalent to `Z_col[b] = W_col @ col[b]` for each sample in the batch, but the einsum handles the batch loop internally in compiled C code.

---

### 5.3 Forward Pass — Step by Step

**Input:** `x` shape `(B, C_in, H, W)` from `prv_layer.layer_output`.

**Step 1 — Spatial zero-padding:**
```
x_pad = pad(x, ((0,0), (0,0), (p,p), (p,p)))
x_pad shape: (B, C_in, H+2p, W+2p)
```
Padding adds `p` rows/columns of zeros around the spatial dimensions. This allows the kernel to "slide over" edge pixels, preserving spatial resolution when `p = (k−1)/2` (same-padding).

**Step 2 — im2col:**
```
col = _im2col(x_pad, self._k, self._i, self._j, out_H, out_W)
col shape: (B, C_in·kH·kW, out_H·out_W)
```

**Step 3 — Reshape kernel and multiply:**
```
W_col = K.reshape(n_f, C_in·kH·kW)                   shape: (n_f, C_in·kH·kW)
Z_col = einsum('fc, bcn -> bfn', W_col, col)           shape: (B, n_f, out_H·out_W)
Z_col += b[None, :, None]                              broadcast bias
Z     = Z_col.reshape(B, n_f, out_H, out_W)           shape: (B, n_f, out_H, out_W)
```

**Output spatial size formula** (derived from the sliding window geometry):
```
out_H = floor((H + 2p − kH) / s) + 1
out_W = floor((W + 2p − kW) / s) + 1
```

For Conv1: `H=28, p=1, k=3, s=1` → `(28 + 2 − 3) // 1 + 1 = 28`. Spatial size preserved ✓.

**Step 4 — Batch Normalization (optional):** see §5.4. Output: `Z` (post-BN).

**Step 5 — Activation:**
```
A = act_func(Z)    shape: (B, n_f, out_H, out_W)  →  self.layer_output
```

Cache `col` (needed for `dW` in backward) and `Z` (needed for `act_func` derivative).

---

### 5.4 Batch Normalization — Forward

**Motivation:** Without normalization, the distribution of each layer's inputs shifts as the weights update (called *internal covariate shift*). This forces downstream layers to constantly re-adapt. BN re-centers and re-scales activations per channel, stabilizing training.

**Spatial BN** normalizes over all axes except the channel: axes `(batch=0, H=2, W=3)`.

Let `N = B × out_H × out_W` = total elements per channel per forward pass.

**Step 1 — Compute batch statistics per channel c:**
```
μ_c   = (1/N) · Σ_{b, h, w}  Z[b, c, h, w]          shape: (1, C, 1, 1)
σ²_c  = (1/N) · Σ_{b, h, w}  (Z[b, c, h, w] − μ_c)² shape: (1, C, 1, 1)
```

In NumPy:
```python
mean = Z.mean(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)
var  = Z.var( axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)
```

**Step 2 — Standardize:**
```
x̂[b, c, h, w] = (Z[b, c, h, w] − μ_c) / √(σ²_c + ε_bn)
```

The `+ ε_bn` (1e-5) prevents division by zero when a channel has near-zero variance.

**Step 3 — Scale and shift with learnable parameters:**
```
BN(Z)[b, c, h, w] = γ_c · x̂[b, c, h, w] + β_c
```

`γ` initialized to 1 and `β` to 0 means BN starts as identity — the network can learn to undo normalization if needed.

**Running statistics for inference** (Exponential Moving Average):
```
μ_run  ← mom · μ_run  + (1−mom) · μ_batch    [mom = 0.9]
σ²_run ← mom · σ²_run + (1−mom) · σ²_batch
```

During `predict()` (where `self.training = False`), `μ_run` and `σ²_run` are used instead of batch stats, giving deterministic output regardless of batch content.

---

### 5.5 Backward Pass — Conv: Full Chain Rule

**Incoming gradient:** `incoming = next_layer.layer_delta_term = ∂L/∂A`, shape `(B, n_f, out_H, out_W)`.

---

#### Step ① — Through Activation: `dZ = ∂L/∂Z`

Chain rule: `∂L/∂Z = (∂L/∂A) · (∂A/∂Z)`.

`A = act_func(Z)` is applied element-wise, so `∂A[b,f,h,w]/∂Z[b',f',h',w'] = 0` whenever the indices differ. The Jacobian is diagonal:

```
∂L/∂Z[b, f, h, w] = (∂L/∂A[b, f, h, w]) · act_func'(Z[b, f, h, w])
```

In vectorized form:
```python
dZ = act_func(Z, derived=True) ⊙ incoming   # element-wise, shape (B, n_f, out_H, out_W)
```

For ReLU: `act_func'(z) = 1 if z > 0 else 0`. So `dZ` inherits the spatial pattern of `incoming`, but zeros out everywhere `Z ≤ 0`.

---

#### Step ② — Through Batch Norm (if enabled) → see §5.6

---

#### Step ③ — Kernel Gradient `dK`

Reshape `dZ` to match the shape of `Z_col`:
```
dZ_col = dZ.reshape(B, n_f, out_H·out_W)   shape: (B, n_f, out_H·out_W)
```

Recall the forward pass: `Z_col[b] = W_col @ col[b] + b`. Taking the derivative w.r.t. `W_col`:

```
∂L/∂W_col = Σ_b  ∂L/∂Z_col[b] · col[b]ᵀ
```

**Element-wise:** For a single element `W_col[f, c]`:
```
∂L/∂W_col[f, c] = Σ_b Σ_n  (∂L/∂Z_col[b, f, n]) · col[b, c, n]
```

This is a batched outer product summed over the batch dimension. In einsum:
```python
dW_col = np.einsum('bfn, bcn -> fc', dZ_col, self.col, optimize=True)
# shape: (n_f, C_in·kH·kW)
dK = dW_col.reshape(n_f, C_in, kH, kW)
```

Reading the einsum `'bfn, bcn -> fc'`:
- `b` → summed out (reduce over batch)
- `n` → summed out (reduce over spatial positions)
- `f` → kept (filter)
- `c` → kept (kernel channel×row×col element)

**Bias gradient:**
```python
db = dZ_col.sum(axis=(0, 2))   # shape: (n_f,)
```

`b[f]` broadcasts to every position `(b, f, n)` in `Z_col`, so its gradient is the sum of all upstream deltas touching it.

---

#### Step ④ — Gradient w.r.t. Input: `col2im`

The gradient `∂L/∂col` is the "reverse" of the matmul `Z_col = W_col @ col`:

```
∂L/∂col[b, :, n] = W_colᵀ · ∂L/∂Z_col[b, :, n]
```

Vectorized over all `b` and `n`:
```python
d_col = np.einsum('fc, bfn -> bcn', W_col, dZ_col, optimize=True)
# shape: (B, C_in·kH·kW, out_H·out_W)
```

Now `d_col[b, p, n]` is the gradient w.r.t. the `p`-th element of the receptive field at output position `n` for batch sample `b`. We must **scatter** this back to the original input grid.

**`_col2im` — the scatter operation:**

For each output position `n` with spatial coordinates `(oh, ow)`:
- The `p`-th row of `col` was gathered from `x_pad[b, k[p], i[p, oh], j[p, ow]]`.
- Therefore `d_col[b, p, n]` must be added to `dx_pad[b, k[p], i[p, oh], j[p, ow]]`.

```python
col_r = d_col.reshape(B, C_in·kH·kW, out_H, out_W)

np.add.at(
    x_pad,                                      # target accumulation buffer
    (arange(B)[:, None, None, None],            # (B, 1, 1, 1)
     k[None, :, None, None],                    # (1, M, 1, 1)
     i[None, :, :,    None],                    # (1, M, out_H, 1)
     j[None, :, None, :   ]),                   # (1, M, 1, out_W)
    col_r,                                      # values to scatter-add
)
```

**Why `np.add.at` and not `x_pad[...] += col_r`?**

For stride `s < k`, adjacent output positions share overlapping receptive fields. A single pixel `x_pad[b, c, r, q]` can appear in multiple columns of `col` (up to `kH × kW` columns for full overlap). When scattering gradients back, we must *accumulate* all contributions to that pixel. 

NumPy's `+=` with fancy indexing silently drops all-but-one contribution when duplicate target indices are present. `np.add.at` is the buffered (non-fancy) form that correctly handles repeated indices.

**Example of the overlap problem (stride=1, k=2):**

Pixel `x_pad[0, 0, 1, 1]` participates in receptive fields at output positions `(0,0), (0,1), (1,0), (1,1)` — 4 different output windows. All four windows contribute gradients back to the same pixel. `np.add.at` accumulates all four correctly.

Finally, crop the padding:
```python
layer_delta_term = x_pad[:, :, p:-p, p:-p]   # shape: (B, C_in, H, W)
```

---

### 5.6 Batch Normalization — Backward: Full Computation Graph

**Inputs to backward:** `d_out = ∂L/∂(BN output)`, shape `(B, C, out_H, out_W)`.

We need `∂L/∂Z` where `Z` is the raw (pre-BN) conv output. The computation graph through BN has three dependent paths from `Z` to the output:

```
Z  →  x̂  →  BN_out           [path 1: direct via x̂]
Z  →  μ   →  x̂  →  BN_out    [path 2: through the mean]
Z  →  σ²  →  x̂  →  BN_out    [path 3: through the variance]
```

We work backwards through each path using the chain rule.

**Learnable parameter gradients (easy — just contractions):**

Since `BN_out = γ · x̂ + β`:
```
∂L/∂γ_c = Σ_{b,h,w} d_out[b,c,h,w] · x̂[b,c,h,w]     shape: (1,C,1,1)
∂L/∂β_c = Σ_{b,h,w} d_out[b,c,h,w]                    shape: (1,C,1,1)
```

In code:
```python
self.d_gamma = (d_out * self.x_hat).sum(axis=(0,2,3), keepdims=True)
self.d_beta  =  d_out             .sum(axis=(0,2,3), keepdims=True)
```

**Gradient through the normalization — step by step:**

Define: `inv_std = 1/√(σ²+ε)`, `diff = Z − μ`, `N = B·H·W`.

**Step 1: `dx̂ = ∂L/∂x̂`** — gradient of loss w.r.t. normalized value:
```
dx̂ = d_out · γ                    shape: (B, C, H, W)
```
Because `BN_out[b,c,h,w] = γ_c · x̂[b,c,h,w] + β_c`, the local Jacobian `∂BN_out/∂x̂ = γ`.

**Step 2: `dσ² = ∂L/∂σ²_c`** — gradient of loss w.r.t. variance:

`x̂ = diff · (σ²+ε)^{−½}`, so `∂x̂/∂σ² = diff · (−½) · (σ²+ε)^{−3/2}`:
```
dσ² = Σ_{b,h,w}  dx̂[b,c,h,w] · diff[b,c,h,w] · (−½) · (σ²_c+ε)^{−3/2}
    shape: (1, C, 1, 1)
```

In code:
```python
dvar = (dx_hat * diff * (-0.5) * (var + eps)**(-1.5)).sum(axis=(0,2,3), keepdims=True)
```

**Step 3: `dμ = ∂L/∂μ_c`** — gradient of loss w.r.t. mean:

`μ` affects `x̂` through `diff = Z − μ`, giving `∂x̂/∂μ = −inv_std`. Also `μ` affects `σ²` through `σ² = (1/N)·Σ diff²`, giving `∂σ²/∂μ = (−2/N)·Σ diff`:

```
dμ = [Σ_{b,h,w} dx̂ · (−inv_std)]  +  dσ² · (−2/N) · [Σ_{b,h,w} diff]
   shape: (1, C, 1, 1)
```

In code:
```python
dmean = ((dx_hat * (-inv_std)).sum(axis=(0,2,3), keepdims=True)
       + dvar * (-2.0 * diff).sum(axis=(0,2,3), keepdims=True) / N)
```

**Step 4: `∂L/∂Z`** — combine all three paths:

`Z` affects `x̂` directly (via `diff`), via `μ`, and via `σ²`:

```
∂L/∂Z[b,c,h,w] = dx̂[b,c,h,w] · inv_std              [path 1: direct]
                + dσ² · 2·diff[b,c,h,w] / N           [path 3: through variance]
                + dμ / N                               [path 2: through mean]
```

In code:
```python
dZ = dx_hat * inv_std  +  dvar * 2.0 * diff / N  +  dmean / N
```

This `dZ` is what gets passed to the conv backward step to compute `dK` and `d_col`.

---

## 6. Layer 2 — MaxPoolLayer

Max pooling down-samples the feature maps, reducing spatial resolution while retaining the strongest activations.

### 6.1 Forward Pass — Strided-View Trick

**Output size:**
```
out_H = (H − p_size) // s + 1
out_W = (W − p_size) // s + 1
```

A naïve forward pass would loop over `(b, c, oh, ow)` and call `max()` on each `p_size × p_size` window. Instead, the implementation creates a virtual 6D view using **NumPy stride tricks**:

```python
shape   = (B, C, out_H, out_W, p_size, p_size)
st      = x.strides          # (bytes_per_sample, bytes_per_channel, bytes_per_row, bytes_per_elem)
strides = (st[0],     st[1],     s*st[2],   s*st[3],   st[2],   st[3])
windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
```

**What strides mean:** A stride value tells NumPy "to advance one step along this axis, jump this many bytes in memory."

| Axis | stride | Meaning |
|---|---|---|
| 0 (batch) | `st[0]` | Next sample |
| 1 (channel) | `st[1]` | Next channel |
| 2 (output row `oh`) | `s · st[2]` | Jump `s` rows forward in input |
| 3 (output col `ow`) | `s · st[3]` | Jump `s` columns forward in input |
| 4 (pool row `r`) | `st[2]` | Step one row within the pool window |
| 5 (pool col `q`) | `st[3]` | Step one column within the pool window |

`as_strided` **creates no new memory allocation** — it is a zero-copy view. All `(B, C, out_H, out_W, p, p)` windows are described entirely by the stride table pointing into the original `x` buffer.

```python
out  = windows.max(axis=(4, 5))                          # (B, C, out_H, out_W)
mask = (windows == out[:, :, :, :, None, None])          # (B, C, out_H, out_W, p, p)
```

The boolean `mask` is `True` at the position(s) that achieved the maximum in each window.

### 6.2 Backward Pass — Gradient Routing

Max pooling passes the gradient **only to the positions that produced the maximum value** — all other positions get zero gradient. This makes intuitive sense: only the maximum contributed to the output, so only it needs to be adjusted.

**Tied maxima** (two positions share the same max): split the gradient equally.

```python
count     = mask.sum(axis=(4,5), keepdims=True).clip(min=1)   # number of max positions
d_windows = incoming[:,:,:,:,None,None] * (mask / count)
#           shape: (B, C, out_H, out_W, p, p)
```

**Why `clip(min=1)`?** To prevent division by zero if a window somehow has `count=0` (impossible in practice but safe).

**Scatter back to input grid:**
```python
d_input = np.zeros(self.x_shape, dtype=incoming.dtype)   # (B, C, H, W)

for oh in range(out_H):
    for ow in range(out_W):
        d_input[:, :, oh*s : oh*s+p, ow*s : ow*s+p] += d_windows[:, :, oh, ow]
```

The loop is over output positions only (`7×7 = 49` for pool2 in MNIST). Each iteration does a vectorized slice assignment across the full `(B, C)` dimensions. The `+=` is correct here (no duplicate index problem) because each output position maps to a distinct spatial region of `d_input` when stride `s ≥ p` (non-overlapping). For `s < p` (overlapping pools), `+=` accumulates from multiple windows — which is still mathematically correct because each pixel's total gradient is the sum of gradients from all output positions that included it.

---

## 7. Layer 3 — FlattenLayer

The bridge between the 4D conv world and the 2D dense world.

**Forward:**
```python
self.conv_shape   = x.shape                # save (B, C, H, W) for backward
self.layer_output = x.reshape(x.shape[0], -1)   # (B, C·H·W)
```

`reshape` is **zero-copy** — it returns a view of the same memory with new strides. No data is moved.

**Backward:**
```python
self.layer_delta_term = self.next_layer.layer_delta_term.reshape(self.conv_shape)
# (B, C·H·W) → (B, C, H, W)
```

The incoming gradient from the first DenseLayer has shape `(B, 1568)`. We simply reinterpret it as `(B, 32, 7, 7)` so MaxPool2 can read it correctly. Again zero-copy.

**Why is the reshape valid?** The memory layout of the flattened array is identical to the 4D array because NumPy uses C-order (row-major) by default, and `.reshape(-1)` preserves this order. The backward pass reshape reverses the exact same memory order.

---

## 8. Layer 4 — DenseLayer

Standard fully-connected layer: every input neuron connects to every output neuron.

**Weights:** `W` shape `(n_neuron, n_in)` — rows are neurons, columns are inputs.

---

### 8.1 Forward Pass & Inverted Dropout (Proof of Expectation)

**Linear transformation:**
```
Z = x @ Wᵀ + b
```

Expanding element-wise:
```
Z[b, j] = Σ_{i=0}^{n_in−1}  x[b, i] · W[j, i]  +  b[j]
```

**Activation:** `A = act_func(Z)`, shape `(B, n_neuron)`.

#### Inverted Dropout — Why Divide by `(1 − drop_rate)`?

**Standard dropout** randomly sets `drop_rate` fraction of activations to zero. The problem: if dropout is on during training but off at test time, the expected magnitude of a neuron's output changes.

**At training time (with dropout, rate `p`):**
- Expected value of one neuron `A_j` after masking: `E[A_j · mask] = A_j · (1−p)`.

**At test time (no dropout):**
- The neuron outputs `A_j` directly.

This mismatch means the downstream layer receives inputs of a different expected scale at test time, requiring all weights to be rescaled.

**Inverted dropout** fixes this by dividing the surviving activations by `(1−p)` *during training*:

```
A_masked = A · mask / (1−p)
E[A_masked] = A_j · (1−p) / (1−p) = A_j
```

The expected value is now the same as at test time — no adjustment needed at inference. The model sees consistent statistics regardless of training or evaluation mode.

In code:
```python
mask = (np.random.random(A.shape) > drop_rate) / (1.0 - drop_rate)
A    = A * mask
```

`np.random.random(A.shape) > drop_rate` produces a boolean array where each entry is `True` with probability `1 − drop_rate`. Dividing by `(1 − drop_rate)` applies the rescaling in-place. During `predict()`, `self.training = False` causes the `if` block to be skipped entirely — `drop_mask = None` and `A` is returned unchanged.

---

### 8.2 Backward Pass — Output Layer: Softmax Jacobian Proof & CCE Chain

The output layer combines:
1. **Categorical Cross-Entropy (CCE)** loss
2. **Softmax** activation

Each is differentiated separately (not using the combined shortcut) to keep the code modular.

#### Categorical Cross-Entropy Loss

For a batch of `B` samples and `C=10` classes:

```
L = −(1/B) · Σ_{b=0}^{B−1} Σ_{c=0}^{C−1}  y[b,c] · log(ŷ[b,c] + 1e−15)
```

where `y[b,c]` is the one-hot target (0 or 1) and `ŷ[b,c] = A[b,c]` is the softmax probability.

Since `y` is one-hot, only the true class term survives:
```
L = −(1/B) · Σ_b  log(ŷ[b, true_class_b] + 1e−15)
```

**Derivative `∂L/∂ŷ[b,c]`:**
```
∂L/∂ŷ[b,c] = −(1/B) · y[b,c] / (ŷ[b,c] + 1e−15)
```

For non-true classes where `y[b,c]=0`, this is zero. For the true class, it is `−(1/B)/ŷ`. Shape: `(B, C)`.

#### Softmax Function and Its Jacobian — Full Derivation

For a single sample with logit vector `Z` of length `C`:
```
S_i = exp(Z_i − max(Z)) / Σ_{j} exp(Z_j − max(Z))
```

The numerically stable form subtracts `max(Z)` — this doesn't change the output (common factor cancels) but prevents overflow.

**Jacobian element `∂S_i/∂Z_j`:** We need this for the chain rule.

Let `e_i = exp(Z_i − max(Z))` and `D = Σ_j e_j`, so `S_i = e_i / D`.

**Case 1: `i = j`** (diagonal)
```
∂S_i/∂Z_i = ∂(e_i/D)/∂Z_i
           = (e_i · D − e_i · e_i) / D²
           = S_i − S_i²
           = S_i(1 − S_i)
           = S_i(δ_{ii} − S_i)
```

**Case 2: `i ≠ j`** (off-diagonal)
```
∂S_i/∂Z_j = ∂(e_i/D)/∂Z_j
           = (0 · D − e_i · e_j) / D²
           = −S_i · S_j
           = S_i(0 − S_j)
           = S_i(δ_{ij} − S_j)
```

**Unified formula:**
```
∂S_i/∂Z_j = S_i · (δ_{ij} − S_j)
```

In matrix form for one sample, the full Jacobian `J ∈ ℝ^{C×C}`:
```
J = diag(S) − S·Sᵀ
```

**Batched Jacobian in NumPy:**
```python
s         = S[:, :, None]                          # (B, C, 1)
diag_part = s * np.eye(C)[None, :, :]              # (B, C, C)  — scaled identity
outer_part = s * s.transpose(0, 2, 1)             # (B, C, C)  — outer product S·Sᵀ
J          = diag_part - outer_part                # (B, C, C)
```

#### Full Chain Rule: `δ = ∂L/∂Z`

We need `∂L/∂Z[b, j]` = total effect of `Z[b,j]` on the loss.

```
∂L/∂Z[b, j] = Σ_i  (∂L/∂S[b, i]) · (∂S[b, i]/∂Z[b, j])
             = Σ_i  (∂L/∂ŷ[b, i]) · J[b, i, j]
```

In matrix form for one sample: `δ[b, :] = (∂L/∂ŷ[b, :]) @ J[b, :, :]`.

For the batch:
```python
grad_loss = compute_loss(derived=True, targets=targets)    # (B, C)
grad_loss_exp = grad_loss[:, None, :]                      # (B, 1, C)
delta = (grad_loss_exp @ J).reshape(B, C)                  # (B, C)
```

The `(B, 1, C) @ (B, C, C)` batched matmul contracts the `C` inner dimension, giving `(B, 1, C)`. Reshape to `(B, C)`.

> [!NOTE]
> **The shortcut:** If you use CCE + Softmax together, the chain `(∂L/∂ŷ) @ J` simplifies beautifully to `ŷ − y`. This implementation deliberately avoids that shortcut to show every step separately.

#### Weight and Bias Gradients

`Z[b, j] = Σ_i x[b, i] · W[j, i] + b[j]`, so:

```
∂L/∂W[j, i] = Σ_b  δ[b, j] · x[b, i]
```

In matrix form: `dW = δᵀ @ x`.

```python
self.dW = delta.T @ self.prv_layer.layer_output   # (n_neuron, n_in)
self.db = delta.sum(axis=0)                        # (n_neuron,)
```

`db[j] = Σ_b δ[b, j]` — bias gradient is just the sum of deltas over the batch.

#### Gradient Passed Backward: `layer_delta_term`

`layer_delta_term = ∂L/∂x = δ @ W`

```
∂L/∂x[b, i] = Σ_j  δ[b, j] · W[j, i]
```

In matrix form: `delta @ W`, shape `(B, n_in)`.

```python
self.layer_delta_term = delta @ self.weights   # (B, n_in)
```

This is what the FlattenLayer (or previous DenseLayer) reads as `next_layer.layer_delta_term`.

---

### 8.3 Backward Pass — Hidden Dense Layer: Full dW / db / delta Derivation

For a hidden DenseLayer (not the output), the incoming gradient is:
```
incoming = self.next_layer.layer_delta_term    # ∂L/∂A, shape (B, n_neuron)
```

#### Step 1 — Apply Dropout Mask to Incoming Gradient

During forward, `A = act_func(Z) ⊙ mask`. Since `mask` gates which neurons are active, only active neurons contribute to the output and thus to the loss. The backward pass must apply the same mask to the incoming gradient:

```python
if self.drop_mask is not None:
    incoming = incoming * self.drop_mask   # zero out gradient for dropped neurons
```

This is the chain rule: `∂L/∂A_pre_dropout[b,j] = (∂L/∂A[b,j]) · mask[b,j]`.

Without this, the network would update neurons that were deactivated during forward — a contradiction since they didn't contribute to the loss.

#### Step 2 — Through Activation: `delta = ∂L/∂Z`

```python
delta = act_func(Z, derived=True) * incoming    # (B, n_neuron)
```

Element-wise chain rule: `∂L/∂Z[b,j] = (∂L/∂A[b,j]) · f'(Z[b,j])`.

For **ReLU**: `f'(z) = 1 if z > 0 else 0`. The gradient is zeroed wherever the pre-activation was non-positive — these neurons are said to be in the "dead" state.

#### Step 3 — Weight and Bias Gradients

Identical algebra to the output layer:

```
∂L/∂W[j, i] = Σ_b  δ[b, j] · x_input[b, i]
∂L/∂b[j]    = Σ_b  δ[b, j]
```

```python
self.dW = delta.T @ self.prv_layer.layer_output   # (n_neuron, n_in)
self.db = delta.sum(axis=0)                        # (n_neuron,)
```

#### Step 4 — Propagate Backward: `layer_delta_term = ∂L/∂x_input`

```
∂L/∂x_input[b, i] = Σ_j  δ[b, j] · W[j, i]
```

```python
self.layer_delta_term = delta @ self.weights   # (B, n_in)
```

For the first DenseLayer, `n_in = 1568` and this result will be read by the FlattenLayer.

#### Weight Update (SGD)

After all delta terms are computed for all layers, `_backward_propagation()` applies gradient descent:

```python
def update_weights(self, lr):
    self.weights -= lr * self.dW
    self.biases  -= lr * self.db
```

The update is: `W_{t+1} = W_t − lr · ∂L/∂W_t`. The negative sign: we move in the direction that *decreases* the loss.

---

## 9. Activation Functions & Loss Functions

### ReLU (Rectified Linear Unit)

```
f(z)  = max(0, z)      forward
f'(z) = 1  if z > 0    backward (indicator function)
        0  if z ≤ 0
```

Derivative is not defined at `z=0`; the code uses the convention `f'(0) = 0` (i.e., `Z > 0` strictly).

**Property:** Non-saturating for positive inputs (gradient = 1, no vanishing gradient in that region). Can suffer from "dying ReLU" where neurons get stuck with `Z ≤ 0` permanently if `lr` is too large.

### Softmax (with numerical stability)

```
Z'   = Z − max(Z, axis=1, keepdims=True)    # shift to prevent exp overflow
S_i  = exp(Z'_i) / Σ_j exp(Z'_j)
```

Output is a valid probability distribution: all values in `(0,1)`, sum to 1 per sample.

Derivative: full Jacobian derived in §8.2.

### Categorical Cross-Entropy

```
L(y, ŷ) = −(1/B) · Σ_{b,c}  y[b,c] · log(ŷ[b,c] + 1e−15)

∂L/∂ŷ = −(1/B) · y / (ŷ + 1e−15)
```

The `1e−15` floor prevents `log(0)`. In practice with softmax output `ŷ ∈ (0,1)`, this rarely triggers.

### SGD Weight Update

Applied after all `layer_delta_term` values are computed:

```
K  ← K  − lr · dK       (ConvLayer kernels)
b  ← b  − lr · db       (ConvLayer biases)
γ  ← γ  − lr · dγ       (BN scale)
β  ← β  − lr · dβ       (BN shift)
W  ← W  − lr · dW       (DenseLayer weights)
b  ← b  − lr · db       (DenseLayer biases)
```

---

## 10. Dynamic Learning Rate Decay

After each epoch completes:
```
lr_{e+1} = lr_e · r    where r = lr_decay_rate ∈ (0, 1]
```

General formula after epoch `e` (starting from `e=0`):
```
lr_e = lr_0 · r^e
```

This is exponential decay. The learning rate approaches zero asymptotically, enabling coarse exploration early in training and fine-grained convergence later.

| `lr_decay_rate` | Effect |
|---|---|
| `1.0` | No decay — constant learning rate |
| `0.95` | −5% per epoch (15 epochs: lr drops to 46% of initial) |
| `0.80` | −20% per epoch (used in `cnn_run.py` test run) |
| `0.50` | Halved every epoch — very aggressive |

**Decay table for `lr_0=0.1`, `r=0.8`:**

| Epoch | lr |
|---|---|
| 1 | 0.100000 |
| 2 | 0.080000 |
| 3 | 0.064000 |
| 5 | 0.040960 |
| 10 | 0.010737 |
| 15 | 0.002814 |

In code, decay happens at the end of the epoch loop, before the convergence check:
```python
self.learning_rate *= self.lr_decay_rate
```

---

## 11. Full Worked Example

We trace **one complete forward + backward pass** for a micro-network:

```
Setup:
  B      = 1 (single sample, for clarity)
  C_in   = 1 (grayscale)
  H = W  = 4 (4×4 input)
  n_f    = 1 (single filter)
  kH=kW  = 2 (2×2 kernel)
  stride = 1
  padding= 0
  No BN, act_func = ReLU
  Dense output: 1 neuron, identity activation
  Loss: MSE
```

**Output size:** `out_H = out_W = (4−2)//1+1 = 3`.

---

### A. im2col Index Pre-computation

```
C_in·kH·kW = 1·2·2 = 4  (4 rows in col matrix)

k = [0, 0, 0, 0]       (all channel 0)

row_off = tile(repeat([0,1], 2), 1) = [0, 0, 1, 1]
col_off = tile(tile([0,1], 2),   1) = [0, 1, 0, 1]
row_out = 1 * [0, 1, 2] = [0, 1, 2]
col_out = 1 * [0, 1, 2] = [0, 1, 2]

i = [[0,0,1,1]]ᵀ + [[0,1,2]]
  = [[0,1,2],
     [0,1,2],
     [1,2,3],
     [1,2,3]]                  shape: (4, 3)

j = [[0,1,0,1]]ᵀ + [[0,1,2]]
  = [[0,1,2],
     [1,2,3],
     [0,1,2],
     [1,2,3]]                  shape: (4, 3)
```

Each row of `i` and `j` gives the input row and column for one of the 4 kernel elements, across all 3 output positions. Element `(p, oh)` of `i` gives the row in `x_pad` that kernel element `p` reads from at output row `oh`.

---

### B. Forward Pass — ConvLayer

**Input x, shape (1, 1, 4, 4):**
```
x[0,0] =
  1  2  3  4
  5  6  7  8
  9 10 11 12
 13 14 15 16
```

No padding (`p=0`), so `x_pad = x`.

**im2col — col, shape (1, 4, 9):**

For sample 0, col matrix columns correspond to the 9 output positions in row-major order: `(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)`.

Row `p` of col = kernel element `p` read from its input position.

```
col[0] (shape 4×9):
           (0,0)(0,1)(0,2)(1,0)(1,1)(1,2)(2,0)(2,1)(2,2)
p=0 (top-left):    1   2   3   5   6   7   9  10  11
p=1 (top-right):   2   3   4   6   7   8  10  11  12
p=2 (bot-left):    5   6   7   9  10  11  13  14  15
p=3 (bot-right):   6   7   8  10  11  12  14  15  16
```

Verify position `(1,1)` (output grid row 1, col 1): receptive field is `x[1:3, 1:3]` = `[6,7; 10,11]`. Column 4 of `col[0]` = `[6, 7, 10, 11]` ✓.

**Kernel K, shape (1, 1, 2, 2) → W_col shape (1, 4):**
```
K[0,0] = [[1, -1],
           [0,  1]]
W_col   = [1, -1, 0, 1]
```

**Convolution via matmul:**

```
Z_col[0, 0, :] = W_col @ col[0]
               = [1,-1,0,1] @ col[0]          shape: (9,)

pos (0,0): [1,-1,0,1]·[1,2,5,6]  = 1-2+0+6  = 5
pos (0,1): [1,-1,0,1]·[2,3,6,7]  = 2-3+0+7  = 6
pos (0,2): [1,-1,0,1]·[3,4,7,8]  = 3-4+0+8  = 7
pos (1,0): [1,-1,0,1]·[5,6,9,10] = 5-6+0+10 = 9
pos (1,1): [1,-1,0,1]·[6,7,10,11]= 6-7+0+11 = 10
pos (1,2): [1,-1,0,1]·[7,8,11,12]= 7-8+0+12 = 11
pos (2,0): [1,-1,0,1]·[9,10,13,14]=9-10+0+14=13
pos (2,1): [1,-1,0,1]·[10,11,14,15]=10-11+0+15=14
pos (2,2): [1,-1,0,1]·[11,12,15,16]=11-12+0+16=15
```

```
Z = Z_col.reshape(1, 1, 3, 3) =
  [[ 5,  6,  7],
   [ 9, 10, 11],
   [13, 14, 15]]
```

**After ReLU:** All values positive → `A = Z` (no change).

**After Flatten:** `A_flat = [5, 6, 7, 9, 10, 11, 13, 14, 15]`, shape `(1, 9)`.

**DenseLayer (9 inputs → 1 output, identity activation):**
```
W_dense = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]    shape: (1, 9)
b_dense = 0

Z_dense = A_flat @ W_denseᵀ + b
        = 0.1·(5+6+7+9+10+11+13+14+15)
        = 0.1 · 90 = 9.0

ŷ = 9.0   (identity activation)
```

**Loss (MSE), target `y = 10.0`:**
```
L = (1/(2·1)) · (9.0 − 10.0)² = 0.5 · 1 = 0.5
```

---

### C. Backward Pass

**MSE derivative:**
```
∂L/∂ŷ = (1/1) · (ŷ − y) = 9.0 − 10.0 = −1.0
```

**DenseLayer backward:**
```
delta_dense = ∂L/∂ŷ · f'(Z) = −1.0 · 1 = −1.0     (identity f'=1)

dW_dense = delta_dense.T @ A_flat
         = [−1.0] @ [[5,6,7,9,10,11,13,14,15]]
         = [−5, −6, −7, −9, −10, −11, −13, −14, −15]   shape: (1, 9)

db_dense = delta_dense.sum() = −1.0

layer_delta_term_dense = delta_dense @ W_dense
                       = [−1.0] @ [[0.1,...,0.1]]
                       = [−0.1, −0.1, ..., −0.1]        shape: (1, 9)
```

**FlattenLayer backward:** Reshape `(1, 9)` → `(1, 1, 3, 3)`:
```
d_conv =
  [[-0.1, -0.1, -0.1],
   [-0.1, -0.1, -0.1],
   [-0.1, -0.1, -0.1]]       (incoming to ConvLayer)
```

**ConvLayer backward — through ReLU:**
```
dZ = (Z > 0) ⊙ d_conv = 1 ⊙ d_conv = d_conv   (all Z > 0, ReLU passes all)
```

**Kernel gradient:**
```
dZ_col = dZ.reshape(1, 1, 9) = [[-0.1, ..., -0.1]]    shape: (1, 1, 9)

dW_col = einsum('bfn, bcn -> fc', dZ_col, col)
       = Σ_{n=0}^{8}  dZ_col[0,0,n] · col[0,:,n]
       = −0.1 · col[0,:,:].sum(axis=1)
       = −0.1 · [row sums of col[0]]

row sums of col[0]:
  p=0: 1+2+3+5+6+7+9+10+11 = 54
  p=1: 2+3+4+6+7+8+10+11+12= 63
  p=2: 5+6+7+9+10+11+13+14+15 = 90
  p=3: 6+7+8+10+11+12+14+15+16 = 99

dW_col = [−5.4, −6.3, −9.0, −9.9]

dK = dW_col.reshape(1, 1, 2, 2) =
  [[-5.4, -6.3],
   [-9.0, -9.9]]
```

**Gradient to input via col2im:**
```
d_col = einsum('fc, bfn -> bcn', W_col, dZ_col)
      = W_colᵀ · dZ_col[0,0,:]
      = [1,-1,0,1]ᵀ · [−0.1,…,−0.1]
      = [[−0.1], [0.1], [0.0], [−0.1]] broadcast to shape (1, 4, 9)
```

Each row `p` of `d_col[0]` = `W_col[0,p] · (−0.1)` = `[−0.1, 0.1, 0.0, −0.1]`.

`col2im` scatters these back to `dx_pad` of shape `(1,1,4,4)`. Pixel `(0,0)` appears only in window `(0,0)` so:
```
dx[0,0,0,0] = d_col[0, 0, 0] = −0.1   (from p=0, output pos 0)
```

Pixel `(0,1)` appears in windows `(0,0)` (as p=1) and `(0,1)` (as p=0):
```
dx[0,0,0,1] = d_col[0,1,0] + d_col[0,0,1] = 0.1 + (−0.1) = 0.0
```

This gradient overlap accumulation via `np.add.at` is the key correctness requirement in col2im.

**Weight Update (lr = 0.01):**
```
K ← K − 0.01 · dK
  = [[1,-1],[0,1]] − 0.01 · [[-5.4,-6.3],[-9.0,-9.9]]
  = [[1+0.054, -1+0.063],
     [0+0.090,  1+0.099]]
  = [[1.054, -0.937],
     [0.090,  1.099]]
```

The kernel has updated in the direction that would reduce the loss — specifically, it has increased values that contributed positively to the output (since the output was too low).

---

## Summary — Data Flow Shapes for MNIST LeNet

```
Stage                        Tensor               Shape
─────────────────────────────────────────────────────────────────────────
Raw input                    x                    (B,  1, 28, 28)
After pad Conv1 (p=1)        x_pad                (B,  1, 30, 30)
im2col Conv1                 col                  (B,  9, 784)
                             [C·k²=1·9=9, 28²=784]
Pre-BN Conv1                 Z_raw                (B, 16, 28, 28)
Post-BN, Post-ReLU Conv1     A                    (B, 16, 28, 28)
After MaxPool1               A                    (B, 16, 14, 14)
After pad Conv2 (p=1)        x_pad                (B, 16, 16, 16)
im2col Conv2                 col                  (B, 144, 196)
                             [C·k²=16·9=144, 14²=196]
Pre-BN Conv2                 Z_raw                (B, 32, 14, 14)
Post-BN, Post-ReLU Conv2     A                    (B, 32, 14, 14)
After MaxPool2               A                    (B, 32,  7,  7)
After Flatten                A                    (B, 1568)
Dense1 Z, A                  Z, A                 (B, 256)
Dense2 Z, ŷ                  Z, A                 (B, 10)
Loss L                       scalar               ()
─────────────────────────────────────────────────────────────────────────
Backward — layer_delta_term (∂L/∂layer_input) shapes:
Dense2   → Dense1            delta_term           (B, 256)
Dense1   → Flatten           delta_term           (B, 1568)
Flatten  → MaxPool2          delta_term           (B, 32, 7, 7)
MaxPool2 → Conv2             delta_term           (B, 32, 14, 14)
Conv2    → MaxPool1          delta_term           (B, 16, 14, 14)
MaxPool1 → Conv1             delta_term           (B, 16, 28, 28)
Conv1    → [InputLayer]      delta_term           (B,  1, 28, 28)
─────────────────────────────────────────────────────────────────────────
Trainable parameter counts:
Conv1:  kernels (16·1·3·3=144) + biases (16) + BN γ,β (16+16)   = 192
Conv2:  kernels (32·16·3·3=4608) + biases (32) + BN γ,β (32+32) = 4704
Dense1: W (256·1568=401408) + b (256)                             = 401664
Dense2: W (10·256=2560)     + b (10)                              = 2570
                                                           TOTAL  = 409,130
```

---

*End of tutorial. Every formula maps 1-to-1 with source code in [`cnn.py`](file:///c:/Users/admin/projects/Python/neural_network/cnn.py). Cross-reference section numbers with the corresponding layer classes and their `forward()`, `compute_delta_term()`, and `update_weights()` methods.*

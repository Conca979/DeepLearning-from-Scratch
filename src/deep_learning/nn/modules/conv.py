import numpy as np
from typing import Callable
from ..core import Module


def _im2col_indices(C_in: int, kH: int, kW: int, out_H: int, out_W: int, stride: int):
  row_off = np.repeat(np.arange(kH), kW)
  row_off = np.tile(row_off, C_in)  # (C_in*kH*kW,)
  row_out = stride * np.arange(out_H)  # (out_H,)

  col_off = np.tile(np.arange(kW), kH)
  col_off = np.tile(col_off, C_in)  # (C_in*kH*kW,)
  col_out = stride * np.arange(out_W)  # (out_W,)

  k = np.repeat(np.arange(C_in), kH * kW)  # (C_in*kH*kW,)
  i = row_off[:, None] + row_out[None, :]  # (C_in*kH*kW, out_H)
  j = col_off[:, None] + col_out[None, :]  # (C_in*kH*kW, out_W)
  return k, i, j


def _im2col(x_pad: np.ndarray, k: np.ndarray, i: np.ndarray, j: np.ndarray,
            out_H: int, out_W: int) -> np.ndarray:

  col = x_pad[:, k[:, None, None], i[:, :, None], j[:, None, :]]
  # col shape before reshape: (batch, C_in*kH*kW, out_H, out_W)
  return col.reshape(x_pad.shape[0], -1, out_H * out_W)


def _col2im(col: np.ndarray, x_shape: tuple, k: np.ndarray, i: np.ndarray,
            j: np.ndarray, out_H: int, out_W: int, padding: int) -> np.ndarray:

  batch, C_in, H, W = x_shape
  H_pad = H + 2 * padding
  W_pad = W + 2 * padding
  x_pad = np.zeros((batch, C_in, H_pad, W_pad), dtype=col.dtype)

  col_r = col.reshape(batch, -1, out_H,
                      out_W)  # (batch, C_in*kH*kW, out_H, out_W)

  # Vectorised scatter: broadcast batch and spatial axes simultaneously.
  # np.add.at handles duplicate indices (overlapping receptive fields).
  np.add.at(
      x_pad,
      (np.arange(batch)[:, None, None, None],  # (batch, 1,         1,     1    )
                      k[None, :, None, None],  # (1,     C*kH*kW,  1,     1    )
                      i[None, :, :, None],  # (1,     C*kH*kW,  out_H, 1    )
                      j[None, :, None, :]),  # (1,     C*kH*kW,  1,     out_W)
                      col_r,
              )
  return x_pad[:, :, padding:-padding,
               padding:-padding] if padding > 0 else x_pad

class InputLayer:
  def __init__(self, x_train: np.ndarray | None):
    self.layer_input = x_train  # (N, C, H, W)
    self.out_shape = x_train.shape[1:]  # (C, H, W)  — spatial dims only
    self.layer_output = None
    self.next_layer = None

  def forward(self, x: np.ndarray = None) -> None:
    self.layer_output = x if x is not None else self.layer_input

class Conv2D(Module):
  def __init__(self,
               n_filters: int,
               kernel_size: int,
               act_func: Callable,
               stride: int = 1,
               padding: int = 0,
               use_bn: bool = True,
               bn_momentum: float = 0.9,
               prv_layer=None,
               next_layer=None):
    self.n_filters = n_filters
    self.kH = self.kW = kernel_size
    self.act_func = act_func
    self.stride = stride
    self.padding = padding
    self.use_bn = use_bn
    self.bn_mom = bn_momentum
    self.prv_layer = prv_layer
    self.next_layer = next_layer
    self.network = None  # injected by ConvolutionalNetwork after build

  def init(self) -> None:
    C_in, H_in, W_in = self.prv_layer.out_shape
    self.out_H = (H_in + 2 * self.padding - self.kH) // self.stride + 1
    self.out_W = (W_in + 2 * self.padding - self.kW) // self.stride + 1
    self.out_shape = (self.n_filters, self.out_H, self.out_W)

    # He initialisation  (optimal for ReLU)
    fan_in = C_in * self.kH * self.kW
    self.kernels = (np.random.default_rng().standard_normal(
        (self.n_filters, C_in, self.kH, self.kW)) *
                    np.sqrt(2.0 / fan_in)).astype(np.float32)
    self.biases = np.zeros(self.n_filters, dtype=np.float32)

    # Pre-compute index arrays once — reused every forward/backward call
    self._k, self._i, self._j = _im2col_indices(C_in, self.kH, self.kW,
                                                self.out_H, self.out_W,
                                                self.stride)

    # Batch Norm parameters
    if self.use_bn:
      shape = (1, self.n_filters, 1, 1)
      self.bn_gamma = np.ones(shape, dtype=np.float32)
      self.bn_beta = np.zeros(shape, dtype=np.float32)
      self.bn_run_mean = np.zeros(shape, dtype=np.float32)  # inference stats
      self.bn_run_var = np.ones(shape, dtype=np.float32)
      self.bn_eps = 1e-5
      # backward caches (populated during forward)
      self.Z_pre_bn = None
      self.x_hat = None
      self.bn_mean = None
      self.bn_var = None
      self.d_gamma = None
      self.d_beta = None

    # General caches
    self.col = None
    self.x_shape_cache = None
    self.Z = None
    self.layer_output = None
    self.layer_delta_term = None
    self.dW = None
    self.db = None

  # ── Forward pass ─────────────────────────────────────────────

  def forward(self) -> None:
    x = self.prv_layer.layer_output  # (batch, C_in, H, W)
    batch = x.shape[0]
    self.x_shape_cache = x.shape

    # 1. Pad & im2col  →  col : (batch, C_in*kH*kW, out_H*out_W)
    # np.pad(array, pad_width) expects a tuple of (before, after) pairs for every dimension of the array:
    x_pad = (np.pad(x, ((0, 0), (0, 0), (self.padding, ) * 2, (self.padding, ) * 2)) if self.padding > 0 else x)
    col = _im2col(x_pad, self._k, self._i, self._j, self.out_H, self.out_W)
    self.col = col  # saved for kernel gradient in backward

    # 2. Z = W_col @ col + b  (one matmul for the entire batch)
    W_col = self.kernels.reshape(self.n_filters, -1)  # (n_f, C*kH*kW)
      # einsum: einstain summation
    Z_col = np.einsum('fc,bcn->bfn', W_col, col, optimize=True)
    Z_col += self.biases[None, :, None]  # broadcast bias
    Z = Z_col.reshape(batch, self.n_filters, self.out_H, self.out_W)

    # 3. Optional Batch Norm  (normalise over batch × H × W per channel)
    if self.use_bn:
      Z = self._bn_forward(Z)
    self.Z = Z  # self.Z is the input to the activation function

    # 4. Activation
    self.layer_output = self.act_func(Z)

  def _bn_forward(self, Z: np.ndarray) -> np.ndarray:
    """
    Spatial Batch Norm: normalise over axes (0, 2, 3) — i.e. batch, H, W.
    Uses batch statistics during training, running stats at inference.
    """
    self.Z_pre_bn = Z  # save raw conv output for backward
    training = self.network.training if self.network else True

    if training:
      mean = Z.mean(axis=(0, 2, 3), keepdims=True)
      var = Z.var(axis=(0, 2, 3), keepdims=True)
      self.bn_mean = mean
      self.bn_var = var
      # Exponential moving average — used at inference time
      self.bn_run_mean = self.bn_mom * self.bn_run_mean + (1 -
                                                           self.bn_mom) * mean
      self.bn_run_var = self.bn_mom * self.bn_run_var + (1 - self.bn_mom) * var
    else:
      mean = self.bn_run_mean
      var = self.bn_run_var

    x_hat = (Z - mean) / np.sqrt(var + self.bn_eps)
    self.x_hat = x_hat
    return self.bn_gamma * x_hat + self.bn_beta

  # ── Backward pass ────────────────────────────────────────────

  def compute_delta_term(self, network, targets: np.ndarray) -> None:

    incoming = self.next_layer.layer_delta_term  # (batch, n_f, out_H, out_W)

    # ① Through activation:  dL/dZ = act'(Z) ⊙ incoming
    dZ = self.act_func(self.Z, derived=True) * incoming

    # ② Through Batch Norm  →  dL/dZ_raw
    if self.use_bn:
      dZ = self._bn_backward(dZ)

    # ③ Conv backward  (im2col transpose)
    batch = dZ.shape[0]
    W_col = self.kernels.reshape(self.n_filters, -1)  # (n_f, C*kH*kW)
    dZ_col = dZ.reshape(batch, self.n_filters, -1)  # (batch, n_f, out_H*out_W)

    # dW = Σ_batch  dZ_col[b] @ col[b]ᵀ
    dW_col = np.einsum('bfn,bcn->fc', dZ_col, self.col, optimize=True)
    self.dW = dW_col.reshape(self.kernels.shape)

    # db = Σ over batch and spatial positions
    self.db = dZ_col.sum(axis=(0, 2))  # (n_filters,)

    # ④ dL/d_col = W_colᵀ @ dZ_col  →  col2im  →  dL/d_input
    d_col = np.einsum('fc,bfn->bcn', W_col, dZ_col, optimize=True)
    self.layer_delta_term = _col2im(d_col, self.x_shape_cache, self._k,
                                    self._i, self._j, self.out_H, self.out_W,
                                    self.padding)

  def _bn_backward(self, d_out: np.ndarray) -> np.ndarray:

    Z = self.Z_pre_bn
    mean = self.bn_mean
    var = self.bn_var
    eps = self.bn_eps
    N = d_out.shape[0] * d_out.shape[2] * d_out.shape[3]  # batch*H*W

    # Learnable param gradients
    self.d_gamma = (d_out * self.x_hat).sum(axis=(0, 2, 3), keepdims=True)
    self.d_beta = d_out.sum(axis=(0, 2, 3), keepdims=True)

    inv_std = 1.0 / np.sqrt(var + eps)
    dx_hat = d_out * self.bn_gamma  # (batch, C, H, W)
    diff = Z - mean

    dvar = (dx_hat * diff * (-0.5) * (var + eps)**(-1.5)).sum(axis=(0, 2, 3),
                                                              keepdims=True)

    dmean = ((dx_hat * (-inv_std)).sum(axis=(0, 2, 3), keepdims=True) + dvar *
             (-2.0 * diff).sum(axis=(0, 2, 3), keepdims=True) / N)

    dZ = dx_hat * inv_std + dvar * 2.0 * diff / N + dmean / N
    return dZ

  def update_weights(self, lr: float) -> None:
    self.kernels -= lr * self.dW
    self.biases -= lr * self.db
    if self.use_bn:
      self.bn_gamma -= lr * self.d_gamma
      self.bn_beta -= lr * self.d_beta
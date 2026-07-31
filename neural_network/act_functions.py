import numpy as np

class ActivationFunction:
  def identity(Z: np.ndarray, derived: bool= False) -> np.ndarray:
    if derived:
      return 1
    else:
      return Z

  # Rectified linear unit
  def ReLU(Z: np.ndarray, derived: bool= False) -> np.ndarray:
    if derived:
      return (Z > 0).astype(float)
    else:
      return np.maximum(0, Z)
  
  def sigmoid(Z: np.ndarray, derived: bool= False) -> np.ndarray:
    sigmoid = 1 / (1 + np.exp(-Z))
    if derived:
      return sigmoid*(1-sigmoid)
    else:
      return sigmoid
    
  # Due to the learning purpose, we wont use the shortcut for loss/softmax derivative
  def softmax(Z: np.ndarray, derived: bool= False) -> np.ndarray:
    # Preventing overflow
    Z = Z - np.max(Z, axis= 1, keepdims= True) 
    numerator = np.exp(Z)
    denominator = np.sum(numerator, axis= 1, keepdims= True)
    S = numerator / denominator
    if derived: # Jacobian form
      # Reshape S to (batch, #neuron, 1) to allow broadcasting
      s_reshaped = S[:, :, None]
      # Identity matrix scaled by S_i
      diag_part = s_reshaped * np.eye(S.shape[1])[None, :, :]
      # outer product S_i * S_j
      outer_part = s_reshaped * s_reshaped.transpose(0, 2, 1)      
      return diag_part - outer_part
    else:
      return S

  # Hyperbolic tangent
  def Tanh(Z: np.ndarray, derived= False) -> np.ndarray:
    tanh = 2 / (1 + np.exp(-2*Z)) - 1
    if derived:
      return 1 - tanh**2
    else:
      return tanh
  
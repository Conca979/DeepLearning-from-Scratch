import numpy as np

# Due to the learning purpose, we wont use the shortcut for loss/softmax derivative
class LossFunction:
  # Catergorical cross-entropy loss, targets are one-hot encoded matrix
  def cc_loss(predicted_values: np.ndarray, targets: np.ndarray, derived: bool= False) -> np.ndarray | np.float64:
    batch = predicted_values.shape[0]
    if derived:
      # We add a tiny value to prevent divission by 0
      return (-1 / batch) * targets / (predicted_values + 1e-15)
    else:
      class_indices = np.argmax(targets, axis= 1)
      return (-1 / batch)*np.sum(np.log(predicted_values[np.arange(batch), class_indices] + 1e-15))

  # Binary classification
  def Binary_loss():
    pass

  def MSE(predicted_values: np.ndarray, targets: np.ndarray, derived: bool= False) -> np.ndarray | np.float64:
    batch = predicted_values.shape[0]
    if derived:
      return (1 / batch) * (predicted_values - targets)
    else:
      return (1 / (2*batch)) * np.sum((predicted_values - targets)**2)

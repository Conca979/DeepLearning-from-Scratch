from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
  def identity(Z: np.ndarray, derived: bool= False) -> np.ndarray:
    if derived:
      return 1
    else:
      return Z

  # Rectified linear unit
  def ReLU(Z: np.ndarray, derived: bool= False) -> np.ndarray:
    if derived:
      return np.maximum(0, np.minimum(1, Z))
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

class BasicNeuralNetwork:
  class HiddenLayer:
    def __init__(self,
                l_rate: float,
                n_neuron: int, 
                act_func: Callable[[np.ndarray, bool], np.ndarray],
                prv_layer= None,
                next_layer= None,
                weights_initialization: str= "He"): # Xavier Initialization (Glorot): used for sigmid, tanh, ...
                                                    # He Initialization (Kaiming): used for ReLU (most common case)):
      self.w_init_method = weights_initialization
      self.learning_rate = l_rate
      self.n_neuron = n_neuron
      self.act_func = act_func
      self.prv_layer = prv_layer
      self.next_layer = next_layer

    # Must be guaranteed that the neural network is initialized before initializing each layer
    def init(self) -> None:
      _temp = np.random.default_rng().standard_normal(size= (self.n_neuron, self.prv_layer.n_neuron))
      if self.w_init_method == "He":
        self.weights = _temp * np.sqrt(2 / self.prv_layer.n_neuron)
      else:
        self.weights = _temp * np.sqrt(1 / self.prv_layer.n_neuron)
      self.biases = np.zeros(self.n_neuron)
      self.Z = None
      self.layer_output = None
      self.layer_delta_term = None

    def forward(self) -> None:
      Z = self.prv_layer.layer_output @ self.weights.T + self.biases
      self.Z = Z
      self.layer_output = self.act_func(self.Z)

    def compute_delta_term(self, network: object) -> None:
      if self.next_layer == None:
        if self.act_func.__name__ == "softmax":
          # Loss: (batch, neuron) ; softmax: (batch, neuron, neuron)
          # Extend loss'dim to (batch, 1, neuron) for broadcasting
          grad_loss = network.compute_loss(derived= True)[:, None, :]
          solf_jacobian = self.act_func(self.Z, derived= True)
          # Shink down the middle dim
          delta = (grad_loss @ solf_jacobian).reshape(grad_loss.shape[0], grad_loss.shape[-1])
          self.layer_delta_term = delta
        else:
          grad_loss = network.compute_loss(derived= True)
          # Element-wise
          delta = self.act_func(self.Z, derived= True)*grad_loss
          self.layer_delta_term = delta
      else:
        delta = self.act_func(self.Z, derived= True)*(self.next_layer.layer_delta_term @ self.next_layer.weights)
        self.layer_delta_term = delta

  #-----------------------------------------

  class InputLayer:
    def __init__(self, input_layer: np.ndarray):
      self.layer_input = input_layer
      self.layer_output = input_layer
      self.n_neuron = input_layer.shape[1]

    def forward(self, predict_input: np.ndarray= None) -> np.ndarray:
      if predict_input is not None:
        self.layer_output = predict_input
      else: 
        self.layer_output = self.layer_input
  #-----------------------------------------

  def __init__(self,
               layers_init: tuple[list[int], list[Callable]], # int: number of neurons in this layer
                                                 # function: activation fucntion used in this layer
               training_set: tuple[np.ndarray, np.ndarray],  # shape[0]: X features
                                                             # [1]: One-hot encoded targets
               loss_func: Callable,
               leanring_rate: float= 0.01,
               epsilon: float= 0.0001,
               test_set: tuple[np.ndarray, np.ndarray]= None,
               iteration_event_trigger= -1):
    # Initializing network
    self.iet = iteration_event_trigger
    self.network_layers = []
    self.network_loss = 0
    self.epsilon = epsilon
    self.learning_rate = leanring_rate
    self.iterations = 0
    self.loss_func = loss_func
    # Splitting preprocessed X and Y for quick matrix operations
    self.n_layer = layers_init[0]
    self.act_funcs = layers_init[1]
    self.x_train = training_set[0]
    self.y_train = training_set[1]
    self.batch = self.x_train.shape[0]
    if test_set:
      self.x_test = test_set[0]
      self.y_test = test_set[1]
    self._init_network() 

  def _init_network(self) -> None:
    self.network_layers.append(self.InputLayer(input_layer= self.x_train))
  # Forward init
    for n_count, act_func in zip(self.n_layer, self.act_funcs):
      layer = self.HiddenLayer(n_neuron= n_count,
                               l_rate= self.learning_rate,
                               act_func= act_func,
                               prv_layer= self.network_layers[-1])
      self.network_layers.append(layer)
  # Backward init
    indexes = len(self.network_layers)
    for i in range(1, indexes):
      if i != indexes - 1:
        self.network_layers[i].next_layer = self.network_layers[i+1]
      self.network_layers[i].init()

    print("--- Neural network is initialized successfully ---")

  def compute_loss(self, derived: bool= False) -> np.ndarray:
    predicted_vals = self.network_layers[-1].layer_output
    loss = self.loss_func(predicted_values= predicted_vals,
                                        targets= self.y_train,
                                        derived= derived)
    if not derived:
      self.network_loss = loss
    return loss

  def _forward_propagation(self, predict_input: np.ndarray= None) -> None:
    self.network_layers[0].forward(predict_input)
    for layer in self.network_layers[1:]:
      layer.forward()

  def _backward_propagation(self) -> None:
    for layer in self.network_layers[1:]:
      layer.weights -= self.learning_rate*(layer.layer_delta_term.T @ layer.prv_layer.layer_output)
      layer.biases -= self.learning_rate*np.sum(layer.layer_delta_term, axis= 0)

  def _compute_delta_term(self) -> None:
    for layer in self.network_layers[-1:0:-1]:
      layer.compute_delta_term(network= self)

  def fit_model(self) -> None:
    print("--- Start training ---")
    old_loss = 0
    while True:
      self._forward_propagation()
      new_loss = self.compute_loss()
      self._compute_delta_term()
      # --------------
      if self.iterations % self.iet == 0:
        print(self.iterations, new_loss)
      # --------------
      self.iterations += 1
      loss_change = abs((new_loss - old_loss) / new_loss)
      if loss_change < self.epsilon:
        break
      else:
        old_loss = self.network_loss
        self._backward_propagation()
        continue
    
    print(f"Completed after {self.iterations} iterations")

  def predict(self, predict_input: np.ndarray) -> np.ndarray:
    self._forward_propagation(predict_input= predict_input)
    result = self.network_layers[-1].layer_output
    return result
  
  def evaluate(self):
    model = self.loss_func.__name__
    if model == "cc_loss":
      act_classes = np.argmax(self.y_train, axis= 1)
      predictions = self.network_layers[-1].layer_output
      pred_classes = np.argmax(predictions, axis= 1)
      precision = np.mean(pred_classes == act_classes) * 100
      return precision
    elif model == "MSE":
      act_mean = np.mean(self.y_train)
      sst = np.sum((self.y_train - act_mean)**2)
      ssr = np.sum((self.y_train - self.network_layers[-1].layer_output)**2)

      return (1 - ssr/sst)*100
  
  def predict_vs_target(self):
    model = self.loss_func.__name__
    if model == "cc_loss":
      pass
    elif model == "MSE":
      pred_vals = self.network_layers[-1].layer_output.flatten()
      act_vals = self.y_train.flatten()
      plt.scatter(act_vals, pred_vals, c= 'r')
      plt.ylabel("Predicted Values")
      plt.xlabel("Actual Values")
      plt.title("Prediction vs targets")

      upper = max(max(pred_vals), max(act_vals))
      lower = min(min(pred_vals), min(act_vals))
      plt.plot([lower, upper], [lower, upper], c= "b", alpha= 0.5)
      plt.show()
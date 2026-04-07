import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
  # Rectified linear unit
  def ReLU(self, input, derived= False):
    if derived:
      return np.maximum(0, np.minimum(1, input))
    else:
      return np.maximum(0, input)
  
  def sigmoid(self, input, derived= False):
    sigmoid = 1 / (1 + np.e**-input)
    if derived:
      return sigmoid*(1-sigmoid)
    else:
      return sigmoid
    
  def softmax(self, input, derived= True):
    if derived:
      # Hope that we dont have to use this in jacobian matrix form
      # We would have simplified it in the delta term of the output layer ∂L/∂f(Z) . ∂Z/∂W, aka
      # ∂L/∂Z = ∂L/∂f(Z) . ∂Z/∂W = predicted_y - actual_y
      pass
    else:
      numerator = np.e**input
      denominator = np.sum(numerator)
      return numerator / denominator

  # Hyperbolic tangent
  def Tanh(self, input, derived= False):
    tanh = 2 / (1 + np.e**-(2*input)) - 1
    if derived:
      return 1 - tanh**2
    else:
      return tanh
    
class LossFunction:
  #   
  def CNN_loss(self, predicted_values, one_hot_encoded_values, derived= False):
    if derived:
      pass
    else:
      # The "True" probability indexes
      b = np.argmax(one_hot_encoded_values, axis= 1)
      return -1/predicted_values.shape[0]*np.sum(np.log(predicted_values[range(predicted_values.shape[0], b)]))

  # Binary classification
  def Binary_loss(self):
    pass

# Aim for multi-class classification
class BasicNeuralNetwork:
  class HiddenLayer:
    def __init__(self,
                l_rate,
                n_neuron, 
                act_func,
                prv_layer= None,
                next_layer= None):
      self.learning_rate = l_rate
      self.n_neuron = n_neuron
      self.act_func = act_func
      self.prv_layer = prv_layer
      self.next_layer = next_layer

    # Must be guaranteed that the neural networ is initialized before calling this method
    def init(self):
      self.weights = np.random.default_rng().standard_normal(size= (self.n_neuron, self.prv_layer.n_neuron))
      self.biases = np.zeros(self.n_neuron)
      # The input of this layer is the ouput of the previous layer
      self.layer_input = self.prv_layer.layer_output
      self.layer_output = self.forward()
      self.layer_delta_term = None

    def forward(self):
      pre_activated_output = np.dot(self.layer_input, self.weights.T) + self.biases
      self.layer_output = self.act_func(pre_activated_output)

    def delta_term(self):
      self.layer_delta_term = self.layer_output*self.next_layer.layer_delta_term

  #-----------------------------------------

  class InputLayer:
    def __init__(self, input_layer):
      self.layer_output = input_layer

  #-----------------------------------------

  class OutPutLayer:
    # the δ term of this output layer is computed by ∂L/∂z = ∂L/∂H . ∂H/∂Z
    # Where ∂H/∂Z is the derived form of the pre_layer's act_f

    def __init__(self, prv_layer, act_func, loss_func, expected_outputs):
      self.expected_outputs = expected_outputs
      self.prv_layer = prv_layer
      self.act_func = act_func
      self.loss_func = loss_func
      self.delta = None

    def delta_term(self): # result in matrix form
      # The simplified form of ∂L/∂Z = ∂L/∂f(Z) . ∂Z/∂W = predicted_y - expected_y
      result = self.prv_layer.layer_ouput - self.expected_outputs
      return result
      
  #-----------------------------------------

  def __init__(self,
               n_layer, # Ordered number of neuron for all hidden layers
               training_set,  # [0]: X features
                              # [1]: One-hot encoded outcomes
               test_set= None):
    # Splitting X test and Y test for matrix operations
    self.n_layer = n_layer
    self.x_train = training_set[0]
    self.y_train = training_set[1]
    if test_set:
      self.x_test = test_set[0]
      self.y_test = test_set[1]

    # 
    self.network_layers = []

  def _init_network(self):
    self.network_layers.append(self.InputLayer(input_layer= self.x_train))
  # Forward init
    for n_count in self.n_layer:
      layer = self.HiddenLayer(n_neuron= n_count,
                               prv_layer= self.network_layers[-1])
      self.network_layers.append(layer)
    # Add the ouput layer
    output_layer = self.OutPutLayer(prv_layer= self.network_layers[-1],
                                    expected_outputs= self.y_train)
    self.network_layers.append(output_layer)
    # Layer count
    self.layer_count = len(self.network_layers)
  # Backward init
    for i in range(1, self.layer_count - 1):
      self.network_layers[i].next_layer = self.network_layers[i+1]
      self.network_layers[i].init()

  def _forward_propagation(self):
    for hd_layer_index in range(1, self.layer_count - 1):
      self.network_layers[hd_layer_index].forward()

  def _backward_propagation(self):
    for hd_layer_index in range(1, self.layer_count - 1):
      pass
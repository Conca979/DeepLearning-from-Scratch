import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
  # Rectified linear unit
  def ReLU(input, derived= False):
    if derived:
      return np.maximum(0, np.minimum(1, input))
    else:
      return np.maximum(0, input)
  
  def sigmoid(input, derived= False):
    sigmoid = 1 / (1 + np.e**-input)
    if derived:
      return sigmoid*(1-sigmoid)
    else:
      return sigmoid
    
  # Hyperbolic tangent
  def Tanh(input, derived= False):
    tanh = 2 / (1 + np.e**-(2*input)) - 1
    if derived:
      return 1 - tanh**2
    else:
      return tanh

# Aim for multi-class classification
class BasicNeuralNetwork:
  class Layer:
    def __init__(self,
                l_rate, 
                n_neuron, 
                prv_layer= None,
                next_layer= None):
      self.learning_rate = l_rate
      self.n_neuron = n_neuron
      self.prv_layer = prv_layer
      self.next_layer = next_layer

    # Must be guaranteed that the neural networ is initialized before calling this method
    def init(self):
      self.weights = np.random.default_rng().standard_normal(size= (self.n_neuron, self.prv_layer.n_neuron))
      self.biases = np.zeros(self.n_neuron)
      # The input of this layer is the ouput of the previous layer
      self.layer_input = self.prv_layer.output_layer
      self.output_layer = self.forward()

    def forward(self):
      new_ouput = np.dot(self.layer_input, self.weights.T) + self.biases
      self.output_layer = new_ouput

    def delta_term(self):
      pass

  #-----------------------------------------

  class InputLayer():
    def __init__(self, input):
      self.output_layer = input

  #-----------------------------------------

  class OutPutLayer():
    pass

  #-----------------------------------------

  class HiddenLayer(Layer):
    def __init__(self, n_neuron, prv_layer):
      super().__init__(n_neuron= n_neuron,
                       prv_layer= prv_layer)
      
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
    self.network_layers.append(self.InputLayer(self.x_train))
    # Forward init
    for n_count in self.n_layer:
      layer = self.HiddenLayer(n_neuron= n_count,
                               prv_layer= self.network_layers[-1])
      self.network_layers.append(layer)
    # Layer count for quick layer iteration
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
class SGD:

  def __init__(self, lr=0.01):
    self.lr = lr

  def step(self, network_layers):
    for layer in network_layers[1:]:
      layer.update_weights(self.lr)

import neural_network as nn
import numpy as np


# Data preprocessing
data_set = np.loadtxt('data/Iris.csv', delimiter=',', dtype=str, skiprows=1, usecols= range(1,6))
targets = list(set(data_set[:,-1]))
x_train = data_set[:, :-1].astype(float)
y_train = data_set[:, -1]

# One-hot encoded
unique, int_encoded = np.unique(y_train, return_inverse=True)
y_train = np.eye(unique.shape[0])[int_encoded]

#--------------------------------
act = nn.ActivationFunction
# --- How Many Neurons per Layer? ---
# The "In-Between" Rule: The most common choice is a number between the size of the input layer and the size of the output layer.
# The 2/3 Rule: A classic heuristic is: (Number of Inputs * 2/3) + Number of Outputs
# The Funnel Architecture: If you use multiple hidden layers, you generally want the network to "compress" the information as it moves forward. Make the first hidden layer the largest, and decrease the size for subsequent layers.
t = int(x_train.shape[1] * 2/3) + unique.shape[0]
n_of_neuron_per_hidden_layer_respectively = [t, unique.shape[0]]
acts = [act.ReLU, act.softmax]
model_inits = [n_of_neuron_per_hidden_layer_respectively, 
               acts]

#--------------------------------
loss_func = nn.LossFunction.cc_loss

model = nn.BasicNeuralNetwork(layers_init= model_inits,
                              loss_func= loss_func,
                              training_set= (x_train, y_train),
                              leanring_rate= 0.01,
                              epsilon= 0.00001,
                              test_set= None,
                              iteration_event_trigger= 1
                              )

model.fit_model()
import neural_network as nn
import numpy as np


# Data preprocessing
data_set = np.loadtxt('data/Iris.csv', delimiter=',', dtype=str, skiprows=1, usecols= range(1,6))
x_train = data_set[:, :-1].astype(float)
y_train = data_set[:, -1]

# One-hot encoded
unique, int_encoded = np.unique(y_train, return_inverse=True)
one_hot_encoded = np.eye(unique.shape[0])
y_train = one_hot_encoded[int_encoded]
print(f"{unique}, \n{np.eye(unique.shape[0])}")

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
                              iteration_event_trigger= 1000
                              )

model.fit_model()

# --- Prediction ----
predict_input = [[5.9,3.0,5.1,1.8],
                 [5.5,3.5,1.3,0.2],
                 [4.4,2.9,1.4,0.2]
                 ]
predicted_target_probabilities = model.predict(predict_input= predict_input)
print(f"--- Predictions ---\n [sample -> prediction = probability]")
for predict, target_p in zip(predict_input, predicted_target_probabilities):
  arg_max = np.argmax(target_p)
  print(f"{predict} -> {unique[arg_max]} = {round(target_p[arg_max]*100, 4)}%")
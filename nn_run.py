import neural_network as nn
import numpy as np
import csv

with open('data/StudentScore.csv', newline='') as f:
  reader = csv.DictReader(f)
  rows = list(reader)

# --- Helper functions ---
  # Binary encoding
def binary_encode(rows, col, positive_value):
  return np.array([1 if r[col] == positive_value else 0 for r in rows], dtype=float)

  # Extracting numeric columns 
def get_col(rows, col):
  return np.array([float(r[col]) for r in rows])

  # One-hot encoding for ulti-class columns
def onehot_encode(rows, col):
  values = sorted(set(r[col] for r in rows))
  mapping = {v: i for i, v in enumerate(values)}
  matrix = np.zeros((len(rows), len(values)))
  for i, r in enumerate(rows):
    matrix[i, mapping[r[col]]] = 1
  return matrix
# --------------------------

gender = binary_encode(rows, 'gender', 'female')
lunch = binary_encode(rows, 'lunch', 'standard')
test_prep = binary_encode(rows, 'test preparation course', 'completed')

race_oh = onehot_encode(rows, 'race/ethnicity') # 5 groups
edu_oh = onehot_encode(rows, 'parental level of education')  # 6 levels

math = get_col(rows, 'math score')
reading = get_col(rows, 'reading score')
writing = get_col(rows, 'writing score')

# Assembling full data matrix
data_set = np.column_stack([gender, lunch, test_prep, race_oh, edu_oh, reading, math, writing])

# Scaling features
means = np.mean(data_set, axis=0)
stds = np.std(data_set, axis=0)
stds[stds == 0] = 1 # Avoid division by zero on binary cols
data_set = (data_set - means) / stds

# Slitting data
split_index = int(0.8*len(data_set))
train_set = data_set[:split_index,:]
test_set = data_set[split_index:, :]

x_train = train_set[:, :-1]
y_train = train_set[:, -1][:, None]

x_test = test_set[:, :-1]
y_test = test_set[:, -1][:, None]

#--------------------------------
act = nn.ActivationFunction
# --- How Many Neurons per Layer? ---
# The "In-Between" Rule: The most common choice is a number between the size of the input layer and the size of the output layer.
# The 2/3 Rule: A classic heuristic is: (Number of Inputs * 2/3) + Number of Outputs
# The Funnel Architecture: If you use multiple hidden layers, you generally want the network to "compress" the information as it moves forward. Make the first hidden layer the largest, and decrease the size for subsequent layers.
t = int(x_train.shape[1] * 2/3) + 1
n_of_neuron_per_hidden_layer_respectively = [t, 1]
acts = [act.ReLU, act.identity]
model_inits = [n_of_neuron_per_hidden_layer_respectively, 
               acts]

#--------------------------------
loss_func = nn.LossFunction.MSE

model = nn.BasicNeuralNetwork(layers_init= model_inits,
                              loss_func= loss_func,
                              training_set= (x_train, y_train),
                              test_set= (x_test, y_test),
                              leanring_rate= 0.01,
                              epsilon= 0.00001,
                              iteration_event_trigger= 10000
                              )

model.fit_model()
print(f"Model goodness of fit: {model.evaluate()}")
model.predict_vs_target()


# ----- Below is the model for Advertising data_set

# # Data preprocessing
# data_set = np.loadtxt('data/Iris.csv', delimiter=',', dtype=str, skiprows=1, usecols= range(1,6))
# x_train = data_set[:, :-1].astype(float)
# y_train = data_set[:, -1]

# # One-hot encoded
# unique, int_encoded = np.unique(y_train, return_inverse=True)
# one_hot_encoded = np.eye(unique.shape[0])
# y_train = one_hot_encoded[int_encoded]
# print(f"{unique}, \n{np.eye(unique.shape[0])}")

# #--------------------------------
# act = nn.ActivationFunction
# # --- How Many Neurons per Layer? ---
# # The "In-Between" Rule: The most common choice is a number between the size of the input layer and the size of the output layer.
# # The 2/3 Rule: A classic heuristic is: (Number of Inputs * 2/3) + Number of Outputs
# # The Funnel Architecture: If you use multiple hidden layers, you generally want the network to "compress" the information as it moves forward. Make the first hidden layer the largest, and decrease the size for subsequent layers.
# t = int(x_train.shape[1] * 2/3) + unique.shape[0]
# n_of_neuron_per_hidden_layer_respectively = [t, unique.shape[0]]
# acts = [act.ReLU, act.softmax]
# model_inits = [n_of_neuron_per_hidden_layer_respectively, 
#                acts]

# #--------------------------------
# loss_func = nn.LossFunction.cc_loss

# model = nn.BasicNeuralNetwork(layers_init= model_inits,
#                               loss_func= loss_func,
#                               training_set= (x_train, y_train),
#                               leanring_rate= 0.01,
#                               epsilon= 0.00001,
#                               test_set= None,
#                               iteration_event_trigger= 1000
#                               )

# model.fit_model()

# # --- Prediction ----
# predict_input = [[5.9,3.0,5.1,1.8],
#                  [5.5,3.5,1.3,0.2],
#                  [4.4,2.9,1.4,0.2]
#                  ]
# predicted_target_probabilities = model.predict(predict_input= predict_input)
# print(f"--- Predictions ---\n [sample -> prediction = probability]")
# for predict, target_p in zip(predict_input, predicted_target_probabilities):
#   arg_max = np.argmax(target_p)
#   print(f"{predict} -> {unique[arg_max]} = {round(target_p[arg_max]*100, 4)}%")
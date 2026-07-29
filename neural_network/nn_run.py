from keras.datasets import mnist  # just for data loading
import test as nn
import numpy as np
import time

(x_train, y_train_labels), (x_test, y_test_labels) = mnist.load_data()
# Flatten 28x28 images → 784 features, normalize to [0, 1]
x_train = np.asarray(x_train.reshape(60000, 784) / 255.0, dtype=np.float32)
x_test  = np.asarray(x_test.reshape(10000, 784)  / 255.0, dtype=np.float32)

# One-hot encode labels (10 classes: digits 0–9)
def onehot(labels, n_classes=10):
    m = np.zeros((len(labels), n_classes))
    m[np.arange(len(labels)), labels] = 1
    return np.asarray(m, dtype=np.float32)

y_train = onehot(y_train_labels)
y_test  = onehot(y_test_labels)

#--------------------------------
act = nn.ActivationFunction
# --- How Many Neurons per Layer? ---
# The "In-Between" Rule: The most common choice is a number between the size of the input layer and the size of the output layer.
# The 2/3 Rule: A classic heuristic is: (Number of Inputs * 2/3) + Number of Outputs
# The Funnel Architecture: If you use multiple hidden layers, you generally want the network to "compress" the information as it moves forward. Make the first hidden layer the largest, and decrease the size for subsequent layers.
model_inits = [[512, 256, 10], 
               [act.ReLU, act.ReLU, act.softmax]]
loss_func = nn.LossFunction.cc_loss

#--------------------------------
start = time.time()

model = nn.BasicNeuralNetwork(layers_init= model_inits,
                              loss_func= loss_func,
                              training_set= (x_train, y_train),
                              test_set= (x_test, y_test),
                              batch= 128,
                              leanring_rate= 0.01,
                              epsilon= 0.00001,
                              epoch_limit= 20,
                              iteration_event_trigger= 100
                              )

model.fit_model()

end = time.time()

print("-"*30)
print(f"{"(!Forced)" if model.epoch == model.epoch_limit else ''} Completed after {model.iterations} iterations, {model.epoch} epochs")
print(f"Model is trained in {end-start:.6g}s, batch size of {model.batch} samples")
print(f"Evaluated: {model.evaluate()}%")
model.predict_vs_target()
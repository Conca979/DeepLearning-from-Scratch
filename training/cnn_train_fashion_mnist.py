import numpy as np
import time
import struct
import sys
import os

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from deep_learning import Network, Conv2D, MaxPool2D, Flatten, Dense, InputLayer, ActivationFunction, LossFunction

# -------------
# 1. Load data
# -------------

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "weights", "cnn_fashion_mnist_weights.npz")

def load_idx_images(filename):
  with open(filename, 'rb') as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    return np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

def load_idx_labels(filename):
  with open(filename, 'rb') as f:
    magic, num = struct.unpack(">II", f.read(8))
    return np.fromfile(f, dtype=np.uint8)

def onehot(labels, n=10):
  m = np.zeros((len(labels), n), dtype=np.float32)
  m[np.arange(len(labels)), labels] = 1
  return m

print("Loading Fashion MNIST ...")
data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "fashion_mnist")
x_train = (load_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte")).reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)
x_test  = (load_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte")).reshape(-1, 1, 28, 28)  / 255.0).astype(np.float32)
y_train = onehot(load_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte")))
y_test  = onehot(load_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte")))
print(f"  x_train: {x_train.shape}  x_test: {x_test.shape}")

# -------------
# 2. Build model
# -------------

act = ActivationFunction

layers = [
  InputLayer(x_train),
  Conv2D(16, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  MaxPool2D(2, 2),
  Conv2D(32, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  MaxPool2D(2, 2),
  Flatten(),
  Dense(256, act_func=act.ReLU, use_dropout=True, drop_rate=0.3),
  Dense(10,  act_func=act.softmax, use_dropout=False, drop_rate=0.0)
]

model = Network(
  layers=layers,
  training_set=(x_train, y_train),
  test_set=(x_test, y_test),
  loss_func=LossFunction.cc_loss,
  batch=512,
  learning_rate=0.1,
  epsilon=1e-8,
  epoch_limit=5,
  iteration_event_trigger=1,
)

# -------------
# 3. Train
# -------------

print("\nTraining (this may take 20-40 min on CPU) ...\n")
t0 = time.time()
model.fit_model()
elapsed = time.time() - t0
print(f"\nTraining done in {elapsed / 60:.1f} min")

# -------------
# 4. Evaluate & save
# -------------

accuracy = model.evaluate()
print(f"Test accuracy: {accuracy:.2f}%")

model.save_weights(WEIGHTS_FILE, accuracy=accuracy)
from PIL import Image
import numpy as np
import time
import pandas as pd
import io
import sys
import os

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from deep_learning import Network, Dense, InputLayer, ActivationFunction, LossFunction

# -------------
# 1. Load data
# -------------

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "weights", "nn_mnist_weights.npz")

def load_mnist_parquet(path):
  df = pd.read_parquet(path)
  images, labels = [], []
  for row in df.itertuples():
    img_bytes = row.image['bytes']
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    images.append(np.array(img, dtype=np.uint8))
    labels.append(row.label)
  return np.array(images), np.array(labels)

def onehot(labels, n=10):
  m = np.zeros((len(labels), n), dtype=np.float32)
  m[np.arange(len(labels)), labels] = 1
  return m

print("Loading MNIST ...")
train_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist", "train.parquet")
test_path  = os.path.join(os.path.dirname(__file__), "..", "data", "mnist", "test.parquet")
x_train_raw, y_train_lbl = load_mnist_parquet(train_path)
x_test_raw,  y_test_lbl  = load_mnist_parquet(test_path)

# Flatten 28x28 -> 784, normalize to [0, 1]
x_train = np.asarray(x_train_raw.reshape(-1, 784) / 255.0, dtype=np.float32)
x_test  = np.asarray(x_test_raw.reshape(-1, 784)  / 255.0, dtype=np.float32)
y_train = onehot(y_train_lbl)
y_test  = onehot(y_test_lbl)
print(f"  x_train: {x_train.shape}  x_test: {x_test.shape}")

# -------------
# 2. Build model
# -------------

act = ActivationFunction

layers = [
  InputLayer(x_train),
  Dense(512, act_func=act.ReLU),
  Dense(256, act_func=act.ReLU),
  Dense(128, act_func=act.ReLU),
  Dense(10,  act_func=act.softmax)
]

epoch_limit = 50
gamma = 0.01**(1/epoch_limit)

print(f'--- Learning rate decay = {gamma:.5f} for epoch limit of {epoch_limit} --')

model = Network(
  layers=layers,
  training_set=(x_train, y_train),
  test_set=(x_test, y_test),
  loss_func=LossFunction.cc_loss,
  batch=64,
  learning_rate=0.01,
  lr_decay=gamma,
  epsilon=1e-6,
  epoch_limit=epoch_limit,
  iteration_event_trigger=100,
  eval_every=3
)

# -------------
# 3. Train
# -------------

print("\nTraining (this may take a minute) ...\n")
t0 = time.time()
model.fit_model()
elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s")

# -------------
# 4. Evaluate & save
# -------------

accuracy = model.evaluate()
print(f"Test accuracy: {accuracy:.2f}%")

res = '...'
while res != 'n':
  res = input("Want to save the pre-trained weights? -> 'y' for yes 'n' for no - ")
  if res == 'y':
    model.save_weights(WEIGHTS_FILE, accuracy=accuracy)
    break

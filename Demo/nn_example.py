"""
Draw-and-Predict Demo
=====================
Trains the neural network on MNIST, then opens a drawing canvas.
Draw a digit -> click "Predict" -> see the model's prediction.
"""

import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import numpy as np
import pandas as pd
import io
import time
import os

import sys

sys.path.append(
  os.path.abspath(
    os.path.join(
      os.path.dirname(__file__), 
      '..', 
      'src'
      )
    )
  )

from deep_learning import Network, Dense, InputLayer, ActivationFunction, LossFunction

# =====================================================================
#  1.  Train the model
# =====================================================================
print("Loading MNIST dataset ...")
def load_mnist_parquet(path):
  df = pd.read_parquet(path)
  images, labels = [], []
  for row in df.itertuples():
    from PIL import Image
    img_bytes = row.image['bytes']
    img = Image.open(io.BytesIO(img_bytes)).convert('L')
    images.append(np.array(img, dtype=np.uint8))
    labels.append(row.label)
  return np.array(images), np.array(labels)

train_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist", "train.parquet")
test_path = os.path.join(os.path.dirname(__file__), "..", "data", "mnist", "test.parquet")
(x_train, y_train_labels) = load_mnist_parquet(train_path)
(x_test, y_test_labels) = load_mnist_parquet(test_path)
# Flatten 28x28 images -> 784 features, normalize to [0, 1]
x_train = np.asarray(x_train.reshape(60000, 784) / 255.0, dtype=np.float32)
x_test = np.asarray(x_test.reshape(10000, 784) / 255.0, dtype=np.float32)


# One-hot encode labels (10 classes: digits 0–9)
def onehot(labels, n_classes=10):
  m = np.zeros(
    (len(labels),
    n_classes)
    )
  m[np.arange(len(labels)), labels] = 1
  return np.asarray(m, dtype=np.float32)


y_train = onehot(y_train_labels)
y_test = onehot(y_test_labels)

act = ActivationFunction
layers = [
  InputLayer(x_train),
  Dense(512, act_func=act.ReLU),
  Dense(256, act_func=act.ReLU),
  Dense(128, act_func=act.ReLU),
  Dense(10, act_func=act.softmax)
]
loss_func = LossFunction.cc_loss

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "weights", "nn_weights.npz")

model = Network(
    layers=layers,
  loss_func=loss_func,
  training_set=(x_train, y_train),
  test_set=(x_test, y_test),
  batch=64,
  learning_rate=0.01,
  epsilon=0.000001,
  epoch_limit=5,
  iteration_event_trigger=100,
)

if os.path.exists(WEIGHTS_FILE):
  print(f"\nFound saved weights -> loading {WEIGHTS_FILE}  (skip training)")
  model.load_weights(WEIGHTS_FILE)
  accuracy = model.evaluate()
else:
  print("Training model …  (this may take a minute)")
  start = time.time()
  model.fit_model()
  elapsed = time.time() - start
  accuracy = model.evaluate()
  print(
    f"Training complete in {elapsed:.1f}s  —  Test accuracy: {accuracy:.2f}%"
  )
  model.save_weights(WEIGHTS_FILE)

# =====================================================================
#  2.  Drawing UI
# =====================================================================

# ── Colour palette ───────────────────────────────────────────────────
BG_DARK = "#0f0f1a"
PANEL_BG = "#1a1a2e"
CANVAS_BG = "#111122"
ACCENT = "#7c3aed"  # purple
ACCENT_HOVER = "#9f67ff"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
SUCCESS = "#22d3ee"  # cyan
BAR_BG = "#2a2a40"
BAR_FILL = "#7c3aed"

CANVAS_PX = 280  # display canvas size (10× MNIST)
MNIST_PX = 28
BRUSH_R = 10  # brush radius on the display canvas


class DrawApp:

  def __init__(self, root: tk.Tk):
    self.root = root
    root.title("Neural Network — Digit Recogniser")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    # ── Fonts ────────────────────────────────────────────────────
    self.font_title = tkfont.Font(family="Segoe UI", size=16, weight="bold")
    self.font_digit = tkfont.Font(family="Consolas", size=72, weight="bold")
    self.font_label = tkfont.Font(family="Segoe UI", size=11)
    self.font_btn = tkfont.Font(family="Segoe UI", size=11, weight="bold")
    self.font_small = tkfont.Font(family="Segoe UI", size=9)
    self.font_bar = tkfont.Font(family="Consolas", size=9)

    # ── PIL backing image (white-on-black, same as MNIST) ────────
    self.pil_image = Image.new("L", (CANVAS_PX, CANVAS_PX), 0)
    self.pil_draw = ImageDraw.Draw(self.pil_image)

    self._build_ui()

  # ─────────────────────────────────────────────────────────────────
  #  UI layout
  # ─────────────────────────────────────────────────────────────────
  def _build_ui(self):
    # Title bar
    title_frame = tk.Frame(self.root, bg=BG_DARK, pady=12)
    title_frame.pack(fill="x")
    tk.Label(title_frame,
             text="✦  Digit Recogniser",
             font=self.font_title,
             fg=ACCENT,
             bg=BG_DARK).pack()
    tk.Label(
        title_frame,
        text=f"Model accuracy: {accuracy:.2f}%  ·  Draw a digit below",
        font=self.font_small,
        fg=TEXT_MUTED,
        bg=BG_DARK,
    ).pack()

    # Main content: canvas (left) + results panel (right)
    main = tk.Frame(self.root, bg=BG_DARK, padx=16, pady=4)
    main.pack()

    # ── Canvas ───────────────────────────────────────────────────
    canvas_frame = tk.Frame(main, bg=ACCENT, padx=2, pady=2)  # border
    canvas_frame.pack(side="left")

    self.canvas = tk.Canvas(
        canvas_frame,
        width=CANVAS_PX,
        height=CANVAS_PX,
        bg=CANVAS_BG,
        highlightthickness=0,
        cursor="circle",
    )
    self.canvas.pack()
    self.canvas.bind("<B1-Motion>", self._paint)
    self.canvas.bind("<Button-1>", self._paint)

    # Model Input Preview (middle)
    pf = tk.Frame(main, bg=BG_DARK)
    pf.pack(side="left", padx=(16, 0))

    pc_border = tk.Frame(pf, bg="#334155", padx=2, pady=2)
    pc_border.pack()
    self.preview_canvas = tk.Canvas(pc_border,
                                    width=CANVAS_PX,
                                    height=CANVAS_PX,
                                    bg="black",
                                    highlightthickness=0)
    self.preview_canvas.pack()
    tk.Label(pf,
             text="Model Input (28x28)",
             font=self.font_small,
             fg=TEXT_MUTED,
             bg=BG_DARK).pack(pady=(4, 0))

    # ── Right panel ──────────────────────────────────────────────
    right = tk.Frame(main, bg=BG_DARK, padx=16)
    right.pack(side="left", fill="y")

    # Predicted digit display
    digit_card = tk.Frame(right, bg=PANEL_BG, padx=24, pady=8)
    digit_card.pack(pady=(0, 8))
    tk.Label(digit_card,
             text="Prediction",
             font=self.font_label,
             fg=TEXT_MUTED,
             bg=PANEL_BG).pack()
    self.digit_label = tk.Label(
        digit_card,
        text="—",
        font=self.font_digit,
        fg=SUCCESS,
        bg=PANEL_BG,
        width=3,
    )
    self.digit_label.pack()

    # Confidence label
    self.conf_label = tk.Label(
        right,
        text="",
        font=self.font_small,
        fg=TEXT_MUTED,
        bg=BG_DARK,
    )
    self.conf_label.pack(pady=(4, 6))

    # ── Probability bars ─────────────────────────────────────────
    bars_frame = tk.Frame(right, bg=BG_DARK)
    bars_frame.pack(fill="x", pady=(0, 10))

    self.bar_canvases = []
    self.bar_labels = []
    BAR_W, BAR_H = 180, 12

    for i in range(10):
      row = tk.Frame(bars_frame, bg=BG_DARK)
      row.pack(fill="x", pady=1)

      lbl = tk.Label(row,
                     text=f"{i}",
                     font=self.font_bar,
                     fg=TEXT_MUTED,
                     bg=BG_DARK,
                     width=2,
                     anchor="e")
      lbl.pack(side="left")

      bar = tk.Canvas(row,
                      width=BAR_W,
                      height=BAR_H,
                      bg=BAR_BG,
                      highlightthickness=0)
      bar.pack(side="left", padx=(4, 4))
      self.bar_canvases.append(bar)

      pct = tk.Label(row,
                     text="",
                     font=self.font_bar,
                     fg=TEXT_MUTED,
                     bg=BG_DARK,
                     width=6,
                     anchor="w")
      pct.pack(side="left")
      self.bar_labels.append(pct)

    self.BAR_W = BAR_W
    self.BAR_H = BAR_H

    # ── Buttons ──────────────────────────────────────────────────
    btn_frame = tk.Frame(right, bg=BG_DARK)
    btn_frame.pack(pady=(4, 0))

    self.predict_btn = tk.Button(
        btn_frame,
        text="⚡ Predict",
        font=self.font_btn,
        bg=ACCENT,
        fg="white",
        activebackground=ACCENT_HOVER,
        activeforeground="white",
        relief="flat",
        padx=18,
        pady=6,
        cursor="hand2",
        command=self._on_predict,
    )
    self.predict_btn.pack(side="left", padx=4)

    self.clear_btn = tk.Button(
        btn_frame,
        text="✕ Clear",
        font=self.font_btn,
        bg="#334155",
        fg=TEXT_PRIMARY,
        activebackground="#475569",
        activeforeground="white",
        relief="flat",
        padx=18,
        pady=6,
        cursor="hand2",
        command=self._on_clear,
    )
    self.clear_btn.pack(side="left", padx=4)

    # Bottom padding
    tk.Frame(self.root, bg=BG_DARK, height=14).pack()

  # ─────────────────────────────────────────────────────────────────
  #  Drawing
  # ─────────────────────────────────────────────────────────────────
  def _paint(self, event):
    x, y = event.x, event.y
    r = BRUSH_R
    # Draw on tkinter canvas
    self.canvas.create_oval(
        x - r,
        y - r,
        x + r,
        y + r,
        fill="white",
        outline="white",
    )
    # Mirror onto the PIL image
    self.pil_draw.ellipse(
        [x - r, y - r, x + r, y + r],
        fill=255,
    )

  # ─────────────────────────────────────────────────────────────────
  #  Predict
  # ─────────────────────────────────────────────────────────────────
  def _on_predict(self):
    # 1. Centre-crop & resize to 28×28 (like MNIST preprocessing)
    img = self.pil_image.copy()

    # Apply slight Gaussian blur to mimic MNIST anti-aliasing
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Find bounding box of drawn content & centre it (MNIST-style)
    bbox = img.getbbox()
    if bbox is None:
      self.digit_label.config(text="—")
      self.conf_label.config(text="Draw something first!")
      return

    # Crop to drawn content
    cropped = img.crop(bbox)

    # Fit into a 20×20 box (MNIST digits are ~20×20 centred in 28×28)
    cw, ch = cropped.size
    scale = min(20 / cw, 20 / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    # Paste centred into 28×28 black image
    final = Image.new("L", (MNIST_PX, MNIST_PX), 0)
    offset_x = (MNIST_PX - new_w) // 2
    offset_y = (MNIST_PX - new_h) // 2
    final.paste(cropped, (offset_x, offset_y))

    # Update preview
    preview_img = final.resize((CANVAS_PX, CANVAS_PX), Image.NEAREST)
    self.tk_preview = ImageTk.PhotoImage(preview_img)
    self.preview_canvas.create_image(CANVAS_PX // 2,
                                     CANVAS_PX // 2,
                                     image=self.tk_preview)

    # 2. Convert to the same format the model expects
    pixel_data = np.asarray(final, dtype=np.float32).reshape(1, 784) / 255.0

    # 3. Run prediction
    probs = model.predict(input=pixel_data)[0]  # shape (10,)
    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit]) * 100

    # 4. Update UI
    self.digit_label.config(text=str(predicted_digit))
    self.conf_label.config(text=f"Confidence: {confidence:.1f}%")
    self._draw_bars(probs)

  def _draw_bars(self, probs: np.ndarray):
    best = int(np.argmax(probs))
    for i, p in enumerate(probs):
      bar = self.bar_canvases[i]
      bar.delete("all")
      fill_w = max(1, int(p * self.BAR_W))
      colour = SUCCESS if i == best else BAR_FILL
      bar.create_rectangle(0, 0, fill_w, self.BAR_H, fill=colour, outline="")
      self.bar_labels[i].config(text=f"{p*100:5.1f}%")

  # ─────────────────────────────────────────────────────────────────
  #  Clear
  # ─────────────────────────────────────────────────────────────────
  def _on_clear(self):
    self.canvas.delete("all")
    self.preview_canvas.delete("all")
    self.pil_image = Image.new("L", (CANVAS_PX, CANVAS_PX), 0)
    self.pil_draw = ImageDraw.Draw(self.pil_image)
    self.digit_label.config(text="—")
    self.conf_label.config(text="")
    for bar in self.bar_canvases:
      bar.delete("all")
    for lbl in self.bar_labels:
      lbl.config(text="")


# =====================================================================
#  3.  Launch
# =====================================================================
if __name__ == "__main__":
  root = tk.Tk()
  app = DrawApp(root)
  root.mainloop()

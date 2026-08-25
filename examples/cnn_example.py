"""
cnn_run.py  —  CNN training + interactive draw-and-predict demo
================================================================
Architecture  (LeNet-style, 98–99 % expected accuracy on MNIST):

 Input  (batch, 1, 28, 28)
 ├─ Conv(16, 3×3, pad=1)  + BN + ReLU  →  (batch, 16, 28, 28)
 ├─ MaxPool(2, stride=2)               →  (batch, 16, 14, 14)
 ├─ Conv(32, 3×3, pad=1)  + BN + ReLU  →  (batch, 32, 14, 14)
 ├─ MaxPool(2, stride=2)               →  (batch, 32, 7, 7)
 ├─ Flatten                            →  (batch, 1568)
 ├─ Dense(256, ReLU, Dropout=0.3)
 └─ Dense(10,  softmax)

Saved weights file:  cnn_weights.npz
 On first run the model is trained and weights are saved.
 Subsequent runs skip training and load straight from file.
"""

import os, time
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import numpy as np

from keras.datasets import mnist
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from deep_learning import Network, Conv2D, MaxPool2D, Flatten, Dense, InputLayer, ActivationFunction, LossFunction

# =====================================================================
#  1.  Data
# =====================================================================
WEIGHTS_FILE = "../weights/cnn_weights.npz"

print("Loading MNIST …")
(x_train_raw, y_train_lbl), (x_test_raw, y_test_lbl) = mnist.load_data()

# Shape: (N, 1, 28, 28),  normalised to [0, 1]
x_train = (x_train_raw.reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)
x_test  = (x_test_raw .reshape(-1, 1, 28, 28) / 255.0).astype(np.float32)

def onehot(labels, n=10):
  m = np.zeros((len(labels), n), dtype=np.float32)
  m[np.arange(len(labels)), labels] = 1
  return m

y_train = onehot(y_train_lbl)
y_test  = onehot(y_test_lbl)

# =====================================================================
#  2.  Build model
# =====================================================================
act = ActivationFunction

layers = [
  InputLayer(x_train),
  Conv2D(16, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  MaxPool2D(2, 2),
  Conv2D(32, 3, act_func=act.ReLU, stride=1, padding=1, use_bn=True),
  MaxPool2D(2, 2),
  Flatten(),
  Dense(256, act_func=act.ReLU, use_dropout=True, drop_rate=0.3),
  Dense(10, act_func=act.softmax, use_dropout=False, drop_rate=0.0)
]

model = Network(
  layers        = layers,
  training_set  = (x_train, y_train),
  test_set      = (x_test,  y_test),
  loss_func     = LossFunction.cc_loss,
  batch         = None,
  learning_rate = 0.1,
  epsilon       = 1e-8,
  epoch_limit   = 3,
  iteration_event_trigger = 1,
)

# =====================================================================
#  3.  Train or load
# =====================================================================
if os.path.exists(WEIGHTS_FILE):
  print(f"\nFound saved weights → loading {WEIGHTS_FILE}  (skip training)")
  model.load_weights(WEIGHTS_FILE)
  model.training = False
  accuracy = model.evaluate()
else:
  print("\nNo saved weights found — training from scratch …")
  print("(this may take 20–40 min on CPU; weights will be saved for next run)\n")
  t0 = time.time()
  model.fit_model()
  elapsed = time.time() - t0
  accuracy = model.evaluate()
  print(f"\nTraining done in {elapsed/60:.1f} min  |  Test accuracy: {accuracy:.2f}%")
  model.save_weights(WEIGHTS_FILE)

print(f"Model accuracy: {accuracy:.2f}%\n")

# =====================================================================
#  4.  Draw-and-predict UI  (same dark theme as draw_demo.py)
# =====================================================================

BG_DARK      = "#0f0f1a"
PANEL_BG     = "#1a1a2e"
CANVAS_BG    = "#111122"
ACCENT       = "#06b6d4"        # cyan  (different from MLP version to tell them apart)
ACCENT_HOVER = "#22d3ee"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED   = "#94a3b8"
SUCCESS      = "#a855f7"        # purple for the predicted digit
BAR_BG       = "#2a2a40"
BAR_FILL     = "#06b6d4"

CANVAS_PX = 280
MNIST_PX  = 28
BRUSH_R   = 10


class DrawApp:
  def __init__(self, root: tk.Tk):
    self.root = root
    root.title("CNN — Digit Recogniser")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    self.font_title  = tkfont.Font(family="Segoe UI",  size=16, weight="bold")
    self.font_digit  = tkfont.Font(family="Consolas",  size=72, weight="bold")
    self.font_label  = tkfont.Font(family="Segoe UI",  size=11)
    self.font_btn    = tkfont.Font(family="Segoe UI",  size=11, weight="bold")
    self.font_small  = tkfont.Font(family="Segoe UI",  size=9)
    self.font_bar    = tkfont.Font(family="Consolas",  size=9)

    self.pil_image = Image.new("L", (CANVAS_PX, CANVAS_PX), 0)
    self.pil_draw  = ImageDraw.Draw(self.pil_image)
    self._build_ui()

  def _build_ui(self):
    # Title bar
    tf = tk.Frame(self.root, bg=BG_DARK, pady=12)
    tf.pack(fill="x")
    tk.Label(tf, text="✦  CNN Digit Recogniser", font=self.font_title,
         fg=ACCENT, bg=BG_DARK).pack()
    tk.Label(tf, text=f"Conv + BN + Dropout  ·  Accuracy: {accuracy:.2f}%",
         font=self.font_small, fg=TEXT_MUTED, bg=BG_DARK).pack()

    # Main row
    main = tk.Frame(self.root, bg=BG_DARK, padx=16, pady=4)
    main.pack()

    # Drawing canvas (with accent border)
    cf = tk.Frame(main, bg=ACCENT, padx=2, pady=2)
    cf.pack(side="left")
    self.canvas = tk.Canvas(cf, width=CANVAS_PX, height=CANVAS_PX,
                bg=CANVAS_BG, highlightthickness=0, cursor="circle")
    self.canvas.pack()
    self.canvas.bind("<B1-Motion>", self._paint)
    self.canvas.bind("<Button-1>",  self._paint)

    # Model Input Preview (middle)
    pf = tk.Frame(main, bg=BG_DARK)
    pf.pack(side="left", padx=(16, 0))
    
    pc_border = tk.Frame(pf, bg="#334155", padx=2, pady=2)
    pc_border.pack()
    self.preview_canvas = tk.Canvas(pc_border, width=CANVAS_PX, height=CANVAS_PX,
                                    bg="black", highlightthickness=0)
    self.preview_canvas.pack()
    tk.Label(pf, text="Model Input (28x28)", font=self.font_small,
             fg=TEXT_MUTED, bg=BG_DARK).pack(pady=(4, 0))

    # Right panel
    right = tk.Frame(main, bg=BG_DARK, padx=16)
    right.pack(side="left", fill="y")

    # Predicted digit
    dc = tk.Frame(right, bg=PANEL_BG, padx=24, pady=8)
    dc.pack(pady=(0, 8))
    tk.Label(dc, text="Prediction", font=self.font_label,
         fg=TEXT_MUTED, bg=PANEL_BG).pack()
    self.digit_label = tk.Label(dc, text="—", font=self.font_digit,
                  fg=SUCCESS, bg=PANEL_BG, width=3)
    self.digit_label.pack()

    self.conf_label = tk.Label(right, text="", font=self.font_small,
                  fg=TEXT_MUTED, bg=BG_DARK)
    self.conf_label.pack(pady=(4, 6))

    # Probability bars
    bars = tk.Frame(right, bg=BG_DARK)
    bars.pack(fill="x", pady=(0, 10))
    self.bar_canvases = []
    self.bar_labels   = []
    BAR_W, BAR_H = 180, 12
    self.BAR_W, self.BAR_H = BAR_W, BAR_H

    for digit in range(10):
      row = tk.Frame(bars, bg=BG_DARK)
      row.pack(fill="x", pady=1)
      tk.Label(row, text=f"{digit}", font=self.font_bar,
           fg=TEXT_MUTED, bg=BG_DARK, width=2, anchor="e").pack(side="left")
      bar = tk.Canvas(row, width=BAR_W, height=BAR_H,
              bg=BAR_BG, highlightthickness=0)
      bar.pack(side="left", padx=(4, 4))
      self.bar_canvases.append(bar)
      pct = tk.Label(row, text="", font=self.font_bar,
              fg=TEXT_MUTED, bg=BG_DARK, width=6, anchor="w")
      pct.pack(side="left")
      self.bar_labels.append(pct)

    # Buttons
    bf = tk.Frame(right, bg=BG_DARK)
    bf.pack(pady=(4, 0))
    tk.Button(bf, text="⚡ Predict", font=self.font_btn,
         bg=ACCENT, fg="white", activebackground=ACCENT_HOVER,
         activeforeground="white", relief="flat", padx=18, pady=6,
         cursor="hand2", command=self._on_predict).pack(side="left", padx=4)
    tk.Button(bf, text="✕ Clear", font=self.font_btn,
         bg="#334155", fg=TEXT_PRIMARY, activebackground="#475569",
         activeforeground="white", relief="flat", padx=18, pady=6,
         cursor="hand2", command=self._on_clear).pack(side="left", padx=4)

    tk.Frame(self.root, bg=BG_DARK, height=14).pack()

  def _paint(self, event):
    x, y, r = event.x, event.y, BRUSH_R
    self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

  def _on_predict(self):
    img  = self.pil_image.copy()
    img  = img.filter(ImageFilter.GaussianBlur(radius=1))
    bbox = img.getbbox()
    if bbox is None:
      self.conf_label.config(text="Draw something first!")
      return

    # Centre digit inside 20×20 box (MNIST convention) within 28×28 canvas
    cropped = img.crop(bbox)
    cw, ch  = cropped.size
    scale   = min(20 / cw, 20 / ch)
    nw, nh  = max(1, int(cw * scale)), max(1, int(ch * scale))
    cropped = cropped.resize((nw, nh), Image.LANCZOS)

    final   = Image.new("L", (MNIST_PX, MNIST_PX), 0)
    final.paste(cropped, ((MNIST_PX - nw) // 2, (MNIST_PX - nh) // 2))

    # Update preview
    preview_img = final.resize((CANVAS_PX, CANVAS_PX), Image.NEAREST)
    self.tk_preview = ImageTk.PhotoImage(preview_img)
    self.preview_canvas.create_image(CANVAS_PX//2, CANVAS_PX//2, image=self.tk_preview)

    # Build the exact input the CNN expects: (1, 1, 28, 28) float32 in [0, 1]
    arr = np.asarray(final, dtype=np.float32).reshape(1, 1, MNIST_PX, MNIST_PX) / 255.0

    probs     = model.predict(arr)[0]           # shape (10,)
    best      = int(np.argmax(probs))
    confidence = float(probs[best]) * 100

    self.digit_label.config(text=str(best))
    self.conf_label.config(text=f"Confidence: {confidence:.1f}%")
    self._draw_bars(probs, best)

  def _draw_bars(self, probs, best):
    for i, p in enumerate(probs):
      bar = self.bar_canvases[i]
      bar.delete("all")
      fw = max(1, int(p * self.BAR_W))
      bar.create_rectangle(0, 0, fw, self.BAR_H,
                 fill=SUCCESS if i == best else BAR_FILL, outline="")
      self.bar_labels[i].config(text=f"{p*100:5.1f}%")

  def _on_clear(self):
    self.canvas.delete("all")
    self.preview_canvas.delete("all")
    self.pil_image = Image.new("L", (CANVAS_PX, CANVAS_PX), 0)
    self.pil_draw  = ImageDraw.Draw(self.pil_image)
    self.digit_label.config(text="—")
    self.conf_label.config(text="")
    for bar in self.bar_canvases:
      bar.delete("all")
    for lbl in self.bar_labels:
      lbl.config(text="")


# =====================================================================
#  5.  Launch
# =====================================================================
if __name__ == "__main__":
  root = tk.Tk()
  DrawApp(root)
  root.mainloop()

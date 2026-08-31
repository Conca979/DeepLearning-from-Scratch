import os, sys, time
import tkinter as tk
import numpy as np
import pandas as pd
import io

from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageFilter, ImageTk

from deep_learning import Network, Dense, InputLayer, ActivationFunction, LossFunction

# Draw-and-predict UI
class DrawApp:
  BG_DARK      = "#0f0f1a"
  PANEL_BG     = "#1a1a2e"
  CANVAS_BG    = "#111122"
  ACCENT       = "#7c3aed"   # purple — distinct from CNN demos
  ACCENT_HOVER = "#9f67ff"
  TEXT_PRIMARY = "#e2e8f0"
  TEXT_MUTED   = "#94a3b8"
  SUCCESS      = "#22d3ee"   # cyan
  BAR_BG       = "#2a2a40"
  BAR_FILL     = "#7c3aed"

  CANVAS_PX = 280
  MNIST_PX  = 28
  BRUSH_R   = 10

  def __init__(self, root: tk.Tk, accuracy: float):
    self.root     = root
    self.accuracy = accuracy
    root.title("Neural Network — Digit Recogniser")
    root.configure(bg=self.BG_DARK)
    root.resizable(False, False)

    self.font_title = tkfont.Font(family="Segoe UI", size=16, weight="bold")
    self.font_digit = tkfont.Font(family="Consolas", size=72, weight="bold")
    self.font_label = tkfont.Font(family="Segoe UI", size=11)
    self.font_btn   = tkfont.Font(family="Segoe UI", size=11, weight="bold")
    self.font_small = tkfont.Font(family="Segoe UI", size=9)
    self.font_bar   = tkfont.Font(family="Consolas", size=9)

    self.pil_image = Image.new("L", (self.CANVAS_PX, self.CANVAS_PX), 0)
    self.pil_draw  = ImageDraw.Draw(self.pil_image)
    self._build_ui()

  def _build_ui(self):
    # Title bar
    tf = tk.Frame(self.root, bg=self.BG_DARK, pady=12)
    tf.pack(fill="x")
    tk.Label(tf,
             text="✦  Digit Recogniser",
             font=self.font_title,
             fg=self.ACCENT,
             bg=self.BG_DARK).pack()
    tk.Label(tf,
             text=f"Dense NN  ·  Model accuracy: {self.accuracy:.2f}%  ·  Draw a digit below",
             font=self.font_small,
             fg=self.TEXT_MUTED,
             bg=self.BG_DARK).pack()

    # Main row
    main = tk.Frame(self.root, bg=self.BG_DARK, padx=16, pady=4)
    main.pack()

    # Drawing canvas (accent border)
    cf = tk.Frame(main, bg=self.ACCENT, padx=2, pady=2)
    cf.pack(side="left")
    self.canvas = tk.Canvas(cf,
                            width=self.CANVAS_PX,
                            height=self.CANVAS_PX,
                            bg=self.CANVAS_BG,
                            highlightthickness=0,
                            cursor="circle")
    self.canvas.pack()
    self.canvas.bind("<B1-Motion>", self._paint)
    self.canvas.bind("<Button-1>",  self._paint)

    # Model input preview (middle)
    pf = tk.Frame(main, bg=self.BG_DARK)
    pf.pack(side="left", padx=(16, 0))
    pc_border = tk.Frame(pf, bg="#334155", padx=2, pady=2)
    pc_border.pack()
    self.preview_canvas = tk.Canvas(pc_border,
                                    width=self.CANVAS_PX,
                                    height=self.CANVAS_PX,
                                    bg="black",
                                    highlightthickness=0)
    self.preview_canvas.pack()
    tk.Label(pf,
             text="Model Input (28x28)",
             font=self.font_small,
             fg=self.TEXT_MUTED,
             bg=self.BG_DARK).pack(pady=(4, 0))

    # Right panel
    right = tk.Frame(main, bg=self.BG_DARK, padx=16)
    right.pack(side="left", fill="y")

    # Predicted digit card
    dc = tk.Frame(right, bg=self.PANEL_BG, padx=24, pady=8)
    dc.pack(pady=(0, 8))
    tk.Label(dc,
             text="Prediction",
             font=self.font_label,
             fg=self.TEXT_MUTED,
             bg=self.PANEL_BG).pack()
    self.digit_label = tk.Label(dc,
                                text="—",
                                font=self.font_digit,
                                fg=self.SUCCESS,
                                bg=self.PANEL_BG,
                                width=3)
    self.digit_label.pack()

    self.conf_label = tk.Label(right,
                               text="",
                               font=self.font_small,
                               fg=self.TEXT_MUTED,
                               bg=self.BG_DARK)
    self.conf_label.pack(pady=(4, 6))

    # Probability bars
    bars = tk.Frame(right, bg=self.BG_DARK)
    bars.pack(fill="x", pady=(0, 10))
    self.bar_canvases = []
    self.bar_labels   = []
    BAR_W, BAR_H = 180, 12
    self.BAR_W, self.BAR_H = BAR_W, BAR_H

    for digit in range(10):
      row = tk.Frame(bars, bg=self.BG_DARK)
      row.pack(fill="x", pady=1)
      tk.Label(row,
               text=f"{digit}",
               font=self.font_bar,
               fg=self.TEXT_MUTED,
               bg=self.BG_DARK,
               width=2,
               anchor="e").pack(side="left")
      bar = tk.Canvas(row,
                      width=BAR_W,
                      height=BAR_H,
                      bg=self.BAR_BG,
                      highlightthickness=0)
      bar.pack(side="left", padx=(4, 4))
      self.bar_canvases.append(bar)
      pct = tk.Label(row,
                     text="",
                     font=self.font_bar,
                     fg=self.TEXT_MUTED,
                     bg=self.BG_DARK,
                     width=6,
                     anchor="w")
      pct.pack(side="left")
      self.bar_labels.append(pct)

    # Buttons
    bf = tk.Frame(right, bg=self.BG_DARK)
    bf.pack(pady=(4, 0))
    tk.Button(bf,
              text="⚡ Predict",
              font=self.font_btn,
              bg=self.ACCENT,
              fg="white",
              activebackground=self.ACCENT_HOVER,
              activeforeground="white",
              relief="flat",
              padx=18, pady=6,
              cursor="hand2",
              command=self._on_predict).pack(side="left", padx=4)
    tk.Button(bf,
              text="✕ Clear",
              font=self.font_btn,
              bg="#334155",
              fg=self.TEXT_PRIMARY,
              activebackground="#475569",
              activeforeground="white",
              relief="flat",
              padx=18, pady=6,
              cursor="hand2",
              command=self._on_clear).pack(side="left", padx=4)

    tk.Frame(self.root, bg=self.BG_DARK, height=14).pack()

  def _paint(self, event):
    x, y, r = event.x, event.y, self.BRUSH_R
    self.canvas.create_oval(x - r, y - r, x + r, y + r,
                            fill="white", outline="white")
    self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

  def _on_predict(self):
    img = self.pil_image.copy()
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    bbox = img.getbbox()
    if bbox is None:
      self.conf_label.config(text="Draw something first!")
      return

    # Centre drawn content inside 20x20 box within 28x28 canvas
    cropped = img.crop(bbox)
    cw, ch  = cropped.size
    scale   = min(20 / cw, 20 / ch)
    nw, nh  = max(1, int(cw * scale)), max(1, int(ch * scale))
    cropped = cropped.resize((nw, nh), Image.LANCZOS)

    final = Image.new("L", (self.MNIST_PX, self.MNIST_PX), 0)
    final.paste(cropped, ((self.MNIST_PX - nw) // 2, (self.MNIST_PX - nh) // 2))

    # Update preview
    preview_img     = final.resize((self.CANVAS_PX, self.CANVAS_PX), Image.NEAREST)
    self.tk_preview = ImageTk.PhotoImage(preview_img)
    self.preview_canvas.create_image(self.CANVAS_PX // 2, self.CANVAS_PX // 2,
                                     image=self.tk_preview)

    # Dense NN expects flat (1, 784) float32 in [0, 1]
    arr   = np.asarray(final, dtype=np.float32).reshape(1, 784) / 255.0
    probs = model.predict(input=arr)[0]   # shape (10,)
    best  = int(np.argmax(probs))

    self.digit_label.config(text=str(best))
    self.conf_label.config(text=f"Confidence: {float(probs[best]) * 100:.1f}%")
    self._draw_bars(probs, best)

  def _draw_bars(self, probs, best):
    for i, p in enumerate(probs):
      bar = self.bar_canvases[i]
      bar.delete("all")
      fw = max(1, int(p * self.BAR_W))
      bar.create_rectangle(0, 0, fw, self.BAR_H,
                           fill=self.SUCCESS if i == best else self.BAR_FILL,
                           outline="")
      self.bar_labels[i].config(text=f"{p * 100:5.1f}%")

  def _on_clear(self):
    self.canvas.delete("all")
    self.preview_canvas.delete("all")
    self.pil_image = Image.new("L", (self.CANVAS_PX, self.CANVAS_PX), 0)
    self.pil_draw  = ImageDraw.Draw(self.pil_image)
    self.digit_label.config(text="—")
    self.conf_label.config(text="")
    for bar in self.bar_canvases:
      bar.delete("all")
    for lbl in self.bar_labels:
      lbl.config(text="")

# Main — requires pre-trained weights (run training/nn_train_mnist.py first)
if __name__ == "__main__":
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

  WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "weights", "nn_mnist_weights.npz")

  if not os.path.exists(WEIGHTS_FILE):
    print(f"[ERROR] Weights not found: {WEIGHTS_FILE}")
    print("  -> Run  training/nn_train_mnist.py  first to generate weights.")
    sys.exit(1)

  act = ActivationFunction
  layers = [
    InputLayer(None, input_shape=(-1, 784)),
    Dense(512, act_func=act.ReLU),
    Dense(256, act_func=act.ReLU),
    Dense(128, act_func=act.ReLU),
    Dense(10,  act_func=act.softmax)
  ]

  model = Network(layers=layers, loss_func=LossFunction.cc_loss, learning_rate=0.01)
  model.training = False
  accuracy = model.load_weights(WEIGHTS_FILE) or 0.0

  root = tk.Tk()
  DrawApp(root, accuracy)
  root.mainloop()

"""
The drawing board: draw a digit with the mouse and watch the network process it, neuron by neuron.

  1st column  the board and, below it, the 28x28 photo that the network really receives
  2nd column  the network. Every circle is a neuron: the brighter it is, the more active it is. The lines are
              the connections that matter most right now (orange = they push, blue = they hold back).
              Clicking a neuron selects it: the lab shows its values.
  3rd column  the answer: the probability of every digit

Keys: left = draw, right / Delete = clear, E = random photo from the test set
"""
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from gui import base
from i18n import tr
from neural_net.data import prepare_drawing

SIDE = 280                             # side of the board, in pixels
NET_WIDTH, NET_HEIGHT = 540, 580
INPUT_SIDE = 84                        # the 28x28 photo inside the network diagram (enlarged 3 times)


class DrawingBoard:
    def __init__(self, parent, load_test):
        """load_test() must return the test photos (photos, digits), or raise FileNotFoundError."""
        self.parent = parent
        self.load_test = load_test
        self.net = None
        self.test = None           # test photos, read the first time they are needed
        self.activations = None    # the outputs of all the layers for the current drawing (None = empty board)
        self.switched_off = set()  # neurons switched off by the lab: pairs (layer, neuron)
        self.selected = None       # the clicked neuron: (layer, neuron)
        self.on_click = None       # function to call when a neuron is clicked
        self.on_change = None      # function to call after every new answer of the network
        self.brush = 20
        self.noise = 0.0
        self.fixed_noise = np.random.default_rng(1).normal(size=784).astype(np.float32)
        self.waiting = False

        # 1st column: the board, the drawing controls and what the network sees
        column = self._column(0, tr("Draw a digit"))
        self.board = tk.Canvas(column, width=SIDE, height=SIDE, bg="black", highlightthickness=1,
                               highlightbackground=base.BORDER, cursor="crosshair")
        self.board.pack()
        self.board.bind("<Button-1>", self._start_stroke)
        self.board.bind("<B1-Motion>", lambda event: self._stroke(event.x, event.y))
        self.board.bind("<Button-3>", lambda _: self.clear())
        base.explain(self.board, tr("Draw a digit with the left mouse button (the right one clears): the network "
                                    "answers while you draw."))
        r = base.row(column, pady=(8, 0))
        base.button(r, tr("Clear"), self.clear, explanation=tr("Cleans the board (also with the right button, "
                                                               "Delete or Esc).")).pack(side="left", expand=True,
                                                                                         fill="x", padx=(0, 4))
        base.button(r, tr("Photo from the test set"), self.example, explanation=tr(
            "Puts on the board a random photo from the test set, never seen during training (also with the "
            "E key).")).pack(side="left", expand=True, fill="x", padx=(4, 0))
        base.slider(column, tr("Brush thickness"), 6, 40, 1, 20, lambda v: setattr(self, "brush", int(v)),
                    show=lambda v: f"{v:.0f} px", explanation=tr(
            "The thickness of the stroke. MNIST digits have a medium stroke: with very thin or very thick "
            "strokes the network gets more confused."))
        base.slider(column, tr("Noise on the drawing"), 0, 0.6, 0.01, 0, self._set_noise,
                    show=lambda v: f"σ = {v:.2f}", explanation=tr(
            "Adds gaussian noise to the 28x28 photo that the network receives. The noise pattern is always the "
            "same, only its strength changes: this way you can see when the answer changes."))
        base.label(column, tr("What the network receives (28 x 28 pixels)"), base.TEXT, 10, True, pady=(12, 4))
        self.preview = tk.Label(column, bg="black", highlightthickness=1, highlightbackground=base.BORDER)
        self.preview.pack(anchor="w")
        base.explain(self.preview, tr("The drawing after the preparation: cropped, shrunk to 20x20 and centered "
                                      "on 28x28 like the MNIST photos (plus the noise, if any). These 784 pixels "
                                      "are the input of the network."))
        self.info = base.label(column, "", width=SIDE)

        # 2nd column: the network diagram
        column = self._column(1, tr("The neural network at work"))
        self.diagram = tk.Canvas(column, width=NET_WIDTH, height=NET_HEIGHT, bg=base.PANEL,
                                 highlightthickness=0, cursor="hand2")
        self.diagram.pack()
        self.diagram.bind("<Button-1>", self._click_diagram)
        base.explain(self.diagram, tr(
            "Every circle is a neuron: the more orange it is, the more active it is with this drawing. The lines "
            "are the connections that matter most right now (neuron output x weight): orange ones push towards "
            "switching on, blue ones hold it back. Click a neuron to examine and change it; the switched off ones "
            "have a red border."))

        # 3rd column: the answer
        column = self._column(2, tr("Answer of the network"))
        self.answer = tk.Label(column, text="?", bg=base.BACKGROUND, fg=base.ACCENT, font=(base.FONT, 60, "bold"),
                               width=2)
        self.answer.pack()
        self.confidence = tk.Label(column, bg=base.BACKGROUND, fg=base.TEXT_SOFT, font=(base.FONT, 11))
        self.confidence.pack(pady=(0, 12))
        self.bars = base.Bars(column, length=140, height=32)
        base.explain(self.bars.canvas, tr("The 10 probabilities of the last layer (softmax): they always add up "
                                          "to 100%. The answer is the digit with the highest probability."))

        # Shortcut keys: they work only when the board is visible (that is, when its tab is open)
        window = parent.winfo_toplevel()
        for key, action in (("<Delete>", self.clear), ("<Escape>", self.clear), ("e", self.example)):
            window.bind(key, lambda _, action=action: self.board.winfo_viewable() and action())
        self._new_drawing()

    def _column(self, number, title):
        column = tk.Frame(self.parent, bg=base.BACKGROUND)
        column.grid(row=0, column=number, padx=10, pady=(12, 0), sticky="n")
        base.title(column, title)
        return column

    # ------------------------------------------------------------------ the network

    def set_net(self, net):
        """Use this network. If its shape is different from the previous one, redraw the diagram."""
        old, self.net = self.net, net
        if old is None or old.layers != net.layers:
            self.selected = None
            self._draw_diagram()
        self.refresh()

    def _draw_diagram(self):
        """The "background" neurons and connections: redrawn only when the shape of the network changes."""
        canvas = self.diagram
        canvas.delete("all")
        layers = self.net.layers
        columns_x = np.linspace(66, NET_WIDTH - 56, len(layers))
        top, bottom = 60, NET_HEIGHT - 16
        center = (top + bottom) / 2

        # Column 0: the input photo (784 pixels would be too many to draw one by one)
        self.input_id = canvas.create_image(columns_x[0], center)
        half = INPUT_SIDE / 2
        canvas.create_rectangle(columns_x[0] - half - 1, center - half - 1, columns_x[0] + half, center + half,
                                outline=base.BORDER)
        canvas.create_text(columns_x[0], 26, text=tr("input\n{n} pixels", n=layers[0]), fill=base.TEXT_SOFT,
                           font=(base.FONT, 9), justify="center")

        # Next columns: one circle for every neuron
        self.positions, self.radii, self.circles = [], [], []
        for k, n in enumerate(layers[1:], start=1):
            step = (bottom - top) / n
            radius = min(step * 0.42, 17)
            ys = top + step * (np.arange(n) + 0.5)
            x = columns_x[k]
            self.positions.append(np.column_stack([np.full(n, x), ys]))
            self.radii.append(radius)
            name = (tr("output\n10 digits") if k == len(layers) - 1
                    else tr("hidden layer {k}\n{n} neurons", k=k, n=n))
            canvas.create_text(x, 26, text=name, fill=base.TEXT_SOFT, font=(base.FONT, 9), justify="center")
            self.circles.append([canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=base.OFF,
                                                    outline="", tags="neuron") for y in ys])
        self.output_numbers = [canvas.create_text(x, y, text=str(c), fill=base.TEXT, font=(base.FONT, 11, "bold"),
                                                  tags="neuron") for c, (x, y) in enumerate(self.positions[-1])]

        # Faint background connections: every neuron is connected to all those of the next layer
        # (if there are too many I don't draw them: it would just be a grey stain)
        faint = base.mix(base.PANEL, base.TEXT_SOFT, 0.08)
        for x, y in self.positions[0]:
            canvas.create_line(columns_x[0] + half, center, x, y, fill=faint)
        for start, end in zip(self.positions[:-1], self.positions[1:]):
            if len(start) * len(end) <= 4096:
                for x0, y0 in start:
                    for x1, y1 in end:
                        canvas.create_line(x0, y0, x1, y1, fill=faint)
        canvas.tag_raise("neuron")

    def _click_diagram(self, event):
        """Find the neuron closest to the clicked point and select it."""
        for layer, positions in enumerate(self.positions):
            if abs(event.x - positions[0, 0]) < 22:
                self.select((layer, int(np.argmin(np.abs(positions[:, 1] - event.y)))))
                if self.on_click:
                    self.on_click(self.selected)
                return

    def select(self, who):
        """Select a neuron (layer, neuron) and circle it in white; None = nobody."""
        self.selected = who
        self.diagram.delete("selection")
        if who is not None:
            (x, y), r = self.positions[who[0]][who[1]], self.radii[who[0]] + 4
            self.diagram.create_oval(x - r, y - r, x + r, y + r, outline="white", width=2, tags="selection")

    def refresh(self):
        """Prepare the drawing, pass it to the network and update all the graphics."""
        self.waiting = False
        if self.net is None:
            return
        photo28 = prepare_drawing(self.drawing)
        if photo28 is None:
            inputs, self.activations = np.zeros(784, np.float32), None
        else:
            inputs = photo28.reshape(784).astype(np.float32) / 255
            if self.noise > 0:
                inputs = np.clip(inputs + self.noise * self.fixed_noise, 0, 1)
            with np.errstate(all="ignore"):
                self.activations = [a[0] for a in self.net.forward(inputs[None])]

        pixels = Image.fromarray((inputs.reshape(28, 28) * 255).astype(np.uint8))
        self._preview_photo = ImageTk.PhotoImage(pixels.resize((140, 140), Image.Resampling.NEAREST))
        self.preview.config(image=self._preview_photo)
        self._input_photo = ImageTk.PhotoImage(pixels.resize((INPUT_SIDE,) * 2, Image.Resampling.NEAREST))
        self.diagram.itemconfig(self.input_id, image=self._input_photo)
        self._color()
        self._show_answer()
        if self.on_change:
            self.on_change()

    def _color(self):
        """Color the neurons by how active they are and draw the most important connections."""
        canvas, act = self.diagram, self.activations
        canvas.delete("connection")
        for k, circles in enumerate(self.circles):
            levels = np.zeros(len(circles))
            if act is not None:
                a = np.nan_to_num(np.abs(act[k + 1]))  # absolute value: with tanh the negatives count too
                levels = a / a.max() if a.max() > 0 else a
            for j, (circle, t) in enumerate(zip(circles, levels)):
                off = (k, j) in self.switched_off
                canvas.itemconfig(circle, fill=base.BACKGROUND if off else base.mix(base.OFF, base.ACCENT, t),
                                  outline=base.RED if off else "", width=2)
        for c, number in enumerate(self.output_numbers):
            canvas.itemconfig(number, fill=base.BACKGROUND if act is not None and act[-1][c] > 0.5 else base.TEXT)

        if act is not None:
            # Connections: contribution = output of the starting neuron x weight of the connection.
            # I draw only the strongest ones, otherwise it would be a tangle.
            for k in range(1, len(self.positions)):
                contribution = np.nan_to_num(act[k][:, None] * self.net.weights[k])
                strength = np.abs(contribution)
                if strength.max() == 0:
                    continue
                for index in np.argsort(strength, axis=None)[-min(strength.size // 4, 150):]:  # weakest to strongest
                    i, j = np.unravel_index(index, strength.shape)
                    m = strength[i, j] / strength.max()
                    color = base.mix(base.PANEL, base.ACCENT if contribution[i, j] > 0 else base.BLUE, 0.2 + 0.8 * m)
                    (x0, y0), (x1, y1) = self.positions[k - 1][i], self.positions[k][j]
                    canvas.create_line(x0, y0, x1, y1, fill=color, width=0.6 + 2.6 * m, tags="connection")
        canvas.tag_raise("neuron")
        canvas.tag_raise("selection")

    def _show_answer(self):
        if self.activations is None:
            self.answer.config(text="?")
            self.confidence.config(text=tr("draw a digit..."))
            return self.bars.show([None] * 10, [base.ACCENT] * 10)
        probabilities = np.nan_to_num(self.activations[-1])
        best = int(probabilities.argmax())
        self.answer.config(text=str(best))
        self.confidence.config(text=tr("{p:.0%} sure", p=probabilities[best]))
        grey = base.mix(base.PANEL, base.TEXT_SOFT, 0.7)
        self.bars.show(probabilities, [base.ACCENT if c == best else grey for c in range(10)])

    # ------------------------------------------------------------------ the drawing

    def _new_drawing(self, image=None):
        self.drawing = image or Image.new("L", (SIDE, SIDE), 0)  # the "real" copy of the drawing, for the network
        self.pen = ImageDraw.Draw(self.drawing)
        self.board.delete("all")

    def _start_stroke(self, event):
        self.info.config(text="")
        self.last_point = (event.x, event.y)
        self._stroke(event.x, event.y)

    def _stroke(self, x, y):
        """Draw both on the screen and on the image that will be given to the network."""
        (x0, y0), r = self.last_point, self.brush / 2
        self.board.create_line(x0, y0, x, y, width=self.brush, fill="white", capstyle=tk.ROUND)
        self.board.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
        self.pen.line([x0, y0, x, y], fill=255, width=self.brush)
        self.pen.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.last_point = (x, y)
        # I don't recompute at every tiny mouse movement: at most once every 30 milliseconds
        if not self.waiting:
            self.waiting = True
            self.board.after(30, self.refresh)

    def _set_noise(self, value):
        self.noise = value
        self.refresh()

    def clear(self):
        self._new_drawing()
        self.info.config(text="")
        self.refresh()

    def example(self):
        """Put on the board a random photo from the test set (never seen during training)."""
        if self.test is None:
            try:
                self.test = self.load_test()
            except FileNotFoundError as error:
                return messagebox.showwarning(tr("Test photos missing"), str(error))
        photos, digits = self.test
        i = np.random.randint(len(digits))
        image = Image.fromarray(photos[i]).resize((SIDE, SIDE), Image.Resampling.BILINEAR)
        self._new_drawing(image)
        self._board_photo = ImageTk.PhotoImage(image)
        self.board.create_image(0, 0, anchor="nw", image=self._board_photo)
        self.info.config(text=tr("Photo from the test set: the right digit is {digit}", digit=digits[i]))
        self.refresh()

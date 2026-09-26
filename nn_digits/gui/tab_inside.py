"""
Tab 5 · Inside the network: the math of one layer, number by number, for a test photo.

At the top the weight matrix (one row for every value that goes in, one column for every neuron),
on the left the values that go in. Below, lined up with the columns, the steps of the math:
bias, weighted sum z and activation; for the last layer the steps of the softmax, which turns
the scores into probabilities. Moving the mouse over a cell lights up the connected ones
and the box at the top explains the math.
"""
import tkinter as tk

import numpy as np
from matplotlib.patches import Rectangle

from nn_digits.gui import base
from nn_digits.i18n import tr
from nn_digits.neural_net import charts
from nn_digits.neural_net.data import normalize
from nn_digits.neural_net.network import NeuralNetwork, softmax_steps
from nn_digits.neural_net.storage import MODEL_FILE

Z_EXPLANATION = tr("The weighted sum: every input value multiplied by its weight (the highlighted column "
                   "in the matrix), all added up, plus the bias.")
BIAS_EXPLANATION = tr("The bias is a learned number that is always added, whatever the photo: it shifts the "
                      "\"threshold\" of the neuron.")
SHIFT_EXPLANATION = tr("The highest score is subtracted, so the biggest becomes 0 and the exponential doesn't "
                       "explode, and it is divided by the temperature T.")
EXPONENTIAL_EXPLANATION = tr("The exponential makes everything positive and widens the differences: the highest "
                             "score is worth 1, the others less than 1.")


class InsideTab(base.Tab):
    title = tr("5 · Inside the network")

    def __init__(self, window):
        super().__init__(window)
        self.net = self.math = self.background = self.position = None
        self.structure = None  # (layer, kind of matrix) of the charts drawn right now
        self.waiting = False

        c = base.column(self.frame)
        base.title(c, tr("Inside the network: weights and softmax"))
        base.label(c, tr("The math of one layer, number by number, for a test photo. Move the mouse over the "
                         "cells: the connected ones light up and the math appears here on the right."))

        base.section(c, tr("Input photo"))
        self.photo_slider = base.slider(c, tr("Test photo no."), 0, 1, 1, 0, self._redraw_soon,
                                        show=lambda v: f"{v:.0f}",
                                        explanation=tr("Which test photo to pass through the network."))
        r = base.row(c, pady=(8, 0))
        base.button(r, tr("Random"), self._random, explanation=tr("A random test photo.")).pack(side="left")
        base.button(r, tr("Next mistake"), self._next_mistake, explanation=tr(
            "The next photo that the network gets wrong: look in the softmax which digits compete for the answer."
        )).pack(side="left", padx=6)
        self.tiles = base.Tiles(c, ("true digit", "answer", "confidence"), size=13, name=tr("Answer for this photo"),
                                explanation=tr(
            "The answer of the real network (temperature 1) for this photo."))

        base.section(c, tr("What to look at"))
        self.layers_place = base.row(c, pady=(6, 0))  # the layer buttons change with the network
        r = base.row(c, pady=(8, 0))
        self.matrix = tk.StringVar(value="weights")
        base.choice_buttons(r, tr("Matrix"), [(tr("weights"), "weights"), (tr("input × weight"), "contributions")],
                            self.matrix, self._redraw, explanation=tr(
            "Weights: the numbers that the network has learned, the same for all the photos. Input × weight: how "
            "much every connection really weighs for THIS photo; the sum of every column (plus the bias) is the z "
            "below."))

        base.section(c, tr("Softmax (last layer)"))
        base.label(c, tr("p(digit) = e^(z / T)  divided by the sum of all the e^(z / T)"), base.TEXT)
        self.temperature = base.slider(c, tr("Temperature T"), -1, 1, 0.02, 0, self._redraw_soon,
                                       show=lambda v: f"{base.power(v):g}", explanation=tr(
            "Divides the scores z before the softmax. T < 1: the differences grow and the answer becomes sharp "
            "(at the limit all the probability goes to the digit with the highest score). T > 1: the "
            "probabilities flatten towards 10% each. The chosen answer never changes, the confidence does. "
            "It is the same \"temperature\" of language models."))
        self.statistics = base.label(c, "", base.TEXT)

        right = base.right_area(self.frame)
        base.explanation_box(right, self.frame, tr("Move the mouse over a cell to see its math."),
                             1220).pack(fill="x", pady=(0, 6))
        self.fig, self.canvas = base.figure(right)
        base.explain(self.canvas.get_tk_widget(), tr(
            "At the top the weight matrix (one row for every value that goes in, one column for every neuron) and, on "
            "the left, the values that go in. Below, lined up with the columns: bias, weighted sum z and activation; "
            "in the last layer the steps of the softmax. Move the mouse over a cell to see its math."),
            tr("The math of the layer"))
        base.explain_charts(self.canvas, {
            "input": (tr("The values that go in"), tr(
                "One cell for each value that enters the layer: the pixels of the photo for layer 1, the outputs of "
                "the layer before for the others. Orange = positive, blue = negative, dark = zero.")),
            "matrix": (tr("The weight matrix"), tr(
                "One row for every value that goes in, one column for every neuron. Orange = positive weight (it "
                "pushes up), blue = negative (it pushes down). With \"input × weight\" each cell shows how much that "
                "connection really counts for this photo."), base.chart_title),
            "photo": (tr("The photo"), tr(
                "The test photo that goes into the network. In layer 1 each pixel is one of the values that go in: "
                "move the mouse over it to find its row in the matrix.")),
            "bias": ("bias", BIAS_EXPLANATION),
            "z": (tr("z = Σ input × weight + bias"), Z_EXPLANATION),
            "activation": (tr("Activation"), tr(
                "The output of each neuron: the activation function applied to z (with relu the negative values "
                "become 0, the neuron is switched off). It is what goes into the next layer."),
                lambda ax: self.net and self.net.activation),
            "shift": ("(z − max) / T", SHIFT_EXPLANATION),
            "exponential": (tr("e = exponential"), EXPONENTIAL_EXPLANATION),
            "probability": ("p = e / Σe", tr(
                "Every e divided by the sum of all of them: the 10 probabilities add up to 1. It is the final "
                "answer of the network.")),
            "bars": (tr("The last step as bars"), tr(
                "The last row drawn as bars, one per neuron: in the output layer the 10 probabilities (the highest "
                "in orange, the green border is the true digit), in the other layers the activations."))})
        self.canvas.mpl_connect("draw_event", self._save_background)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ------------------------------------------------ data and drawing

    def refresh(self):
        """The model or the photos have changed: I reload everything."""
        if not MODEL_FILE.exists():
            return self._empty(tr("No trained model:\ntrain the network in the tab \"2 · Training\""))
        try:
            self.photos, self.digits = self.window.photos("test")
        except FileNotFoundError:
            return self._empty(tr("First download the photos in the tab \"1 · Data\""))
        self.net = NeuralNetwork.load()
        self.X = normalize(self.photos)
        self.predicted = self.net.predict(self.X).argmax(axis=1)
        self.photo_slider.config(to=len(self.digits) - 1)
        self.structure = None  # the new network may have different layers: charts to redo
        self._layer_buttons()
        self._redraw()

    def _layer_buttons(self):
        for old in self.layers_place.winfo_children():
            old.destroy()
        n = len(self.net.weights)
        self.layer = tk.StringVar(value=str(n - 1))  # the last one to start with: the one of the softmax
        base.choice_buttons(self.layers_place, tr("Layer"), [(str(k + 1), str(k)) for k in range(n - 1)] + [
            (tr("output"), str(n - 1))], self.layer, self._redraw, explanation=tr(
            "Which layer to look at. The network is {shape}: layer 1 receives the 784 pixels, the output "
            "receives the last hidden layer and ends with the softmax.", shape=" → ".join(map(str, self.net.layers))))

    def _redraw_soon(self, *_):
        """The sliders call many times per second: I collect the requests and redraw only once."""
        if not self.waiting:
            self.waiting = True
            self.frame.after(30, self._redraw)

    def _redraw(self, *_):
        self.waiting = False
        if self.net is None:
            return
        i, k = int(self.photo_slider.get()), int(self.layer.get())
        output = k == len(self.net.weights) - 1
        activations = self.net.forward(self.X[i:i + 1])
        x, W, b = activations[k][0], self.net.weights[k], self.net.biases[k]
        z = x @ W + b

        # The steps of the math: (name, values, number format, explanation)
        rows = [("bias", b, "{:+.2f}", BIAS_EXPLANATION),
                (tr("z = Σ input × weight + bias"), z, "{:+.2f}", Z_EXPLANATION)]
        if output:
            T = base.power(self.temperature.get())
            shifted, e, p = softmax_steps(z, T)
            rows += [("(z − max) / T", shifted, "{:+.2f}", SHIFT_EXPLANATION),
                     (tr("e = exponential"), e, "{:.2f}", EXPONENTIAL_EXPLANATION),
                     ("p = e / Σe", p, "{:.0%}", tr("Every e divided by the sum of all of them (Σe = {total:.3f}): "
                      "the 10 probabilities add up to 1. It is the final answer of the network.", total=e.sum()))]
            entropy = float(-(p * np.log2(p + 1e-12)).sum())
            self.statistics.config(text=tr("Σe = {total:.3f}   ·   highest p = {top:.1%}\n"
                                           "uncertainty (entropy) = {entropy:.2f} bits  (max 3.32 = all at 10%)",
                                           total=e.sum(), top=p.max(), entropy=entropy))
        else:
            a = activations[k + 1][0]
            rows.append((f"a = {self.net.activation}(z)", a, "{:.2f}",
                         tr("The output of the neuron: the function {activation} applied to z. With relu the "
                            "negative values become 0 (neuron switched off).", activation=self.net.activation)))
            self.statistics.config(text=tr("The softmax is only in the last layer (\"output\").\n"
                                           "Neurons switched on (a > 0) in this layer: {on} out of {total}",
                                           on=np.sum(a > 0), total=len(a)))

        probabilities = activations[-1][0]
        self.tiles.show({"true digit": int(self.digits[i]), "answer": int(probabilities.argmax()),
                         "confidence": f"{probabilities.max():.0%}"})
        self.math = dict(k=k, output=output, x=x, W=W, z=z, rows=rows)
        contributions = self.matrix.get() == "contributions"
        if (k, contributions) != self.structure:  # the layer or the matrix changes: I redo the charts from zero
            self.structure = (k, contributions)
            self.axes = charts.layer_view(self.fig, len(x), rows,
                                          tr("input × weight") if contributions else tr("weights"), output)
            steps = ["bias", "z"] + (["shift", "exponential", "probability"] if output else ["activation"])
            for ax, step in zip(self.axes["rows"], steps):  # which step each row is: Pick explains it
                ax.set_gid(step)
            self._create_boxes()
            self._fill(i, x, W, rows, contributions, output)
            self.canvas.draw()
            # Now the positions of the charts stay fixed: the next drawings skip the layout (faster)
            self.fig.set_layout_engine("none")
        else:  # only the photo or the temperature changes: I just update the numbers
            self._fill(i, x, W, rows, contributions, output)
            self.canvas.draw()

    def _fill(self, i, x, W, rows, contributions, output):
        name = tr("output") if output else tr("layer {k}", k=self.math["k"] + 1)
        title = tr("Photo no. {i} · {name}: {n_in} values go in, {n_out} neurons compute",
                   i=i, name=name, n_in=len(x), n_out=W.shape[1])
        charts.fill_layer_view(self.axes, title + (tr(" · then the softmax") if output else ""),
                               self.photos[i], x, W * x[:, None] if contributions else W, rows,
                               self.digits[i] if output else None)

    def _random(self):
        if self.net is not None:
            self.photo_slider.set(np.random.randint(len(self.digits)))

    def _next_mistake(self):
        if self.net is None:
            return
        mistakes = np.flatnonzero(self.predicted != self.digits)
        if len(mistakes):
            after = mistakes[mistakes > self.photo_slider.get()]
            self.photo_slider.set(after[0] if len(after) else mistakes[0])

    def _empty(self, text):
        self.net = self.math = self.structure = None
        self.tiles.clear()
        self.statistics.config(text="")
        base.message(self.fig, text)
        self.canvas.draw()

    # ------------------------------------------------ cells lit up when the mouse passes over them

    def _create_boxes(self):
        """White boxes, hidden until the mouse passes over a connected cell. They are "animated":
        they draw themselves on top of a snapshot of the figure, without redrawing everything (much faster)."""
        def box(ax):
            return ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="white", lw=1.8, animated=True, visible=False))
        self.boxes = {"matrix": box(self.axes["matrix"]), "input": box(self.axes["input"]),
                      "photo": box(self.axes["photo"]), "rows": [box(ax) for ax in self.axes["rows"]]}
        self.position = None

    def _save_background(self, _event):
        self.background = self.canvas.copy_from_bbox(self.fig.bbox)

    def _on_motion(self, event):
        """Finds the cell under the mouse: (row, column) in the matrix, only the row (an input)
        or only the column (a neuron, in the steps below)."""
        if self.math is None or self.background is None:
            return
        n_in, n_out = self.math["W"].shape
        ax, position = event.inaxes, None
        if ax is not None:
            row = int(np.clip(round(event.ydata), 0, n_in - 1))
            column = int(np.clip(round(event.xdata), 0, n_out - 1))
            if ax is self.axes["matrix"]:
                position = (row, column, None)
            elif ax is self.axes["input"]:
                position = (row, None, None)
            elif ax in self.axes["rows"]:
                position = (None, column, self.axes["rows"].index(ax))
            elif ax is self.axes["bars"]:
                position = (None, column, len(self.math["rows"]) - 1)
            elif ax is self.axes["photo"] and self.math["k"] == 0:  # in layer 1 the inputs are the pixels
                position = (int(np.clip(round(event.ydata), 0, 27)) * 28 + int(np.clip(round(event.xdata), 0, 27)),
                            None, None)
        if position == self.position:
            return
        self.position = position
        self._light_up(*(position or (None, None, None)))
        if position:
            self.frame.explanations.config(text=self._explanation(*position))

    def _light_up(self, row, column, _step):
        n_in, n_out = self.math["W"].shape
        b = self.boxes
        everything = [b["matrix"], b["input"], b["photo"], *b["rows"]]
        for r in everything:
            r.set_visible(False)
        if row is not None or column is not None:  # in the matrix: one cell, one row or one column
            x0, width = (column - 0.5, 1) if column is not None else (-0.5, n_out)
            y0, height = (row - 0.5, 1) if row is not None else (-0.5, n_in)
            b["matrix"].set_bounds(x0, y0, width, height)
            b["matrix"].set_visible(True)
        if row is not None:
            b["input"].set_bounds(-0.5, row - 0.5, 1, 1)
            b["input"].set_visible(True)
            if self.math["k"] == 0:
                b["photo"].set_bounds(row % 28 - 0.5, row // 28 - 0.5, 1, 1)
                b["photo"].set_visible(True)
        if column is not None:
            for r in b["rows"]:
                r.set_bounds(column - 0.5, -0.5, 1, 1)
                r.set_visible(True)
        self.canvas.restore_region(self.background)
        for r in everything:
            if r.get_visible():
                r.axes.draw_artist(r)
        self.canvas.blit(self.fig.bbox)

    def _input_name(self, i):
        k = self.math["k"]
        return (tr("pixel {i} (row {row}, column {column})", i=i, row=i // 28, column=i % 28) if k == 0
                else tr("neuron {n} of layer {k}", n=i, k=k))

    def _output_name(self, j):
        return (tr("output of digit {j}", j=j) if self.math["output"]
                else tr("neuron {n} of layer {k}", n=j, k=self.math["k"] + 1))

    def _explanation(self, row, column, step):
        x, W, z = self.math["x"], self.math["W"], self.math["z"]
        if row is not None and column is not None:
            weight = W[row, column]
            return tr("Weight of the connection {input} → {output}: {weight:+.3f}. The input is worth {x:.3f}, so it "
                      "brings {part:+.3f} to the sum z, which in total is worth {z:+.2f}. Orange = positive weight "
                      "(pushes up), blue = negative (pushes down).", input=self._input_name(row),
                      output=self._output_name(column), weight=weight, x=x[row], part=x[row] * weight, z=z[column])
        if row is not None:
            contributions = x[row] * W[row]
            return (tr("{input} is worth {x:.3f}. Its row of weights says how much it counts for each of the {n} "
                       "neurons: for this photo it brings from {low:+.2f} to {high:+.2f}.",
                       input=self._input_name(row).capitalize(), x=x[row], n=len(z), low=contributions.min(),
                       high=contributions.max())
                    + (tr(" It is 0, so for this photo it doesn't count at all.") if x[row] == 0 else ""))
        name, values, number_format, explanation = self.math["rows"][step]
        contributions = x * W[:, column]
        strongest = np.argsort(-np.abs(contributions))[:3]
        return (f"{self._output_name(column).capitalize()} · {name} = {number_format.format(values[column])}. "
                f"{explanation} " + tr("The inputs that weigh the most here:") + " "
                + ", ".join(f"{i} ({contributions[i]:+.2f})" for i in strongest) + ".")

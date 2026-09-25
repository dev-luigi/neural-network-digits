"""
The lab: put your hands inside the trained network and see what happens.

The changes are made on a COPY of the network: the original stays as it is until you press
"Save as model". At every change:
  1. we start again from the original and apply all the changes again (_apply_changes)
  2. the drawing board shows the new answer on the drawn digit
  3. we measure the network again on the test photos, to see the effect on all the digits
"""
import copy
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from gui import base
from i18n import tr
from neural_net.data import normalize, one_hot
from neural_net.network import NeuralNetwork

NO_CHANGE = {"bias": 0.0, "scale": 1.0, "off": False}
MEASURES = ("test accuracy", "test loss", "mean confidence")  # names of the tiles (Tiles translates them)


class Lab:
    def __init__(self, parent, board, load_test, on_save=None):
        self.board = board
        self.load_test = load_test
        self.on_save = on_save                # who to tell when you save (e.g. the Evaluation tab)
        self.original = None                  # the saved network: don't touch it
        self.modified = None                  # the copy with all the changes applied
        self.changes = {}                     # (layer, neuron) -> {"bias": +x, "scale": x, "off": yes/no}
        self.test = self.X_test = self.original_measures = None
        self.accuracies = None                # test accuracy (original, modified): the assistant reads it
        self._images = []                     # drawn images: they must be kept, otherwise Tkinter deletes them
        board.on_click = self.select
        board.on_change = self._show_neuron

        column = tk.Frame(parent, bg=base.BACKGROUND)
        column.grid(row=0, column=3, padx=10, pady=(12, 0), sticky="n")
        tk.Frame(column, width=310, height=0, bg=base.BACKGROUND).pack()  # keeps the column 310 pixels wide
        base.title(column, tr("Change the network"))
        self.temperature = base.slider(column, tr("Temperature"), -1, 0.7, 0.05, 0, self._recompute,
                                       show=lambda v: f"{base.power(v):g}", explanation=tr(
            "Divides the final scores before the softmax. Above 1 the probabilities \"flatten\" (the network "
            "looks more uncertain), below 1 they \"sharpen\" (more sure). The most likely answer doesn't change, "
            "but loss and confidence do."))
        self.weight_noise = base.slider(column, tr("Noise on the weights"), 0, 1, 0.02, 0, self._recompute,
                                        show=lambda v: tr("{v:.0%} of σ", v=v), explanation=tr(
            "Adds to every weight a random number as big as this percentage of the σ of the weights of its layer "
            "(always the same noise: only its strength changes). How well does the network hold up if its "
            "weights are imprecise?"))
        self.pruning = base.slider(column, tr("Pruning"), 0, 0.99, 0.01, 0, self._recompute,
                                   show=lambda v: tr("{v:.0%} of the weights at zero", v=v), explanation=tr(
            "Sets to zero this percentage of the smallest weights (in absolute value) of every layer. Many "
            "networks still work well even without most of their weights: how far does yours hold up?"))
        self.measures = base.Tiles(column, MEASURES, columns=1, size=11, name=tr("Original network → modified network"),
                                   explanation=tr(
            "Measures on all the test photos: original network → modified network. This way you see the effect "
            "of the changes on all the digits, not only on the drawn one."))
        r = base.row(column, pady=(2, 0))
        base.button(r, tr("Reset everything"), self.reset, explanation=tr(
            "Removes all the changes and goes back to the saved network.")).pack(side="left", expand=True,
                                                                                  fill="x", padx=(0, 4))
        base.button(r, tr("Save as model"), self.save, primary=True, explanation=tr(
            "Overwrites the saved model with the modified network: from then on the other tabs use it "
            "too.")).pack(side="left", expand=True, fill="x", padx=(4, 0))

        base.section(column, tr("Selected neuron"))
        self.neuron_name = base.label(column, "", base.TEXT, 10, True, width=300)
        self.neuron_values = base.Tiles(column, ("weighted sum z", "output a", "bias", "incoming weights"),
                                        columns=2, size=10, name=tr("Values of the selected neuron"),
                                        explanation=tr(
            "z = sum of (input x weight) + bias: it is what the neuron \"feels\". a = activation(z): "
            "what the neuron sends forward. For the output neurons a is the probability of the digit."))
        self.weights_drawing = tk.Canvas(column, width=300, height=128, bg=base.BACKGROUND, highlightthickness=0)
        self.weights_drawing.pack(anchor="w", pady=(2, 0))
        base.explain(self.weights_drawing, tr(
            "First layer: on the left the 784 weights of the neuron redrawn as an image (orange = pixels that "
            "switch it on, blue = that switch it off); on the right weights x drawing, that is what it really "
            "finds in your digit. Later layers: the contribution of every neuron of the layer before."),
            tr("Weights of the selected neuron"))
        self.bias = base.slider(column, tr("Shift the bias"), -5, 5, 0.05, 0, self._neuron_changed,
                                show=lambda v: f"{v:+.2f}", explanation=tr(
            "Adds this value to the bias of the selected neuron: with a higher bias it switches on more "
            "easily, with a lower one it struggles more."))
        self.scale = base.slider(column, tr("Multiply the incoming weights"), -2, 3, 0.05, 1, self._neuron_changed,
                                 show=lambda v: f"x {v:.2f}", explanation=tr(
            "Multiplies all the weights that reach the selected neuron. x0 = it doesn't feel anything anymore, "
            "x2 = it feels twice as much, negative values = it looks for the exact opposite."))
        self.off = base.checkbox(column, tr("Switch off this neuron"), self._neuron_changed, explanation=tr(
            "Cuts all the outgoing connections of the neuron: for the next layer it is as if it didn't exist. "
            "If it is an output neuron, that digit can no longer be chosen."))

        base.explanation_box(parent, parent, tr("Move the mouse over a control, a value or the diagram to find "
                                                "out what it means. Click a neuron in the diagram to examine "
                                                "and change it."), 1480, lines=2).grid(
            row=1, column=0, columnspan=4, sticky="ew", padx=10, pady=(10, 0))

    # ------------------------------------------------------------------ network and changes

    def set_net(self, net):
        """New starting network (e.g. after a training): clears all the changes."""
        self.original = net
        try:
            self.test = self.load_test()
            self.X_test = normalize(self.test[0])
            self.original_measures = self._measure(net)
        except FileNotFoundError:
            self.test = self.X_test = self.original_measures = None
        self.board.test = self.test
        self.reset()

    def reset(self):
        """Removes all the changes: back to the original network."""
        self.changes.clear()
        for slider, value in ((self.temperature, 0), (self.weight_noise, 0), (self.pruning, 0),
                              (self.bias, 0), (self.scale, 1)):
            slider.set(value)
        self.off.set(False)
        self._recompute()

    def _apply_changes(self):
        """Starts from a copy of the original network and applies, in order, all the changes."""
        net = copy.deepcopy(self.original)
        noise, pruning = self.weight_noise.get(), self.pruning.get()
        rng = np.random.default_rng(0)  # always the same noise: the slider changes only its strength
        for W in net.weights:
            if noise > 0:
                W += noise * W.std() * rng.normal(size=W.shape).astype(np.float32)
            if pruning > 0:
                W[np.abs(W) < np.quantile(np.abs(W), pruning)] = 0  # away with the smallest weights
        for (layer, neuron), m in self.changes.items():
            net.weights[layer][:, neuron] *= m["scale"]
            net.biases[layer][neuron] += m["bias"]
            if m["off"]:
                if layer < len(net.weights) - 1:
                    net.weights[layer + 1][neuron, :] = 0  # its output no longer reaches anybody
                else:
                    net.weights[layer][:, neuron] = 0      # output neuron: that digit no longer wins
                    net.biases[layer][neuron] = -50
        # Temperature: dividing the final scores = dividing weights and biases of the last layer
        temperature = base.power(self.temperature.get())
        net.weights[-1] /= temperature
        net.biases[-1] /= temperature
        return net

    def _recompute(self, _value=None):
        """Applies all the changes to the original network again and shows what changes."""
        if self.original is None:
            return
        self.modified = self._apply_changes()
        self.board.switched_off = {who for who, m in self.changes.items() if m["off"]}
        self.board.set_net(self.modified)
        if self.original_measures is None:
            self.accuracies = None
            return self.measures.show({name: tr("test photos needed") for name in MEASURES})
        (acc0, loss0, conf0), (acc, loss, conf) = self.original_measures, self._measure(self.modified)
        self.accuracies = (float(acc0), float(acc))
        self.measures.show({"test accuracy": f"{acc0:.1%}   →   {acc:.1%}",
                            "test loss": f"{loss0:.3f}   →   {loss:.3f}",
                            "mean confidence": f"{conf0:.0%}   →   {conf:.0%}"})

    def _measure(self, net):
        """(accuracy, loss, mean confidence) of the network on the test photos."""
        with np.errstate(all="ignore"):
            probabilities = net.predict(self.X_test)
        digits = self.test[1]
        return (np.mean(probabilities.argmax(1) == digits), net.loss(probabilities, one_hot(digits)),
                probabilities.max(1).mean())

    def save(self):
        """Saves the modified network in place of the model: from now on the other tabs use it too."""
        if self.modified is None or not messagebox.askyesno(
                tr("Save as model"), tr("Overwrite the saved model with the modified network?\n"
                                        "The original network will be lost.")):
            return
        self.modified.save()
        self.set_net(NeuralNetwork.load())
        if self.on_save:
            self.on_save()

    # ------------------------------------------------------------------ the selected neuron

    def select(self, who):
        """Called by the drawing board when you click a neuron: prepares its controls and shows its values."""
        m = self.changes.get(who, NO_CHANGE)
        self.bias.set(m["bias"])
        self.scale.set(m["scale"])
        self.off.set(m["off"])
        self._show_neuron()

    def _neuron_changed(self, _value=None):
        """A control of the selected neuron has changed: I record the change and recompute."""
        who = self.board.selected
        if who is None or self.original is None:
            return
        change = {"bias": self.bias.get(), "scale": self.scale.get(), "off": self.off.get()}
        if change == NO_CHANGE:
            self.changes.pop(who, None)
        else:
            self.changes[who] = change
        self._recompute()

    def _show_neuron(self):
        """Writes the values of the selected neuron and draws its incoming weights."""
        who, net, act = self.board.selected, self.modified, self.board.activations
        self.weights_drawing.delete("all")
        self._images.clear()
        if who is None or net is None:
            self.neuron_name.config(text=tr("Click a neuron in the network diagram."))
            return self.neuron_values.clear()
        layer, j = who
        output = layer == len(net.weights) - 1
        self.neuron_name.config(text=tr("Output: the neuron of digit {j}", j=j) if output
                                else tr("Hidden layer {k}, neuron {j}", k=layer + 1, j=j))
        W, b = net.weights[layer][:, j], float(net.biases[layer][j])
        values = {"bias": f"{b:+.3f}", "incoming weights": f"{len(W)}, σ {W.std():.3f}",
                  "weighted sum z": "-", "output a": "-"}
        if act is not None:
            inputs = act[layer]  # the outputs of the layer before (for the first layer: the pixels)
            values["weighted sum z"] = f"{float(inputs @ W) + b:+.3f}"
            values["output a"] = f"{act[layer + 1][j]:.1%}" if output else f"{act[layer + 1][j]:.3f}"
        self.neuron_values.show(values)

        if layer == 0:  # first layer: the 784 weights are a 28x28 image
            self._image(0, tr("what it looks for (its weights)"), W.reshape(28, 28))
            if act is not None:
                self._image(156, tr("what it finds in the drawing"), (W * act[0]).reshape(28, 28))
        elif act is not None:
            self._bars(W * act[layer], tr("contribution of every neuron of the layer before"))
        else:
            self._bars(W, tr("incoming weights"))

    def _image(self, x, title, values):
        """Draws a 28x28 table of numbers (orange = positive, blue = negative), enlarged 4 times."""
        image = Image.fromarray(base.color_map(values)).resize((112, 112), Image.Resampling.NEAREST)
        self._images.append(ImageTk.PhotoImage(image))
        self.weights_drawing.create_text(x, 0, text=title, anchor="nw", fill=base.TEXT_SOFT, font=(base.FONT, 8))
        self.weights_drawing.create_image(x, 14, anchor="nw", image=self._images[-1])

    def _bars(self, values, title):
        """A bar chart: orange upwards = positive, blue downwards = negative."""
        canvas, center = self.weights_drawing, 70
        canvas.create_text(0, 0, text=title, anchor="nw", fill=base.TEXT_SOFT, font=(base.FONT, 8))
        canvas.create_line(0, center, 300, center, fill=base.BORDER)
        largest, width = np.abs(values).max() + 1e-9, 300 / len(values)
        for i, v in enumerate(values):
            canvas.create_rectangle(i * width, center, i * width + max(width - 1, 1),
                                    center - 52 * v / largest, fill=base.ACCENT if v > 0 else base.BLUE, outline="")

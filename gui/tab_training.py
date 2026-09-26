"""
Tab 2 · Training: make the network learn and watch it while it learns.

How it works:
  - The training runs in a separate thread (a job "in parallel"), so the window does not
    freeze. After each epoch the thread sends a copy of the state and the window draws it.
  - The "Optimization" and "Photos" controls end up in self.params, which the thread reads again
    at every epoch: this is why they can be changed even while the network is training.
  - The "Architecture" controls are used to build the network: they apply from the next "New network".
"""
import threading
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from gui import base
from i18n import tr
from neural_net import charts
from neural_net.data import add_noise, augment, count_photos, normalize
from neural_net.network import ACTIVATIONS
from neural_net.training import Trainer

DASHBOARD = ("epoch", "learning rate", "train loss", "validation loss", "train accuracy",
             "validation accuracy", "seconds per epoch")


def _weights_now(ax):
    """What a gaussian of the weights shows now (for Pick): which layer, and its μ and σ."""
    legend = ax.get_legend()  # no legend if the weights exploded
    return base.chart_title(ax) + (f"  ·  {legend.get_texts()[-1].get_text()}" if legend else "")


class TrainingTab(base.Tab):
    title = tr("2 · Training")

    def __init__(self, window):
        super().__init__(window)
        self.trainer = None     # photos + network + history of the measurements (see neural_net/training.py)
        self.state = None       # the last copy of the state that arrived from the thread
        # Parameters read again by the thread at every epoch (they have the names of Trainer.run_epoch)
        self.params = dict(lr=0.05, momentum=0.9, l2=1e-4, dropout=0.0, batch=32,
                           noise=0.0, rotation=12, shift=2)
        self.pause = 0.0                     # slow motion: seconds to wait between one epoch and the next
        self.cosine = True                   # does the learning rate go down by itself towards the end?
        self.in_progress = False
        self.go = threading.Event()          # traffic light: if it is off, the thread stops and waits (pause)
        self.stop_event = threading.Event()  # on = "stop as soon as you can"
        self.last_draw = 0.0                 # when I last redrew the charts...
        self.draw_time = 0.0                 # ...and how long it took
        self.examples = None                 # 8 photos for the preview

        c = base.column(self.frame)
        base.title(c, tr("Training"))

        base.section(c, tr("Architecture  (applies from the next \"New network\")"))
        r = base.row(c)
        self.neurons1 = base.NumberField(r, tr("layer 1"), 1, 512, 64, 8, explanation=tr(
            "Neurons of the first hidden layer. More neurons = the network can learn more complicated shapes, "
            "but it is slower and risks learning by heart."))
        self.neurons2 = base.NumberField(r, tr("layer 2"), 0, 512, 32, 8, explanation=tr(
            "Neurons of the second hidden layer. Put 0 to have a single hidden layer."))
        self.seed = base.NumberField(r, tr("seed"), 0, 9999, 0, 1, explanation=tr(
            "The seed of the random numbers: it decides the initial weights and the order of the photos. Same seed = "
            "same experiment, repeatable. Change it to see how much luck matters."))
        self.activation = base.Choice(c, tr("Activation function"), list(ACTIVATIONS), "relu", explanation=tr(
            "What each hidden neuron does with its weighted sum z.\n"
            "relu: max(0, z), simple and very widely used.   leaky relu: like relu, but lets 0.01·z through "
            "when z is negative, so the neurons do not \"die\".   sigmoid: squashes everything between 0 and 1, "
            "learns more slowly.   tanh: squashes between -1 and 1."))
        self.init_scale = base.slider(c, tr("Width of the initial weights"), -1, 0.7, 0.05, 0,
                                      show=lambda v: f"x {base.power(v):g}", explanation=tr(
            "Multiplies the width of the gaussian the initial weights are drawn from. x1 is the \"right\" "
            "choice (He for relu, LeCun for the others). Try x0.1 (the signal fades out layer after "
            "layer) and x5 (the signal explodes), then look at the gaussians and the corrections per layer."))

        base.section(c, tr("Optimization  (can be changed during training too)"))
        s = self.sliders = {"init_scale": self.init_scale}  # the controls the assistant can move (set_control)
        s["lr"] = base.slider(c, tr("Learning rate"), -4, 0, 0.05, np.log10(0.05),
                              lambda v: self._set(lr=base.power(v)), show=lambda v: f"{base.power(v):g}",
                              explanation=tr(
            "The learning rate: how big each correction of the weights is. Too low: it learns very slowly. "
            "Too high: the loss jumps, the neurons switch off forever or the weights explode."))
        cosine = base.checkbox(c, tr("Slow down towards the end (cosine decay)"),
                               lambda: setattr(self, "cosine", cosine.get()), explanation=tr(
            "If it is on, during the run of epochs the learning rate goes down smoothly from the chosen one to zero "
            "(shaped like a cosine): big steps at the beginning to get close quickly, small steps at the end "
            "to fix the details. It usually gives a few more points of accuracy. The dashboard shows the "
            "rate actually used."))
        cosine.set(True)
        s["momentum"] = base.slider(c, "Momentum", 0, 0.99, 0.01, 0.9, lambda v: self._set(momentum=v),
                                    show=lambda v: f"{v:.2f}", explanation=tr(
            "How much \"momentum\" the corrections keep: with 0.9 each step keeps 90% of the direction "
            "of the previous steps. It helps to go down faster; too high makes it oscillate. 0 = plain "
            "gradient descent."))
        s["l2"] = base.slider(c, tr("L2 regularization"), -7, -1, 0.25, -4,
                              lambda v: self._set(l2=0 if v <= -7 else base.power(v)),
                              show=lambda v: tr("none") if v <= -7 else f"{base.power(v):g}", explanation=tr(
            "Weight decay: at every step all the weights are pushed a little towards zero. Small weights = a "
            "\"simpler\" network, which usually generalizes better. Too much L2 prevents it from learning "
            "(watch the gaussians shrink)."))
        s["dropout"] = base.slider(c, "Dropout", 0, 0.8, 0.05, 0, lambda v: self._set(dropout=v),
                                   show=lambda v: f"{v:.0%}", explanation=tr(
            "At every step it randomly switches off this percentage of hidden neurons, so the network learns not to "
            "depend on a few neurons. The train loss goes up, but the validation often improves. When the "
            "network answers (prediction) no neuron is switched off."))

        base.section(c, tr("Photos during training  (during training too)"))
        self.noise_slider = base.slider(c, tr("Gaussian noise (σ)"), 0, 0.6, 0.01, 0,
                                        lambda v: self._set(noise=v), show=lambda v: f"{v:.2f}",
                                        explanation=tr(
            "A random number drawn from a gaussian of width σ is added to every pixel of the training photos "
            "(it is the last chart at the bottom). A little noise makes the network more robust; too much prevents "
            "it from seeing the digits."))
        self.noise_slider.bind("<ButtonRelease-1>", lambda _: self._redraw())
        s["noise"] = self.noise_slider
        s["rotation"] = base.slider(c, tr("Maximum rotation"), 0, 45, 1, 12, lambda v: self._set(rotation=v),
                                    show=lambda v: f"± {v:.0f}°", explanation=tr(
            "Data augmentation: at every epoch each photo is rotated randomly up to this angle. The network sees "
            "digits that are always a bit different and learns the shape, not the single pixels."))
        s["shift"] = base.slider(c, tr("Maximum shift"), 0, 5, 1, 2, lambda v: self._set(shift=int(v)),
                                 show=lambda v: f"± {v:.0f} px", explanation=tr(
            "Data augmentation: each photo is shifted randomly up to this many pixels, horizontally and "
            "vertically."))
        self.preview = tk.Label(c, bg=base.BACKGROUND, cursor="hand2")
        self.preview.pack(anchor="w", pady=(8, 0))
        self.preview.bind("<Button-1>", lambda _: self._show_preview())
        base.explain(self.preview, tr("Eight photos as the network sees them during training, with the chosen "
                                      "rotation, shift and noise. Click to pick new ones."), tr("Photo preview"))

        base.section(c, tr("Duration"))
        r = base.row(c)
        self.epochs = base.NumberField(r, tr("epochs to do"), 1, 1000, 60, 10, explanation=tr(
            "How many epochs to do when you press Start. One epoch = one full pass over all the training photos."))
        self.batch = base.NumberField(r, tr("photos per mini-batch"), 1, 512, 32, 8, explanation=tr(
            "How many photos the network looks at before correcting the weights. Small batches = many "
            "\"noisy\" corrections; big batches = few, more precise corrections. It can be changed during training too."))
        self.batch.var.trace_add("write", lambda *_: self._set(batch=self.batch.value()))
        base.slider(c, tr("Slow motion"), 0, 1000, 50, 0, lambda v: setattr(self, "pause", v / 1000),
                    show=lambda v: tr("off") if v == 0 else tr("{ms:.0f} ms between epochs", ms=v), explanation=tr(
            "A pause between one epoch and the next, to follow the charts calmly."))

        buttons = tk.Frame(c, bg=base.BACKGROUND)
        buttons.pack(fill="x", pady=(12, 0))
        buttons.columnconfigure((0, 1), weight=1, uniform="buttons")
        self.start_button = base.button(buttons, tr("Start"), self.start_or_pause, primary=True, explanation=tr(
            "Starts the training, or pauses and resumes it. If the network has not done any epoch "
            "yet, it first recreates it with the chosen architecture."))
        self.step_button = base.button(buttons, tr("+1 epoch"), lambda: self.start(1), explanation=tr(
            "Does a single epoch and stops: to follow the training one step at a time."))
        self.stop_button = base.button(buttons, tr("Stop"), self.stop_training, explanation=tr(
            "Stops the training at the end of the current epoch and saves the model."))
        self.new_button = base.button(buttons, tr("New network"), self.new_network, explanation=tr(
            "Throws away the current network and creates a new one with random weights, using the chosen architecture."))
        for k, widget in enumerate((self.start_button, self.step_button, self.stop_button, self.new_button)):
            widget.grid(row=k // 2, column=k % 2, sticky="ew", padx=(0, 6) if k % 2 == 0 else 0, pady=(0, 6))

        # On the right: the dashboard with the numbers of the last epoch, the status, the explanations and the charts
        right = base.right_area(self.frame)
        self.dashboard = base.Tiles(right, DASHBOARD, name=tr("Numbers of the last epoch"), explanation=tr(
            "The numbers of the last epoch. Loss = how much it gets wrong (lower is better). Accuracy = how many photos "
            "it guesses right. \"train\" are the photos it learns from, \"validation\" the ones it never sees while "
            "learning: if train improves and validation does not, it is learning by heart (overfitting)."))
        self.status = base.label(right, "", base.TEXT, 10, pady=(0, 6), width=1200)
        base.explanation_box(right, self.frame, tr("Move the mouse over a control, a value or a chart "
                                                   "to find out what it means."), 1220).pack(fill="x", pady=(0, 6))
        self.fig, self.canvas = base.figure(right)
        base.explain(self.canvas.get_tk_widget(), tr(
            "At the top: loss curve, accuracy, strength of the corrections (gradient) of each layer with the "
            "percentage of inactive neurons, and what each neuron of the first layer looks for (red = pixels that "
            "switch it on, blue = pixels that switch it off). At the bottom: the gaussians, that is how the "
            "weights of each layer are distributed now (orange) compared to the beginning (dashed)."),
            tr("Training charts"))
        base.explain_charts(self.canvas, {
            "loss": (tr("Loss curve"), tr(
                "How much the network gets wrong, epoch after epoch (lower = better): the light line is every "
                "single mini-batch, blue the training photos, orange the validation ones. If the orange goes back "
                "up while the blue keeps going down, the network is learning by heart.")),
            "accuracy": (tr("Accuracy curve"), tr(
                "The share of photos guessed right at the end of every epoch: blue the training photos, orange the "
                "validation ones. In the title, the last validation accuracy."), base.chart_title),
            "corrections": (tr("Strength of the corrections"), tr(
                "How big the corrections (gradients) of each layer are, epoch after epoch, on a logarithmic scale. "
                "A line that collapses is a layer that stops learning, one that shoots up makes the network "
                "unstable. In brackets, the share of inactive neurons."), base.chart_legend),
            "first layer": (tr("What the first layer looks for"), tr(
                "Each small square is a neuron of the first layer: its 784 weights redrawn as a 28x28 photo. Red = "
                "pixels that switch it on, blue = pixels that switch it off. At the start it is noise; while the "
                "network learns, the shapes of the strokes appear.")),
            "weights": (tr("Gaussian of the weights"), tr(
                "How the weights of this layer are distributed: the histogram, the gaussian now (orange) and the "
                "one at the start (dashed). While learning it usually widens a little; if it explodes, the learning "
                "rate is too high."), _weights_now),
            "noise": (tr("Gaussian of the noise"), tr(
                "The gaussian from which the noise added to each pixel of the training photos is drawn: the wider "
                "it is (the bigger σ), the more spoiled the photos the network learns from."), base.chart_legend)})

    # ------------------------------------------------------------------ controls

    def _set(self, **values):
        """Updates the parameters read by the thread; if the photos change, redoes the preview."""
        self.params.update(values)
        if {"noise", "rotation", "shift"} & values.keys():
            self._show_preview()

    def set_control(self, name, value):
        """Moves a control as if you had done it with the mouse (the assistant uses it to apply a hint).
        name = one of self.sliders or "batch"; value = the real value, for example 0.05 for the learning rate."""
        if name == "batch":
            return self.batch.var.set(str(int(value)))
        logarithmic = name in ("lr", "l2", "init_scale")  # these sliders move on the exponent of 10
        self.sliders[name].set(np.log10(value) if logarithmic else value)

    def chosen_architecture(self):
        """The architecture chosen in the controls (it applies from the next "New network")."""
        return {"hidden": [n for n in (self.neurons1.value(), self.neurons2.value()) if n > 0],
                "activation": self.activation.value(), "init_scale": base.power(self.init_scale.get()),
                "seed": self.seed.value()}

    def refresh(self):
        """When the photos change: starts again with a new network (or a message if there are no photos)."""
        self.trainer = self.state = self.examples = None
        if count_photos("train"):
            return self.new_network()
        base.message(self.fig, tr("Download the photos first in the \"1 · Data\" tab"))
        self.canvas.draw()
        self.dashboard.clear()
        self.status.config(text="")
        self._show_preview()
        self._update_buttons()

    def new_network(self):
        """A new network with random weights: they depend on architecture, initial width and seed."""
        if self.in_progress:
            return
        hidden = [n for n in (self.neurons1.value(), self.neurons2.value()) if n > 0]
        try:
            self.trainer = Trainer(hidden, self.activation.value(), base.power(self.init_scale.get()),
                                   self.seed.value(), self.window.photos("train"))
        except FileNotFoundError as error:
            return messagebox.showwarning(tr("Missing photos"), str(error))
        photos, digits = self.trainer.train_photos, self.trainer.train_digits
        self.examples = np.stack([photos[digits == d][0] for d in range(8)])  # a 0, a 1, ... a 7
        self.state = self.trainer.snapshot()
        self._show_preview()
        self._redraw()
        self.dashboard.clear()
        net = self.trainer.net
        self.status.config(text=tr("New network with random weights: {layers}, activation {activation}, "
                                   "{n} parameters to learn. Press Start.",
                                   layers=" → ".join(map(str, net.layers)), activation=net.activation,
                                   n=net.n_parameters))
        self._update_buttons()

    def start_or_pause(self):
        """The main button: starts; if it is already running it pauses or resumes."""
        if not self.in_progress:
            return self.start(self.epochs.value())
        if self.go.is_set():
            self.go.clear()
            self._redraw()
            self.status.config(text=tr("Paused: feel free to change the parameters, then resume."))
        else:
            self.go.set()
        self._update_buttons()

    def start(self, n_epochs):
        """Starts n_epochs of training in the separate thread."""
        if self.in_progress:
            return
        if self.trainer is None or self.trainer.epoch == 0:
            self.new_network()  # network never trained: I recreate it with the architecture chosen now
            if self.trainer is None:
                return
        if self.trainer.exploded:
            return messagebox.showinfo(tr("Network exploded"), tr("The weights have become infinite. Lower the "
                                                                  "learning rate and press \"New network\"."))
        self.stop_event.clear()
        self.go.set()
        self.in_progress = True
        self._update_buttons()
        trainer = self.trainer
        base.in_background(self.frame, lambda send: self._train(send, trainer, n_epochs), self._news)

    def stop_training(self):
        self.stop_event.set()
        self.go.set()  # if it was paused I wake it up, so it can stop

    # ------------------------------------------------------------------ training thread

    def _train(self, send, trainer, n_epochs):
        """Runs in the separate thread: it does NOT touch the window, it only sends news."""
        for i in range(n_epochs):
            self.go.wait()  # when paused, it waits here
            if self.stop_event.is_set():
                return
            params = dict(self.params)
            if self.cosine:  # the learning rate goes down smoothly to zero at the end of the run
                params["lr"] *= 0.5 * (1 + np.cos(np.pi * i / n_epochs))
            trainer.run_epoch(**params)
            send("epoch", trainer.snapshot())
            if trainer.exploded or self.stop_event.wait(self.pause):  # Stop interrupts the slow motion
                return

    def _news(self, kind, data):
        """Receives the news from the thread (here we are back in the window)."""
        if kind == "epoch":
            self.state = data
            self._show_dashboard()
            # Redrawing the charts is costly (about half a second, and meanwhile the window is frozen):
            # I redo it only after at least three times that time, so the window stays responsive
            if time.monotonic() - self.last_draw > max(0.4, 3 * self.draw_time):
                self._redraw()
            return
        # "done" or "error": the training is over
        self.in_progress = False
        self._update_buttons()
        self._redraw()
        if kind == "error":
            messagebox.showerror(tr("Error during training"), str(data))
        elif self.trainer.exploded:
            self.status.config(text=tr("The network exploded: the weights have become infinite because the "
                                       "learning rate is too high. Lower it and press \"New network\"."))
        elif self.trainer.epoch:
            self.trainer.net.save()
            text = tr("Model saved: try it in tabs 3 and 4.")
            if self.trainer.history["val_acc"][-1] < 0.2:
                text += "  " + tr("Warning: the network is not learning (it guesses at random, ~10%). Look at the "
                                  "corrections per layer and the inactive neurons: usually the learning rate is "
                                  "too high.")
            self.status.config(text=text)
            self.window.model_changed()

    # ------------------------------------------------------------------ drawing

    def _show_dashboard(self):
        h = self.state["history"]
        self.dashboard.show({
            "epoch": len(h["val_acc"]),
            "learning rate": f"{h['lr'][-1]:.2g}",
            "train loss": f"{h['train_loss'][-1]:.3f}",
            "validation loss": f"{h['val_loss'][-1]:.3f}",
            "train accuracy": f"{h['train_acc'][-1]:.1%}",
            "validation accuracy": f"{h['val_acc'][-1]:.1%}",
            "seconds per epoch": f"{h['seconds'][-1]:.2f}",
        })

    def _redraw(self):
        begin = time.monotonic()
        if self.state:
            charts.training(self.fig, self.state, noise=self.noise_slider.get())
            self.canvas.draw()
        self.last_draw = time.monotonic()
        self.draw_time = self.last_draw - begin

    def _show_preview(self):
        """8 photos as the network sees them now: with the chosen rotation, shift and noise."""
        rng, p = np.random.default_rng(), self.params
        if self.examples is None:  # no photos: 8 empty squares, crossed out
            X = np.zeros((8, 784))
        else:
            X = add_noise(normalize(augment(self.examples, rng, p["rotation"], p["shift"])), p["noise"], rng)
        grid = charts.mosaic(X.reshape(-1, 28, 28) * 255, columns=8, border=1, empty=58).astype(np.uint8)
        image = Image.fromarray(grid)
        image = image.resize((int(image.width * 1.25), int(image.height * 1.25)), Image.Resampling.BILINEAR)
        if self.examples is None:
            image = base.crossed_out_image(image)
        self._preview_photo = ImageTk.PhotoImage(image)  # it must be kept, otherwise Tkinter deletes it
        self.preview.config(image=self._preview_photo)

    def _update_buttons(self):
        if self.in_progress:
            self.start_button.config(text=tr("Pause") if self.go.is_set() else tr("Resume"))
        else:
            self.start_button.config(text=tr("Continue") if self.trainer and self.trainer.epoch else tr("Start"))
        for widget in (self.step_button, self.new_button):
            widget.config(state="disabled" if self.in_progress else "normal")
        self.stop_button.config(state="normal" if self.in_progress else "disabled")

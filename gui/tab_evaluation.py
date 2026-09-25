"""
Tab 3 · Evaluation: how good is the network on photos it has never seen?

The test photos have never been used for learning. Here you can also "spoil" them on purpose
(noise, rotation, thinner or thicker stroke) to see how well the network holds up,
and choose a confidence threshold below which the network prefers to say "I don't know".
The big chart can be switched from confusion matrix to "point map": each test photo
becomes a point on a plane, and you can see which photos the network considers similar.
"""
import tkinter as tk
from tkinter import messagebox

import numpy as np

from gui import base
from i18n import tr
from neural_net import charts
from neural_net.data import one_hot
from neural_net.network import NeuralNetwork
from neural_net.storage import MODEL_FILE
from neural_net.training import robustness, test_on

THICKNESSES = {-2: tr("very thin"), -1: tr("thin"), 0: tr("normal"), 1: tr("thick"), 2: tr("very thick")}


class EvaluationTab(base.Tab):
    title = tr("3 · Evaluation")

    def __init__(self, window):
        super().__init__(window)
        self.net = self.photos = self.digits = self.curves = self.probabilities = self.map_photos = None
        self.maps = {}  # point maps already computed: (layer, method, alterations) -> (points, axes)
        self.computing = set()  # maps being computed in the background
        self.ax = self.points = self.background = self.point_label = self.nearest = None

        c = base.column(self.frame)
        base.title(c, tr("Evaluation on the test set"))
        self.accuracy = tk.Label(c, text="-", bg=base.BACKGROUND, fg=base.ACCENT, font=(base.FONT, 32, "bold"))
        self.accuracy.pack(anchor="w")
        base.explain(self.accuracy, tr("Accuracy: the percentage of test photos recognized. The test photos "
                                       "have never been used for learning, so they measure how the network "
                                       "copes with new digits."),
                     tr("Test accuracy"), lambda: self.accuracy.cget("text"))
        self.description = base.label(c)
        self.numbers = base.Tiles(c, ("loss", "average confidence", "errors", "very confident errors"), columns=2,
                                  size=11, name=tr("Measures on the test photos"), explanation=tr(
            "Loss: how much it gets wrong on average (it also counts how sure it was). Confidence: the average "
            "probability given to the chosen answer. Very confident errors: wrong photos with more than 90% "
            "confidence, the most dangerous errors."))

        base.section(c, tr("Spoil the test photos"))
        self.noise = base.slider(c, tr("Gaussian noise (σ)"), 0, 0.6, 0.01, 0, show=lambda v: f"{v:.2f}",
                                 explanation=tr("Adds gaussian noise to the test photos (always the same "
                                                "noise, for repeatable results). Compare a network trained "
                                                "with and without noise: which one holds up better?"))
        self.rotation = base.slider(c, tr("Rotation"), -45, 45, 1, 0, show=lambda v: f"{v:+.0f}°",
                                    explanation=tr("Rotates all the test photos by this angle. The MNIST digits "
                                                   "are almost all straight: does the network recognize them even "
                                                   "when tilted? Does data augmentation with rotation help?"))
        self.thickness = base.slider(c, tr("Stroke thickness"), -2, 2, 1, 0, show=lambda v: THICKNESSES[int(v)],
                                     explanation=tr("Makes the stroke of the digits thicker or thinner (one pass "
                                                    "of a filter for each notch)."))
        for widget in (self.noise, self.rotation, self.thickness):
            widget.bind("<ButtonRelease-1>", lambda _: self._evaluate())  # recompute when you let go of the slider

        base.section(c, tr("Confidence threshold"))
        self.threshold = base.slider(c, tr("Answer only if at least this sure"), 0, 0.99, 0.01, 0,
                                     lambda v: self._show_threshold(), show=lambda v: f"{v:.0%}", explanation=tr(
            "The network answers only when the highest probability exceeds the threshold, otherwise it says "
            "\"I don't know\". Raising it, it answers fewer photos but makes fewer mistakes: this is how networks "
            "are used when a mistake is costly."))
        self.threshold_result = base.label(c, "", base.TEXT)

        base.section(c, tr("Accuracy per digit"))
        self.bars = base.Bars(c, length=190, height=22)
        base.explain(self.bars.canvas, tr("How many photos of each digit are recognized (orange = at least 80%)."),
                     tr("Accuracy per digit"))

        right = base.right_area(self.frame)
        base.explanation_box(right, self.frame, tr("Move the mouse over a control, a value or a chart "
                                                   "to find out what it means."), 1220).pack(fill="x", pady=(0, 6))

        bar = base.row(right, pady=(0, 6))
        self.view = tk.StringVar(value="matrix")
        base.choice_buttons(bar, tr("View"), [(tr("Confusion matrix"), "matrix"), (tr("Point map"), "map")],
                            self.view, self._draw, explanation=tr(
            "Matrix: counts how many photos of each digit end up in each answer. Point map: each test photo is "
            "a point, and the photos that the network \"sees\" as similar are close together (like the embedding "
            "maps in the playgrounds of language models). Move the mouse over the points to see the photos."))
        self.layer_place = tk.Frame(bar, bg=base.BACKGROUND)  # the layer buttons change with the network
        self.layer_place.pack(side="left")
        self.method = tk.StringVar(value="t-SNE")
        base.choice_buttons(bar, tr("Projection"), [("PCA", "PCA"), ("t-SNE", "t-SNE")], self.method,
                            self._choose_map, explanation=tr(
            "Each photo has many coordinates (784 pixels, or one per neuron): to draw it, it must be squashed onto "
            "a plane. PCA looks at the cloud from the side where it is most spread out: it is instant and keeps the "
            "large distances, but the groups overlap. t-SNE moves the points until each one has next to it the "
            "same photos it had close before: it separates the groups well (it takes a few seconds), but the "
            "distance between different groups has no meaning."))

        self.fig, self.canvas = base.figure(right)
        base.explain(self.canvas.get_tk_widget(), tr(
            "Top left the confusion matrix (rows = true digit, columns = predicted digit; the diagonal holds the "
            "right answers) or the point map (color = true digit, red ring = wrong, the line goes towards the "
            "group of the predicted digit). Below: how the accuracy drops with more noise or more rotation (the "
            "dashed line is the current choice). On the right the wrong photos."), tr("Evaluation charts"))
        self.canvas.mpl_connect("draw_event", self._save_background)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def refresh(self):
        """The model or the photos have changed: I reload everything and recompute the robustness curves."""
        if not MODEL_FILE.exists():
            return self._empty(tr("No trained model:\ntrain the network in the \"2 · Training\" tab"))
        try:
            self.photos, self.digits = self.window.photos("test")
        except FileNotFoundError:
            return self._empty(tr("Download the photos first in the \"1 · Data\" tab"))
        self.net = NeuralNetwork.load()
        self.map_photos = charts.map_photos(len(self.digits))
        self.maps = {}
        self._compute_curves()
        self._layer_buttons()
        self._evaluate()

    def _compute_curves(self):
        """The robustness curves test all the photos 26 times: with many photos it takes a few seconds,
        so I compute them in the background (meanwhile the curves say "computing...")."""
        self.curves = {"noise": None, "rotation": None}
        net, photos, digits = self.net, self.photos, self.digits

        def on_news(kind, data):
            if kind == "curves" and self.net is net:  # if the network changed in the meantime, they are useless
                self.curves = data
                self._draw()
            elif kind == "error":
                messagebox.showerror(tr("Robustness curves"), str(data))

        base.in_background(self.frame, lambda send: send("curves", robustness(net, photos, digits)), on_news)

    def _layer_names(self):
        """The name of each layer of the network: pixels, hidden layers, output."""
        n = len(self.net.layers)
        return [tr("pixels")] + [tr("layer {k}", k=k) for k in range(1, n - 1)] + [tr("output")]

    def _layer_buttons(self):
        """One button for each layer of the network: pixels, hidden layers, output."""
        for old in self.layer_place.winfo_children():
            old.destroy()
        self.layer = tk.StringVar(value=str(len(self.net.layers) - 2))  # at the start the last hidden layer
        base.choice_buttons(self.layer_place, tr("Layer"), [(f"{name} ({neurons})", str(k)) for k, (name, neurons)
                                                            in enumerate(zip(self._layer_names(), self.net.layers))],
                            self.layer, self._choose_map, explanation=tr(
            "Which numbers to use as the coordinates of each photo. Pixels: the photo as it is, the digits are "
            "mixed up. Hidden layers: the activations of the neurons, that is how the network \"sees\" the photo; "
            "layer after layer the groups separate. Output: the 10 final probabilities."))

    def _evaluate(self):
        """Puts the network to the test with the chosen alterations and updates numbers and charts."""
        if self.net is None:
            return
        noise, rotation, thickness = self.noise.get(), self.rotation.get(), int(self.thickness.get())
        X, self.probabilities = test_on(self.net, self.photos, noise, rotation, thickness)
        digits, confidence = self.digits, self.probabilities.max(axis=1)
        right = self.probabilities.argmax(axis=1) == digits
        self.accuracy.config(text=f"{right.mean():.1%}")
        self.description.config(text=tr("{right} photos recognized out of {total}\nnetwork {layers}, {activation}",
                                        right=right.sum(), total=len(digits),
                                        layers=" → ".join(map(str, self.net.layers)), activation=self.net.activation))
        self.numbers.show({"loss": f"{self.net.loss(self.probabilities, one_hot(digits)):.3f}",
                           "average confidence": f"{confidence.mean():.0%}",
                           "errors": int(np.sum(~right)),
                           "very confident errors": int(np.sum(~right & (confidence > 0.9)))})
        per_digit = [right[digits == d].mean() for d in range(10)]
        self.bars.show(per_digit, [base.ACCENT if v >= 0.8 else base.BLUE for v in per_digit])
        self._show_threshold()
        self.X, self.alterations = X, (noise, rotation, thickness)
        self._draw()

    def _choose_map(self):
        """Choosing a layer or a projection switches directly to the map."""
        self.view.set("map")
        self._draw()

    def _draw(self):
        """Redraws the charts. The t-SNE map takes a few seconds: I compute it in the background
        (meanwhile the map says "computing...") and keep it aside for the next times."""
        if self.net is None:
            return
        points_map = None
        if self.view.get() == "map":
            layer, method = int(self.layer.get()), self.method.get()
            key = (layer, method, self.alterations)
            points_map = {"points": None, "axes": None, "name": self._layer_names()[layer], "method": method,
                          "photos": self.map_photos}
            if key in self.maps:
                points_map["points"], points_map["axes"] = self.maps[key]
            else:
                self._compute_map(key, self.net.forward(self.X[self.map_photos])[layer])
        self.point_label = self.nearest = None
        self.ax = charts.evaluation(self.fig, self.X.reshape(-1, 28, 28), self.digits, self.probabilities,
                                    self.curves, {"noise": self.alterations[0], "rotation": self.alterations[1]},
                                    points_map=points_map)
        self.points = points_map and points_map["points"]
        self.canvas.draw()

    def _compute_map(self, key, values):
        if key in self.computing:
            return
        self.computing.add(key)
        net = self.net

        def job(send):
            send("map", charts.project_2d(values, key[1]))

        def on_news(kind, data):
            if kind == "map" and self.net is net:  # if the network changed in the meantime, the map is useless
                self.maps[key] = data
                self._draw()
            elif kind in ("done", "error"):
                self.computing.discard(key)
                if kind == "error":
                    messagebox.showerror(tr("Point map"), tr("I could not compute the map:\n{error}", error=data))

        base.in_background(self.frame, job, on_news)

    # ------------------------------------------------ the box above the point under the mouse

    def _save_background(self, _event):
        """After each full drawing I keep a "photograph" of the figure: when the mouse moves over the points
        I redraw only the box on top of this photograph, instead of the whole figure (much faster)."""
        self.background = self.canvas.copy_from_bbox(self.fig.bbox)

    def _on_motion(self, event):
        """Mouse on the map: I show the photo closest to the pointer (within 10 pixels) and what the network thinks."""
        if self.points is None or self.background is None:
            return
        nearest = None
        if event.inaxes is self.ax:
            distances = np.hypot(*(self.ax.transData.transform(self.points) - (event.x, event.y)).T)
            if distances.min() < 10:
                nearest = int(distances.argmin())
        if nearest == self.nearest:
            return
        self.nearest = nearest
        if self.point_label is not None:
            self.point_label.remove()
            self.point_label = None
        self.canvas.restore_region(self.background)
        if nearest is not None:
            photo = self.map_photos[nearest]  # the map may show only some of the test photos
            p, true = self.probabilities[photo], self.digits[photo]
            first, second = np.argsort(p)[::-1][:2]
            right = first == true
            text = tr("true {true} → predicted {first} ({p1:.0%})\n2nd choice: {second} ({p2:.0%})",
                      true=true, first=first, p1=p[first], second=second, p2=p[second])
            self.point_label = charts.point_label(self.ax, self.points[nearest], self.X[photo].reshape(28, 28),
                                                  text, base.GREEN if right else base.RED)
            self.ax.draw_artist(self.point_label)
            self.frame.explanations.config(text=tr(
                "Test photo no. {n}, true digit {true}. The network answers {first} with {p1:.0%} confidence "
                "(second choice: {second}, {p2:.0%}).",
                n=photo, true=true, first=first, p1=p[first], second=second, p2=p[second]) + " " + (
                tr("Right answer.") if right else
                tr("Wrong answer: the red line points towards the group of the {first}s, where the network "
                   "\"moved\" it. The farther the point is from its group, the more different the network sees it "
                   "from the others.", first=first)))
        self.canvas.blit(self.fig.bbox)

    def _show_threshold(self):
        """With a threshold the network says "I don't know" when it is not sure enough."""
        if self.probabilities is None:
            return
        answers = self.probabilities.max(axis=1) >= self.threshold.get()
        right = self.probabilities.argmax(axis=1) == self.digits
        if answers.any():
            text = tr("It answers {answered:.0%} of the photos and on those it guesses {acc:.1%} right "
                      "(without threshold: {all_acc:.1%}).",
                      answered=answers.mean(), acc=right[answers].mean(), all_acc=right.mean())
        else:
            text = tr("With this threshold it never answers.")
        self.threshold_result.config(text=text)

    def _empty(self, text):
        self.net = self.probabilities = self.points = None
        self.accuracy.config(text="-")
        self.description.config(text="")
        self.numbers.clear()
        self.threshold_result.config(text="")
        self.bars.show([None] * 10, [base.ACCENT] * 10)
        base.message(self.fig, text)
        self.canvas.draw()

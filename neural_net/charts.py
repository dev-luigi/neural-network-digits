"""
All the charts of the project, made with matplotlib.

Each function draws inside a "figure" that it receives: this way the same chart is used both
in the graphical interface and in the terminal commands (which also save it to a file).
Figures must be created with layout="constrained" (it arranges the space between the charts by itself).
"""
import numpy as np
from matplotlib import colormaps, patheffects, rcParams
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
from matplotlib.ticker import PercentFormatter

from i18n import tr

DIGIT_COLORS = colormaps["tab10"](np.arange(10))  # one color for each digit in the map of points
ERROR_RED = "#e05d6f"
MAP_PHOTOS = 1000  # at most this many photos on the map of points (see map_photos)
# Cells "inside the network": blue = negative, dark = zero, orange = positive
VALUE_CMAP = LinearSegmentedColormap.from_list("values", ["#4da3ff", "#1d2129", "#ffb347"])


def mosaic(images, columns, border=0, empty=0.0):
    """Puts many 28x28 images next to each other, in a grid with `columns` columns."""
    rows = int(np.ceil(len(images) / columns))
    step = 28 + border
    grid = np.full((rows * step + border, columns * step + border), empty, dtype=np.float32)
    for k, image in enumerate(images):
        y, x = border + (k // columns) * step, border + (k % columns) * step
        grid[y:y + 28, x:x + 28] = image
    return grid


def gaussian(x, mean, sigma):
    """The "bell" curve (normal distribution) with this mean and this width sigma."""
    sigma = max(sigma, 1e-12)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _finite(values):
    """Replaces the infinities with "empty" (NaN), which matplotlib skips: needed if the network explodes."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


# ---------------------------------------------------------------- dataset exploration (pre-training)

def exploration(fig, photos, digits, test_photos, test_digits, title=None):
    """What the dataset looks like: examples, how many photos per digit, "average digit", pixel values."""
    fig.clear()
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1])

    ax = fig.add_subplot(grid[:, 0])
    examples = np.zeros((10, 10, 28, 28))  # 10 rows (one per digit) x 10 examples
    for digit in range(10):
        of_digit = photos[digits == digit][:10]
        examples[digit, :len(of_digit)] = of_digit
    ax.imshow(mosaic(examples.reshape(-1, 28, 28), 10), cmap="gray")
    ax.set_title(tr("10 sample photos for each digit"))
    ax.set_yticks(np.arange(10) * 28 + 14, labels=range(10))
    ax.set_xticks([])

    ax = fig.add_subplot(grid[0, 1])
    ax.bar(np.arange(10) - 0.2, np.bincount(digits, minlength=10), width=0.4, label="train")
    ax.bar(np.arange(10) + 0.2, np.bincount(test_digits, minlength=10), width=0.4, label="test")
    ax.set(title=tr("How many photos per digit"), xlabel=tr("digit"), xticks=range(10))
    ax.legend()

    ax = fig.add_subplot(grid[0, 2])
    ax.imshow(mosaic(np.stack([photos[digits == d].mean(axis=0) for d in range(10)]), 5), cmap="magma")
    ax.set_title(tr("The \"average digit\" of each class"))
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 1:])
    ax.hist(photos.ravel(), bins=64, color="C0")
    ax.set(yscale="log", title=tr("Pixel values (log scale): almost all black (0), the stroke is close to 255"),
           xlabel=tr("pixel value (0 = black, 255 = white)"), ylabel=tr("number of pixels"))


# ---------------------------------------------------------------- training

def training(fig, state, noise=None, title=None):
    """The whole training in one figure:
      top      loss curve, accuracy, strength of the corrections per layer, what the first layer looks for
      bottom   the gaussians: how the weights of each layer are distributed (and the noise on the photos)
    state = copy of the training state (see Trainer.snapshot)."""
    history, weights, initial_weights = state["history"], state["weights"], state["initial_weights"]
    fig.clear()
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.1, 1])
    ax_loss, ax_acc, ax_grad, ax_map = top.subplots(1, 4, width_ratios=[1.3, 1, 1, 0.75])
    axes = bottom.subplots(1, len(weights) + (noise is not None), squeeze=False)[0]
    bottom.suptitle(tr("The gaussians: how the weight values are distributed") if noise is None else
                    tr("The gaussians: how the weight and noise values are distributed"), fontsize=11)
    epochs = np.arange(1, len(history["val_acc"]) + 1)

    # Loss curve: the light line is every single mini-batch, the thick ones the end of each epoch
    ax_loss.plot(history["batch_x"], _finite(history["batch_loss"]), color="C0", alpha=0.25, lw=0.8,
                 label=tr("every mini-batch"))
    ax_loss.plot(epochs, _finite(history["train_loss"]), color="C0", lw=2, label="train")
    ax_loss.plot(epochs, _finite(history["val_loss"]), color="C1", lw=2, label="validation")
    ax_loss.set(title=tr("Loss curve (lower = better)"), xlabel=tr("epoch"))
    ax_loss.set_ylim(bottom=0)

    # Accuracy: if the train one goes up and the validation one does not, the network learns by heart (overfitting)
    ax_acc.plot(epochs, history["train_acc"], color="C0", lw=2, label="train")
    ax_acc.plot(epochs, history["val_acc"], color="C1", lw=2, label="validation")
    ax_acc.set(title=tr("Accuracy (validation {acc:.1%})", acc=history["val_acc"][-1]) if len(epochs) else tr("Accuracy"),
               xlabel=tr("epoch"), ylim=(0, 1.02))
    ax_acc.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Strength of the corrections (gradient) of each layer: if it collapses the layer stops learning,
    # if it explodes the network becomes unstable. In brackets the "inactive" neurons (always the same output).
    for k, strength in enumerate(np.array(history["gradients"]).T):
        if k < len(history["inactive"][-1]):
            name = tr("layer {n} ({share:.0%} inactive)", n=k + 1, share=history["inactive"][-1][k])
        else:
            name = tr("layer {n}", n=k + 1)
        ax_grad.plot(epochs, _finite(strength), lw=2, label=name)
    ax_grad.set(title=tr("Strength of the corrections per layer"), xlabel=tr("epoch"), yscale="log")

    for ax in (ax_loss, ax_acc, ax_grad):
        ax.grid(alpha=0.3)
        if len(epochs):
            ax.legend(fontsize=7)
    if not len(epochs):
        ax_loss.text(0.5, 0.5, tr("press Start to begin"), transform=ax_loss.transAxes, ha="center", alpha=0.6)

    _first_layer_map(ax_map, weights[0])
    for number, (ax, W, W0) in enumerate(zip(axes, weights, initial_weights), start=1):
        _weights_gaussian(ax, W, W0, number)
    if noise is not None:
        _noise_gaussian(axes[-1], noise)


def _first_layer_map(ax, W):
    """The 784 input weights of each neuron, redrawn as a 28x28 image: they show which
    "shape" that neuron looks for in the photo (red = it wants it, blue = it does not want it)."""
    n = W.shape[1]
    normalized = (W / (np.abs(W).max(axis=0) + 1e-9)).T.reshape(n, 28, 28)
    ax.imshow(mosaic(normalized, int(np.ceil(np.sqrt(n))), border=1, empty=np.nan),
              cmap=colormaps["RdBu_r"].with_extremes(bad=(0, 0, 0, 0)), vmin=-1, vmax=1)
    ax.set_title(tr("What each of the {n}\nfirst-layer neurons\n\"looks for\"", n=n), fontsize=10)
    ax.axis("off")


def _weights_gaussian(ax, W, W0, number):
    """Histogram of the weights of one layer, with the current gaussian and the starting one."""
    ax.set_title(tr("Weights of layer {number}  ({n_in} → {n_out})", number=number, n_in=W.shape[0],
                    n_out=W.shape[1]), fontsize=10)
    ax.set_yticks([])
    ax.set_xlabel(tr("weight value"), fontsize=9)
    W, W0 = W.ravel(), W0.ravel()
    if not np.all(np.isfinite(W)):
        ax.text(0.5, 0.5, tr("weights exploded!\nlearning rate too high"), transform=ax.transAxes,
                ha="center", va="center", color="C3", fontsize=11)
        return
    mean, sigma, sigma0 = W.mean(), W.std(), W0.std()
    limit = max(4 * sigma0, np.percentile(np.abs(W), 99.5))
    x = np.linspace(-limit, limit, 300)
    ax.hist(W, bins=np.linspace(-limit, limit, 60), density=True, color="C0", alpha=0.45)
    ax.plot(x, gaussian(x, 0, sigma0), "--", color="gray", lw=1.2, label=tr("at the start"))
    ax.plot(x, gaussian(x, mean, sigma), color="C1", lw=2, label=tr("now: μ={mean:+.3f}  σ={sigma:.3f}",
                                                                    mean=mean, sigma=sigma))
    ax.legend(fontsize=7, loc="upper right")


def _noise_gaussian(ax, sigma):
    """The gaussian from which the noise added to each pixel is drawn."""
    ax.set(title=tr("Noise added to each pixel"), xlim=(-1, 1), xticks=[-1, -0.5, 0, 0.5, 1], yticks=[])
    ax.title.set_fontsize(10)
    ax.set_xlabel(tr("pixel change (0-1 scale)"), fontsize=9)
    if sigma <= 0:
        ax.axvline(0, color="C2", lw=2)
        ax.text(0.5, 0.75, tr("σ = 0: no noise"), transform=ax.transAxes, ha="center", fontsize=9)
        return
    x = np.linspace(-1, 1, 300)
    ax.fill_between(x, gaussian(x, 0, sigma), color="C2", alpha=0.3)
    ax.plot(x, gaussian(x, 0, sigma), color="C2", lw=2, label=f"μ=0  σ={sigma:.2f}")
    ax.legend(fontsize=7, loc="upper right")


# ---------------------------------------------------------------- evaluation

def evaluation(fig, photos, digits, probabilities, curves=None, current=None, title=None, points_map=None):
    """Confusion matrix (or map of points), robustness curves (if any) and the wrong photos.
    curves = {"noise": ..., "rotation": ...} (a curve = None: still being computed).
    current = {"noise": ..., "rotation": ...}: the alterations in use, marked on the curves.
    points_map = {"points", "axes", "name", "method", "photos"}: if given, it draws the map of points instead of the
    matrix (with "points" = None it writes that it is still computing it); "photos" = which photos the points are
    (see map_photos). Returns the big chart at the top left."""
    fig.clear()
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    predicted = probabilities.argmax(axis=1)
    left, right = fig.subfigures(1, 2, width_ratios=[1, 1.15])

    if curves:
        axes = left.subplot_mosaic([["confusion", "confusion"], ["noise", "rotation"]], height_ratios=[2.2, 1])
        for name, label in (("noise", tr("noise σ")), ("rotation", tr("rotation (degrees)"))):
            if curves[name] is None:
                axes[name].text(0.5, 0.5, tr("computing..."), transform=axes[name].transAxes, ha="center",
                                va="center", alpha=0.7)
                axes[name].set(xticks=[], yticks=[], xlabel=label)
            else:
                _robustness_curve(axes[name], *curves[name], label, (current or {}).get(name))
        ax = axes["confusion"]
    else:
        ax = left.subplots()
    big = ax

    if points_map:
        chosen = points_map.get("photos", np.arange(len(digits)))
        _points_map(ax, points_map, digits[chosen], predicted[chosen], len(digits))
    else:
        # Confusion matrix: rows = true digit, columns = digit said by the network
        confusion = np.zeros((10, 10), dtype=int)
        np.add.at(confusion, (digits, predicted), 1)
        ax.imshow(np.ma.masked_equal(confusion, 0), cmap="Blues", vmin=0)
        for real in range(10):
            for said in range(10):
                if confusion[real, said]:
                    color = "white" if confusion[real, said] > confusion.max() / 2 else "black"
                    ax.text(said, real, confusion[real, said], ha="center", va="center", color=color, fontsize=9)
        ax.set(xticks=range(10), yticks=range(10), xlabel=tr("digit said by the network"), ylabel=tr("true digit"),
               title=tr("Confusion matrix (the right answers are on the diagonal)"))

    # The wrong photos (at most 24)
    errors = np.flatnonzero(predicted != digits)
    shown = errors[:24]
    right.suptitle(tr("Wrong photos: {n} (the first 24)", n=len(errors)) if len(errors) > 24 else
                   tr("Wrong photos: {n}", n=len(errors)))
    error_axes = right.subplots(max(1, int(np.ceil(len(shown) / 6))), 6, squeeze=False).ravel()
    for ax in error_axes:
        ax.axis("off")
    for ax, i in zip(error_axes, shown):
        ax.imshow(photos[i], cmap="gray")
        ax.set_title(tr("true {true}, said {said}\n({confidence:.0%} sure)", true=digits[i], said=predicted[i],
                        confidence=probabilities[i].max()), fontsize=8, color="crimson")
    if not len(errors):
        error_axes[0].text(0, 0.5, tr("No mistakes!"), fontsize=14)
    return big


def _robustness_curve(ax, values, accuracies, name, current):
    ax.plot(values, accuracies, color="C1", lw=2, marker="o", markersize=3)
    if current is not None:
        ax.axvline(current, color="C0", ls="--", lw=1.2)  # where we are now
    ax.set(ylim=(0, 1.02), xlabel=name)
    ax.set_title(tr("Accuracy as this changes: {name}", name=name), fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------- map of points

def map_photos(n):
    """Which of the n test photos go on the point map: all of them, or MAP_PHOTOS chosen at random (always the
    same ones). t-SNE compares every pair of photos: with thousands of photos it would take minutes."""
    if n <= MAP_PHOTOS:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, MAP_PHOTOS, replace=False))


def project_2d(values, method="t-SNE"):
    """Squashes onto a plane points that have many coordinates (one per pixel or per neuron),
    trying to keep close the photos that the network "sees" as similar. Returns (points n x 2, axis names).
      PCA   = looks at the cloud of points from the direction in which it is most spread out (fast, faithful to
              the big distances, but the groups overlap)
      t-SNE = moves the points on the plane until each photo has close to it the same photos it had close before
              (slower, separates the groups well; the distance between one group and another does not count though)"""
    X = np.asarray(values, dtype=np.float32)
    centered = X - X.mean(axis=0)
    _, strength, directions = np.linalg.svd(centered, full_matrices=False)
    points = centered @ directions[:2].T  # the 2 directions in which the points are most spread out
    if method == "PCA":
        share = strength[:2] ** 2 / (strength ** 2).sum()
        return points, [tr("component {k}  ({share:.0%} of the variance)", k=k + 1, share=s)
                        for k, s in enumerate(share)]
    return _tsne(X, points), ["t-SNE 1", "t-SNE 2"]


def _distances2(X):
    """Squared distance between all the pairs of points (an n x n table)."""
    q = (X * X).sum(axis=1)
    return np.maximum(q[:, None] + q[None, :] - 2 * X @ X.T, 0)


def _tsne(X, start, perplexity=30, steps=350):
    n = len(X)
    # 1. How similar the photos are in the original space: for each photo a gaussian centered on it,
    #    wide enough to "see" about `perplexity` neighbours (the width is found by trial and error)
    D = _distances2(X)
    D /= np.median(D) + 1e-12
    beta, low, high = np.ones(n, np.float32), np.zeros(n, np.float32), np.full(n, np.inf, np.float32)
    for _ in range(40):
        P = np.exp(-D * beta[:, None])
        np.fill_diagonal(P, 0)
        P /= P.sum(axis=1, keepdims=True) + 1e-12
        too_wide = -(P * np.log(P + 1e-12)).sum(axis=1) > np.log(perplexity)  # it sees too many neighbours
        low, high = np.where(too_wide, beta, low), np.where(too_wide, high, beta)
        beta = np.where(np.isinf(high), beta * 2, (low + high) / 2)
    P = (P + P.T) / (2 * n)

    # 2. The points on the plane start from the PCA and move (gradient descent with momentum) until
    #    the similarities on the plane look like the original ones. At the start P is "exaggerated" x12
    #    to pull the groups well apart from each other.
    Y = start / (start[:, 0].std() + 1e-12) * 1e-2
    velocity = np.zeros_like(Y)
    for step in range(steps):
        closeness = 1 / (1 + _distances2(Y))  # similarity on the plane (Student curve: wide tails)
        np.fill_diagonal(closeness, 0)
        exaggeration, push = (12, 0.5) if step < 150 else (1, 0.8)
        W = (exaggeration * P - closeness / closeness.sum()) * closeness
        gradient = 4 * (W.sum(axis=1)[:, None] * Y - W @ Y)
        velocity = push * velocity - max(n / 48, 50) * gradient
        Y = Y + velocity
    return Y


def _points_map(ax, points_map, digits, predicted, all_photos):
    """Each test photo is a point colored with its true digit. The wrong photos have the red ring
    and a line towards the center of the group of the digit the network said: you can see "where" it gets confused."""
    ax.set_title(tr("Map of points · {name} · {method}\neach point is a test photo: close = the network sees "
                    "them as similar", name=points_map["name"], method=points_map["method"]), fontsize=10)
    if len(digits) < all_photos:
        ax.text(0.99, 0.01, tr("{shown} photos out of {total}, chosen at random", shown=len(digits), total=all_photos),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, alpha=0.7)
    points = points_map["points"]
    if points is None:
        ax.text(0.5, 0.5, tr("computing the map ({method})...", method=points_map["method"]),
                transform=ax.transAxes, ha="center", va="center", alpha=0.7)
        ax.set(xticks=[], yticks=[])
        return
    right = predicted == digits
    wrong = np.flatnonzero(~right)
    centers = [points[digits == d].mean(axis=0) if np.any(digits == d) else None for d in range(10)]
    ax.add_collection(LineCollection([(points[i], centers[predicted[i]]) for i in wrong
                                      if centers[predicted[i]] is not None],
                                     colors=ERROR_RED, linewidths=0.7, alpha=0.45))
    ax.scatter(*points[right].T, c=DIGIT_COLORS[digits[right]], s=12, alpha=0.8, linewidths=0)
    ax.scatter(*points[wrong].T, c=DIGIT_COLORS[digits[wrong]], s=40, edgecolors=ERROR_RED,
               linewidths=1.6, zorder=3)
    for digit, center in enumerate(centers):  # the name of each group, at its center
        if center is not None:
            ax.text(*center, digit, color=DIGIT_COLORS[digit], fontsize=16, fontweight="bold", ha="center",
                    va="center", zorder=4, path_effects=[patheffects.withStroke(linewidth=3, foreground="black")])
    ax.set_xlabel(points_map["axes"][0], fontsize=8)
    ax.set_ylabel(points_map["axes"][1], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=4, color="gray", label=tr("recognized photo")),
        Line2D([], [], ls="", marker="o", ms=6, color="gray", mec=ERROR_RED, mew=1.6, label=tr("wrong photo")),
        Line2D([], [], color=ERROR_RED, lw=1, label=tr("towards the digit said"))], fontsize=7, loc="best")


# ---------------------------------------------------------------- inside the network (weights and softmax)

def layer_view(fig, n_in, rows, matrix_title, with_true_digit=False):
    """Prepares the charts for the math of one layer, in colored cells ("LLM visualizer" style):
      left     the values that go in (one cell per input) and the photo
      top      the matrix: one row per input, one column per output neuron
      below    the steps of the math, lined up with the columns: rows = [(name, values, format, _), ...]
      bottom   the last step as bars (for the output: the probabilities, green border = true digit)
    Here only the structure is created: the numbers are put in by fill_layer_view(), which can be called on
    its own when only the photo or the temperature changes (it is much faster than redoing everything).
    Returns the charts in a dictionary."""
    fig.clear()
    fig.set_layout_engine("constrained")
    n_out = len(rows[0][1])
    grid = fig.add_gridspec(len(rows) + 2, 2, width_ratios=[1, 12], height_ratios=[7] + [0.62] * len(rows) + [2.4])
    axes = {"title": fig.suptitle("", fontsize=12), "input": fig.add_subplot(grid[0, 0])}
    axes["matrix"] = fig.add_subplot(grid[0, 1], sharey=axes["input"])
    axes["rows"] = [fig.add_subplot(grid[1 + r, 1], sharex=axes["matrix"]) for r in range(len(rows))]
    axes["bars"] = fig.add_subplot(grid[-1, 1], sharex=axes["matrix"])
    axes["photo"] = fig.add_subplot(grid[1:, 0])

    _cells(axes["input"], n_in, 1)
    axes["input"].set_title(tr("input"), fontsize=9)
    axes["input"].set_xticks([])
    axes["input"].set_yticks(range(0, n_in, 1 if n_in <= 32 else 8 if n_in <= 64 else 112))
    axes["input"].tick_params(labelsize=7)

    _cells(axes["matrix"], n_in, n_out)
    axes["matrix"].set_title(tr("{title}  ·  {n_in} inputs × {n_out} neurons", title=matrix_title, n_in=n_in,
                                n_out=n_out), fontsize=10)
    axes["matrix"].tick_params(labelleft=False, labelbottom=False, labeltop=True, top=True, labelsize=7)

    axes["numbers"] = []  # the numbers written in the cells of the steps (only if the columns are few)
    for ax, (name, *_) in zip(axes["rows"], rows):
        _cells(ax, 1, n_out)
        axes["numbers"].append([ax.text(c, 0, "", ha="center", va="center", fontsize=8) for c in range(n_out)]
                               if n_out <= 16 else [])
        ax.set_yticks([0], [name], fontsize=8)
        ax.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False, labelright=True)

    ax = axes["bars"]
    axes["rectangles"] = ax.bar(range(n_out), np.zeros(n_out), width=0.7)
    if with_true_digit:
        axes["true_digit"] = ax.bar(0, 0, width=0.7, fill=False, edgecolor="#5fd39a", lw=2, label=tr("true digit"))[0]
        ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(rows[-1][0], fontsize=9)
    ax.set_xticks(range(0, n_out, 1 if n_out <= 32 else 8))
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.3)

    axes["photo"].imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=255)
    axes["photo"].set_title(tr("the photo"), fontsize=9)
    axes["photo"].axis("off")
    return axes


def fill_layer_view(axes, title, photo, inputs, matrix, rows, true_digit=None):
    """Puts the numbers into the charts prepared by layer_view()."""
    axes["title"].set_text(title)
    axes["photo"].images[0].set_data(photo)
    _color(axes["input"], inputs[:, None])
    _color(axes["matrix"], matrix)
    for ax, numbers, (_, values, fmt, _) in zip(axes["rows"], axes["numbers"], rows):
        maximum = _color(ax, values[None, :])
        for text, v in zip(numbers, values):  # dark text on the light cells, light text on the dark ones
            text.set_text(fmt.format(v))
            text.set_color("#14171c" if abs(v) > 0.6 * maximum else "#e6e6e6")

    # The last step as bars: the highest one in orange
    values = rows[-1][1]
    for k, (rectangle, v) in enumerate(zip(axes["rectangles"], values)):
        rectangle.set_height(v)
        rectangle.set_color("#ffb347" if k == values.argmax() else "#4da3ff")
    if true_digit is not None:
        axes["true_digit"].set_x(true_digit - 0.35)
        axes["true_digit"].set_height(values[true_digit])
    margin = 0.08 * max(np.abs(values).max(), 1e-6)
    axes["bars"].set_ylim(min(values.min(), 0) - margin, max(values.max(), 0) + margin)


def _cells(ax, rows, columns):
    """An empty table of colored cells, separated by thin lines."""
    ax.imshow(np.zeros((rows, columns)), cmap=VALUE_CMAP, aspect="auto", interpolation="nearest")
    background = rcParams["figure.facecolor"]
    if columns <= 64:  # the separating lines only if the cells are not too dense
        ax.vlines(np.arange(1, columns) - 0.5, -0.5, rows - 0.5, color=background, lw=1.5)
    if rows <= 64:
        ax.hlines(np.arange(1, rows) - 0.5, -0.5, columns - 0.5, color=background, lw=1.5)


def _color(ax, values):
    """Colors the cells: blue = negative, dark = zero, orange = positive (scale: the biggest value)."""
    maximum = np.abs(values).max() + 1e-9
    image = ax.images[0]
    image.set_data(values)
    image.set_clim(-maximum, maximum)
    return maximum


def point_label(ax, xy, photo, text, color):
    """The box that appears when the mouse goes over a point of the map: the photo and what the network thinks of it.
    It is "animated": it is not part of the normal drawing, it is drawn separately on top (much faster)."""
    content = VPacker(children=[OffsetImage(np.repeat(photo[..., None], 3, axis=2).clip(0, 1), zoom=2.5),
                                TextArea(text, textprops=dict(color=color, fontsize=8))],
                      align="center", pad=0, sep=4)
    x, y = ax.transAxes.inverted().transform(ax.transData.transform(xy))  # where the point is (0-1)
    label = AnnotationBbox(content, xy, xybox=(-75 if x > 0.6 else 75, -65 if y > 0.6 else 65),
                           boxcoords="offset points", pad=0.5, animated=True,
                           bboxprops=dict(facecolor=rcParams["axes.facecolor"], edgecolor=color),
                           arrowprops=dict(arrowstyle="-", color=color))
    label.set_in_layout(False)
    ax.add_artist(label)
    return label

"""
Training: makes the network learn one epoch at a time, and puts it to the test.

An "epoch" is one full round over all the training photos. At every epoch:
  1. the photos are slightly rotated/shifted and, if you want, dirtied by noise;
  2. the network looks at them in small groups (mini-batches) and after each group it corrects the weights;
  3. we measure how wrong it is (loss) and how many it gets right (accuracy).

10% of the photos ("validation") is never used to learn: it is used to check
that the network is understanding the digits and not learning the photos by heart.
"""
import time

import numpy as np

from nn_digits.neural_net.data import add_noise, alter, augment, load_photos, normalize, one_hot
from nn_digits.neural_net.network import NeuralNetwork


class Trainer:
    def __init__(self, hidden=(64, 32), activation="relu", init_scale=1.0, seed=0, photos_and_digits=None):
        """hidden = neurons of the hidden layers; the other parameters go to the NeuralNetwork.
        photos_and_digits = photos already loaded (if missing, it reads them from data/photos/train)."""
        self.rng = np.random.default_rng(seed)
        photos, digits = photos_and_digits if photos_and_digits is not None else load_photos("train")

        # Split the photos at random: 90% to learn, 10% to check (validation)
        order = self.rng.permutation(len(digits))
        n_val = len(digits) // 10
        self.train_photos, self.train_digits = photos[order[n_val:]], digits[order[n_val:]]
        self.val_digits = digits[order[:n_val]]
        self.X_train, self.Y_train = normalize(self.train_photos), one_hot(self.train_digits)
        self.X_val, self.Y_val = normalize(photos[order[:n_val]]), one_hot(self.val_digits)

        self.net = NeuralNetwork((784, *hidden, 10), activation, init_scale, seed)
        self.init_scale = init_scale
        self.initial_weights = [W.copy() for W in self.net.weights]  # to compare the gaussians
        # All the measurements, epoch after epoch (the charts need them)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
                        "batch_x": [], "batch_loss": [], "gradients": [], "inactive": [], "seconds": []}

    @property
    def epoch(self):
        """How many epochs have been done so far."""
        return len(self.history["val_acc"])

    @property
    def exploded(self):
        """True if the weights have become infinite: it happens with a learning rate that is too high."""
        return self.epoch > 0 and not np.isfinite(self.history["train_loss"][-1])

    def run_epoch(self, lr, noise=0.0, batch=32, momentum=0.9, l2=1e-4, dropout=0.0, rotation=12, shift=2):
        """One round over all the training photos, with these parameters. Returns the end-of-epoch measurements."""
        start = time.perf_counter()
        history = self.history
        with np.errstate(all="ignore"):  # if the network "explodes", NumPy does not fill the screen with warnings
            # 1. Slightly different photos at every epoch: rotated/shifted + noise
            X = add_noise(normalize(augment(self.train_photos, self.rng, rotation, shift)), noise, self.rng)
            # 2. In small groups, in random order: after each group the network corrects the weights
            order = self.rng.permutation(len(X))
            n_batches = int(np.ceil(len(X) / batch))
            gradients = np.zeros(len(self.net.weights))
            for k in range(n_batches):
                group = order[k * batch:(k + 1) * batch]
                loss = self.net.train_step(X[group], self.Y_train[group], lr, momentum, l2, dropout, self.rng)
                history["batch_loss"].append(loss)
                history["batch_x"].append(self.epoch + (k + 1) / n_batches)  # position on the chart, in epochs
                gradients += self.net.gradient_norms
            # 3. Measurements on the original photos (train) and on the ones never used to learn (validation)
            val_outputs = self.net.forward(self.X_val)
            P_train, P_val = self.net.predict(self.X_train), val_outputs[-1]
            history["train_loss"].append(self.net.loss(P_train, self.Y_train))
            history["val_loss"].append(self.net.loss(P_val, self.Y_val))
            # "Inactive" neurons: they always give the same output whatever photo they see, so they are useless
            history["inactive"].append([float(np.mean(a.std(axis=0) < 1e-3)) for a in val_outputs[1:-1]])
        history["gradients"].append((gradients / n_batches).tolist())
        history["train_acc"].append(float(np.mean(P_train.argmax(1) == self.train_digits)))
        history["val_acc"].append(float(np.mean(P_val.argmax(1) == self.val_digits)))
        history["lr"].append(lr)
        history["seconds"].append(time.perf_counter() - start)
        return {name: history[name][-1] for name in ("train_loss", "val_loss", "train_acc", "val_acc")}

    def snapshot(self):
        """A copy of the current state, to draw while the training goes on."""
        return {"history": {name: list(values) for name, values in self.history.items()},
                "weights": [W.copy() for W in self.net.weights],
                "initial_weights": self.initial_weights}


def test_on(net, photos, noise=0.0, rotation=0.0, thickness=0):
    """Makes the network answer on the photos, spoiled on purpose if requested (noise, rotation, thickness).
    Returns (photos used with values 0-1, probabilities given by the network)."""
    X = add_noise(normalize(alter(photos, rotation, thickness)), noise, np.random.default_rng(0))
    return X, net.predict(X)


test_on.__test__ = False  # the name starts with "test": this tells pytest it is not a test


def robustness(net, photos, digits):
    """How well the network resists spoiled photos: accuracy as noise and rotation grow.
    Returns {"noise": (values tried, accuracies), "rotation": (values tried, accuracies)}."""
    def accuracy(**alteration):
        _, probabilities = test_on(net, photos, **alteration)
        return float(np.mean(probabilities.argmax(1) == digits))

    sigmas = np.linspace(0, 0.6, 13)
    degrees = np.linspace(-45, 45, 13)
    return {"noise": (sigmas, [accuracy(noise=s) for s in sigmas]),
            "rotation": (degrees, [accuracy(rotation=d) for d in degrees])}

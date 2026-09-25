"""
The hints: simple rules that look at the state of the program and notice the most common mistakes.

They are rules and not a language model on purpose: a rule is always right about the numbers it reads, it is
instant and it can be tested. Every hint says what it sees and, when possible, offers a fix that can be applied
with one click. A fix is a list of steps that the interface knows how to do:
    ("set", control, value)   moves a control of tab 2, for example ("set", "lr", 0.05)
    ("tab", tab)              opens a tab
    ("new network",)          presses "New network" in tab 2
    ("reset lab",)            removes the changes made in the lab of tab 4

The state is a dictionary made by the interface (see gui/app_state.py): the rules only read it.
"""
from collections import namedtuple

import numpy as np

from i18n import tr

# key: to know whether a hint was already shown; topic: the concept of knowledge.py that explains it
Hint = namedtuple("Hint", "key text fix fix_label topic")


def hints(state):
    """All the hints that apply to this state, the most important first."""
    found = []
    for rule in (_missing_photos, _missing_model, _exploded, _not_learning, _slow, _jumping, _overfitting,
                 _inactive, _initial_weights, _strong_l2, _strong_dropout, _heavy_noise, _big_rotation,
                 _momentum, _tiny_batch, _few_photos, _test_harder_than_training, _lab_damage):
        hint = rule(state)
        if hint:
            found.append(hint)
    return found


# ------------------------------------------------------------------ reading the state

def _training(state):
    return state.get("training") or {}


def _history(state):
    """The measurements of the epochs done so far (None if the network has not trained yet)."""
    history = _training(state).get("history")
    return history if history and history.get("val_acc") else None


def _params(state):
    return _training(state).get("params") or {}


def _epochs(state):
    history = _history(state)
    return len(history["val_acc"]) if history else 0


def _exploded_now(state):
    history = _history(state)
    return bool(history) and not np.isfinite(history["train_loss"][-1])


def _smaller_lr(lr):
    """A learning rate 3 times smaller, as the slider of tab 2 can set it (steps of 0.05 on the exponent of 10)."""
    return float(f"{10 ** (round(np.log10(lr / 3) / 0.05) * 0.05):.2g}")


# ------------------------------------------------------------------ the rules

def _missing_photos(state):
    if state.get("tab") in ("training", "evaluation", "draw", "inside") and not state.get("photos", {}).get("train"):
        return Hint("photos", tr("There are no photos yet: the network has nothing to learn from. Download them in "
                                 "tab 1 · Data."), (("tab", "data"),), tr("Go to tab 1"), "mnist")


def _missing_model(state):
    if (state.get("tab") in ("evaluation", "draw", "inside") and state.get("photos", {}).get("train")
            and not state.get("model")):
        return Hint("model", tr("There is no trained model yet: first train the network in tab 2 · Training."),
                    (("tab", "training"),), tr("Go to tab 2"), "network")


def _exploded(state):
    if _exploded_now(state):
        lr = _params(state).get("lr", 0.05)
        new = min(0.05, _smaller_lr(lr))
        return Hint("exploded", tr("The network exploded: the weights have become infinite because the learning rate "
                                   "({lr:g}) is too high.", lr=lr),
                    (("set", "lr", new), ("new network",)),
                    tr("Learning rate {lr:g} and new network", lr=new), "exploded")


def _not_learning(state):
    """After a few epochs it still guesses at random (about 10%)."""
    history, params = _history(state), _params(state)
    if _epochs(state) < 3 or _exploded_now(state) or history["val_acc"][-1] >= 0.25:
        return None
    lr = params.get("lr", 0.05)
    network = _training(state).get("network") or {}
    if lr >= 0.5:
        return Hint("not learning", tr("After {n} epochs the network still guesses at random: the learning rate "
                                       "{lr:g} is too high.", n=_epochs(state), lr=lr),
                    (("set", "lr", 0.05), ("new network",)), tr("Learning rate 0.05 and new network"), "learning rate")
    if network.get("init_scale", 1) <= 0.2:
        return Hint("not learning", tr("After {n} epochs the network still guesses at random: the initial weights "
                                       "are too narrow (x{scale:g}) and the signal fades out.",
                                       n=_epochs(state), scale=network["init_scale"]),
                    (("set", "init_scale", 1.0), ("new network",)), tr("Width x1 and new network"), "initialization")
    if lr <= 0.001:
        return Hint("not learning", tr("After {n} epochs the network still guesses at random: the learning rate "
                                       "{lr:g} is too low.", n=_epochs(state), lr=lr),
                    (("set", "lr", 0.05),), tr("Learning rate 0.05"), "learning rate")
    return Hint("not learning", tr("After {n} epochs the network still guesses at random. Look at the corrections "
                                   "per layer and the inactive neurons: usually the learning rate is too high.",
                                   n=_epochs(state)),
                (("set", "lr", _smaller_lr(lr)),), tr("Learning rate {lr:g}", lr=_smaller_lr(lr)), "learning rate")


def _slow(state):
    """It learns, but very slowly, because the learning rate is tiny."""
    history, lr = _history(state), _params(state).get("lr", 0.05)
    if _epochs(state) >= 5 and lr <= 0.003 and 0.25 <= history["val_acc"][-1] < 0.85:
        return Hint("slow", tr("It learns very slowly: with learning rate {lr:g} every correction is tiny.", lr=lr),
                    (("set", "lr", 0.05),), tr("Learning rate 0.05"), "learning rate")


def _jumping(state):
    """The validation loss goes up and down: the steps are too big."""
    history, lr = _history(state), _params(state).get("lr", 0.05)
    if _epochs(state) < 4 or lr < 0.3 or _exploded_now(state):
        return None
    ups = np.sum(np.diff(history["val_loss"][-5:]) > 0)
    if ups >= 2:
        return Hint("jumping", tr("The validation loss jumps up and down: the learning rate {lr:g} is too high, the "
                                  "corrections overshoot.", lr=lr),
                    (("set", "lr", _smaller_lr(lr)),), tr("Learning rate {lr:g}", lr=_smaller_lr(lr)), "learning rate")


def _overfitting(state):
    """The validation gets worse while the train keeps getting better: it is learning by heart."""
    history, params = _history(state), _params(state)
    if _epochs(state) < 6 or _exploded_now(state) or params.get("lr", 0.05) >= 0.3:
        return None
    val_loss, train_loss = history["val_loss"], history["train_loss"]
    best = int(np.argmin(val_loss))
    worse_for = len(val_loss) - 1 - best
    gap = history["train_acc"][-1] - history["val_acc"][-1]
    if not ((worse_for >= 4 and val_loss[-1] > 1.05 * val_loss[best] and train_loss[-1] < train_loss[best])
            or gap >= 0.08):
        return None
    if params.get("dropout", 0) < 0.1:
        fix, label = (("set", "dropout", 0.2),), tr("Dropout 20%")
    elif params.get("l2", 0) < 1e-3:
        fix, label = (("set", "l2", 1e-3),), tr("L2 0.001")
    elif params.get("rotation", 0) < 10:
        fix, label = (("set", "rotation", 12), ("set", "shift", 2)), tr("Rotation ±12° and shift ±2 px")
    else:
        fix, label = (("tab", "data"),), tr("More photos: tab 1")
    if worse_for >= 4:
        text = tr("Overfitting: the validation loss has been going up for {n} epochs (the best was {best:.3f} at epoch "
                  "{epoch}) while the train loss keeps going down. The network is learning the photos by heart.",
                  n=worse_for, best=val_loss[best], epoch=best + 1)
    else:
        text = tr("Overfitting: {train:.1%} right on the training photos but only {val:.1%} on the validation ones. "
                  "The network is learning the photos by heart.",
                  train=history["train_acc"][-1], val=history["val_acc"][-1])
    return Hint("overfitting", text, fix, label, "overfitting")


def _inactive(state):
    history = _history(state)
    if not history or _exploded_now(state) or not history["inactive"] or not history["inactive"][-1]:
        return None
    shares = history["inactive"][-1]
    layer = int(np.argmax(shares))
    if shares[layer] >= 0.3:
        lr = _params(state).get("lr", 0.05)
        return Hint("inactive", tr("{share:.0%} of the neurons of layer {layer} are inactive: they give the same "
                                   "output for every photo, so they are useless. Usually the learning rate is too "
                                   "high.", share=shares[layer], layer=layer + 1),
                    (("set", "lr", _smaller_lr(lr)),), tr("Learning rate {lr:g}", lr=_smaller_lr(lr)), "inactive")


def _initial_weights(state):
    """An extreme width of the initial weights: fine for an experiment, but better to know it."""
    scale = (_training(state).get("chosen") or {}).get("init_scale", 1.0)
    if state.get("tab") == "training" and (scale <= 0.2 or scale >= 3):
        return Hint("initial weights", tr("The initial weights are x{scale:g}: with a width that is far from x1 the "
                                          "signal fades out (x0.1) or explodes (x5) layer after layer. Fine for an "
                                          "experiment: look at the gaussians and the corrections per layer.",
                                          scale=scale),
                    (("set", "init_scale", 1.0),), tr("Width x1"), "initialization")


def _strong_l2(state):
    history, l2 = _history(state), _params(state).get("l2", 0)
    if l2 >= 0.005 and _epochs(state) >= 3 and history["train_acc"][-1] < 0.9:
        return Hint("l2", tr("L2 {l2:g} is very strong: it pushes the weights towards zero faster than the network "
                             "can learn them.", l2=l2), (("set", "l2", 1e-4),), tr("L2 0.0001"), "l2")


def _strong_dropout(state):
    history, dropout = _history(state), _params(state).get("dropout", 0)
    if dropout >= 0.6 and _epochs(state) >= 2 and history["train_acc"][-1] < 0.9:
        return Hint("dropout", tr("Dropout {dropout:.0%} switches off most of the neurons at every step: the network "
                                  "struggles to learn.", dropout=dropout),
                    (("set", "dropout", 0.2),), tr("Dropout 20%"), "dropout")


def _heavy_noise(state):
    noise = _params(state).get("noise", 0)
    if state.get("tab") == "training" and noise >= 0.4:
        return Hint("noise", tr("With noise σ {noise:.2f} the training photos are hard to recognize even for you (look "
                                "at the preview): the accuracy suffers.", noise=noise),
                    (("set", "noise", 0.1),), tr("Noise σ 0.10"), "noise")


def _big_rotation(state):
    rotation = _params(state).get("rotation", 0)
    if state.get("tab") == "training" and rotation >= 35:
        return Hint("rotation", tr("Rotations up to ±{rotation:.0f}° turn a 6 into a 9 and the other way round: the "
                                   "network gets confused.", rotation=rotation),
                    (("set", "rotation", 12),), tr("Rotation ±12°"), "augmentation")


def _momentum(state):
    params = _params(state)
    momentum, lr = params.get("momentum", 0.9), params.get("lr", 0.05)
    if state.get("tab") == "training" and momentum >= 0.97 and lr >= 0.1:
        return Hint("momentum", tr("Momentum {momentum:.2f} together with learning rate {lr:g}: the corrections pile "
                                   "up and overshoot.", momentum=momentum, lr=lr),
                    (("set", "momentum", 0.9),), tr("Momentum 0.90"), "momentum")


def _tiny_batch(state):
    batch = _params(state).get("batch", 32)
    if state.get("tab") == "training" and batch <= 4 and state.get("photos", {}).get("train", 0) >= 500:
        return Hint("batch", tr("Mini-batches of {batch} photos: many noisy corrections and very slow epochs.",
                                batch=batch), (("set", "batch", 32),), tr("32 photos per mini-batch"), "batch")


def _few_photos(state):
    n = state.get("photos", {}).get("train", 0)
    if state.get("tab") == "training" and 0 < n < 300:
        return Hint("few photos", tr("Only {n} training photos: the network will learn them by heart. Download more in "
                                     "tab 1 · Data.", n=n), (("tab", "data"),), tr("Go to tab 1"), "overfitting")


def _test_harder_than_training(state):
    """In tab 3 the test photos are spoiled more than the network ever saw while learning."""
    evaluation, params = state.get("evaluation") or {}, _params(state)
    if state.get("tab") != "evaluation" or not state.get("model") or not evaluation:
        return None
    rotation, trained_rotation = abs(evaluation.get("rotation", 0)), params.get("rotation", 12)
    if rotation >= 20 and trained_rotation < rotation:
        new = min(45, int(rotation))
        return Hint("test rotation", tr("The test photos are rotated by {rotation:.0f}°, but tab 2 is set to rotations "
                                        "up to ±{trained:.0f}°: train the network with more rotation and it will hold "
                                        "up better.", rotation=rotation, trained=trained_rotation),
                    (("set", "rotation", new), ("tab", "training")),
                    tr("Rotation ±{rotation:.0f}° in tab 2", rotation=new), "augmentation")
    if evaluation.get("noise", 0) >= 0.3 and params.get("noise", 0) < 0.1:
        return Hint("test noise", tr("The test photos have noise σ {noise:.2f}, but tab 2 is set to train almost "
                                     "without noise: train with a little noise and it will hold up better.",
                                     noise=evaluation["noise"]),
                    (("set", "noise", 0.15), ("tab", "training")), tr("Noise σ 0.15 in tab 2"), "noise")


def _lab_damage(state):
    lab = state.get("lab") or {}
    before, after = lab.get("accuracy_before"), lab.get("accuracy_after")
    if state.get("tab") == "draw" and before is not None and after is not None and before - after >= 0.2:
        return Hint("lab", tr("Your changes cost {drop:.0%} of test accuracy ({before:.1%} → {after:.1%}). It is only "
                              "a copy: the saved model is still intact.",
                              drop=before - after, before=before, after=after),
                    (("reset lab",),), tr("Reset everything"), "pruning")

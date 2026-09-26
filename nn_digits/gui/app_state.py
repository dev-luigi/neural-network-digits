"""
The state of the program in a dictionary: which tab is open, the photos, the model, the training, the
evaluation, the drawing, the lab. The assistant reads it (assistant/brain.py and assistant/rules.py), so they
never touch the window: they can be tested with a dictionary written by hand.
"""
import numpy as np

from nn_digits.assistant.knowledge import TABS
from nn_digits.gui import base
from nn_digits.i18n import tr
from nn_digits.neural_net.storage import MODEL_FILE


def app_state(window):
    return {
        "tab": TABS[window.tabs.index("current")],
        "photos": {split: window.photo_count(split) for split in ("train", "test")},
        "model": MODEL_FILE.exists(),
        "training": _training(window.training_tab),
        "evaluation": _evaluation(window.evaluation_tab),
        "lab": _lab(window.draw_tab.lab),
        "drawing": _drawing(window.draw_tab.lab),
        "inside": _inside(window.inside_tab),
    }


def _training(tab):
    trainer, snapshot = tab.trainer, tab.state
    return {
        "running": tab.in_progress,
        "paused": tab.in_progress and not tab.go.is_set(),
        "params": dict(tab.params),
        "cosine": tab.cosine,
        "chosen": tab.chosen_architecture(),
        "network": trainer and {"layers": list(trainer.net.layers), "activation": trainer.net.activation,
                                "init_scale": trainer.init_scale},
        # the copy sent by the training thread: its lists always have the same length
        "history": snapshot["history"] if snapshot else None,
    }


def _evaluation(tab):
    evaluation = {"noise": tab.noise.get(), "rotation": tab.rotation.get(), "thickness": int(tab.thickness.get()),
                  "threshold": tab.threshold.get(), "accuracy": None}
    if tab.net is not None and tab.probabilities is not None:
        right = tab.probabilities.argmax(axis=1) == tab.digits
        evaluation.update(accuracy=float(right.mean()), total=len(tab.digits))
    return evaluation


def _lab(lab):
    if lab is None or lab.accuracies is None:
        return None
    changed = bool(lab.changes) or any(slider.get() != 0 for slider in (lab.temperature, lab.weight_noise,
                                                                           lab.pruning))
    return {"changed": changed, "accuracy_before": lab.accuracies[0], "accuracy_after": lab.accuracies[1],
            "pruning": lab.pruning.get(), "temperature": base.power(lab.temperature.get())}


def _drawing(lab):
    activations = lab and lab.board.activations
    if activations is None:
        return None
    probabilities = np.nan_to_num(activations[-1])
    return {"answer": int(probabilities.argmax()), "confidence": float(probabilities.max())}


def _inside(tab):
    if tab.net is None or tab.math is None:
        return None
    i, k = int(tab.photo_slider.get()), tab.math["k"]
    return {"photo": i, "layer": tr("output") if tab.math["output"] else tr("layer {k}", k=k + 1),
            "true": int(tab.digits[i]), "answer": int(tab.predicted[i])}

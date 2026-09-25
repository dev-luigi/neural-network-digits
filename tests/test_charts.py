"""The charts are drawn without errors; the 2D projections keep similar points together."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from neural_net import charts
from neural_net.network import softmax_steps
from neural_net.training import Trainer

RNG = np.random.default_rng(0)


def draw(fig):
    FigureCanvasAgg(fig).draw()


@pytest.mark.parametrize("method", ["PCA", "t-SNE"])
def test_project_2d_keeps_the_groups_together(method):
    centers = RNG.normal(0, 10, (3, 20))
    group = np.repeat(np.arange(3), 30)
    points, axes = charts.project_2d(centers[group] + RNG.normal(0, 1, (90, 20)), method)
    assert points.shape == (90, 2) and np.isfinite(points).all() and len(axes) == 2
    distances = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert np.mean(group[distances.argmin(axis=1)] == group) > 0.95  # the nearest neighbour is from the same group


def test_training_and_chart():
    photos, digits = RNG.integers(0, 256, (100, 28, 28), dtype=np.uint8), np.arange(100) % 10
    trainer = Trainer(hidden=(16, 8), photos_and_digits=(photos, digits))
    trainer.run_epoch(lr=0.05, noise=0.1)
    assert trainer.epoch == 1 and not trainer.exploded
    fig = Figure(layout="constrained")
    charts.training(fig, trainer.snapshot(), noise=0.1)
    draw(fig)


def test_evaluation_with_matrix_and_with_map():
    photos, digits = RNG.random((40, 28, 28)), np.arange(40) % 10
    probabilities = RNG.dirichlet(np.ones(10), 40)
    for points_map in (None,
                       {"points": RNG.normal(size=(40, 2)), "axes": ["a", "b"], "name": "layer 2", "method": "PCA"},
                       {"points": None, "axes": None, "name": "layer 2", "method": "t-SNE"}):
        fig = Figure(layout="constrained")
        ax = charts.evaluation(fig, photos, digits, probabilities, points_map=points_map)
        draw(fig)
        if points_map and points_map["points"] is not None:
            charts.point_label(ax, points_map["points"][0], photos[0], "true 0", "red")


def test_layer_view():
    x, W, b = RNG.random(32), RNG.normal(size=(32, 10)), RNG.normal(size=10)
    z = x @ W + b
    shifted, e, p = softmax_steps(z, 2.0)
    rows = [("bias", b, "{:+.2f}", ""), ("z", z, "{:+.2f}", ""), ("z - max", shifted, "{:+.2f}", ""),
            ("e", e, "{:.2f}", ""), ("p", p, "{:.0%}", "")]
    fig = Figure(layout="constrained")
    axes = charts.layer_view(fig, 32, rows, "weights", with_true_digit=True)
    charts.fill_layer_view(axes, "title", RNG.random((28, 28)), x, W, rows, true_digit=3)
    draw(fig)
    assert axes["numbers"][-1][p.argmax()].get_text() == f"{p.max():.0%}"

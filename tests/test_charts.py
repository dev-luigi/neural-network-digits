"""The charts are drawn without errors; the 2D projections keep similar points together."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nn_digits.neural_net import charts
from nn_digits.neural_net.network import softmax_steps
from nn_digits.neural_net.training import Trainer

RNG = np.random.default_rng(0)


def draw(fig):
    FigureCanvasAgg(fig).draw()


def gids(fig):
    """The names of the charts of the figure (the ones without a name are left out)."""
    return [ax.get_gid() for ax in fig.axes if ax.get_gid()]


def crossed_out(where):
    """True if the chart (or the whole figure) is crossed out: a thin X from corner to corner, its empty state."""
    transform = where.transAxes if hasattr(where, "transAxes") else where.transFigure
    lines = [line.get_xydata().tolist() for line in where.findobj(Line2D) if line.get_transform() is transform]
    return sorted(lines) == [[[0, 0], [1, 1]], [[0, 1], [1, 0]]]


def test_crossed_out_does_not_move_the_chart():
    fig = Figure()
    ax = fig.subplots()
    ax.plot([2, 3], [5, 7])
    limits = ax.get_xlim(), ax.get_ylim()
    charts.crossed_out(ax, "nothing here")
    assert crossed_out(ax) and (ax.get_xlim(), ax.get_ylim()) == limits  # the X does not change the limits
    charts.crossed_out(fig, "nothing at all")  # on the whole figure
    draw(fig)
    assert crossed_out(fig) and [text.get_text() for text in fig.texts] == ["nothing at all"]


@pytest.mark.parametrize("method", ["PCA", "t-SNE"])
def test_project_2d_keeps_the_groups_together(method):
    centers = RNG.normal(0, 10, (3, 20))
    group = np.repeat(np.arange(3), 30)
    points, axes = charts.project_2d(centers[group] + RNG.normal(0, 1, (90, 20)), method)
    assert points.shape == (90, 2) and np.isfinite(points).all() and len(axes) == 2
    distances = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert np.mean(group[distances.argmin(axis=1)] == group) > 0.95  # the nearest neighbour is from the same group


def test_exploration():
    photos, digits = RNG.integers(0, 256, (100, 28, 28), dtype=np.uint8), np.arange(100) % 10
    fig = Figure(layout="constrained")
    charts.exploration(fig, photos, digits, photos[:30], digits[:30], title="dataset")
    draw(fig)
    assert gids(fig) == ["samples", "per digit", "average digit", "pixel values"]  # Pick explains each chart


def test_training_and_chart():
    photos, digits = RNG.integers(0, 256, (100, 28, 28), dtype=np.uint8), np.arange(100) % 10
    trainer = Trainer(hidden=(16, 8), photos_and_digits=(photos, digits))
    fig = Figure(layout="constrained")
    charts.training(fig, trainer.snapshot(), noise=0.1)  # no epoch yet: the three curves are crossed out
    draw(fig)
    assert [ax.get_gid() for ax in fig.axes if crossed_out(ax)] == ["loss", "accuracy", "corrections"]

    trainer.run_epoch(lr=0.05, noise=0.1)
    assert trainer.epoch == 1 and not trainer.exploded
    charts.training(fig, trainer.snapshot(), noise=0.1)
    draw(fig)
    assert gids(fig) == ["loss", "accuracy", "corrections", "first layer", "weights", "weights", "weights", "noise"]
    assert not any(crossed_out(ax) for ax in fig.axes)

    charts.training(fig, trainer.snapshot(), noise=0)  # no noise: that chart is crossed out, corner to corner
    draw(fig)
    assert [ax.get_gid() for ax in fig.axes if crossed_out(ax)] == ["noise"]


def test_evaluation_with_matrix_and_with_map():
    photos, digits = RNG.random((40, 28, 28)), np.arange(40) % 10
    probabilities = RNG.dirichlet(np.ones(10), 40)
    curves = {"noise": (np.linspace(0, 0.6, 13), RNG.random(13)), "rotation": None}  # rotation: still computing
    for points_map in (None,
                       {"points": RNG.normal(size=(40, 2)), "axes": ["a", "b"], "name": "layer 2", "method": "PCA"},
                       {"points": RNG.normal(size=(10, 2)), "axes": ["a", "b"], "name": "layer 2", "method": "PCA",
                        "photos": np.arange(0, 40, 4)},  # only some of the photos on the map
                       {"points": None, "axes": None, "name": "layer 2", "method": "t-SNE"}):
        fig = Figure(layout="constrained")
        ax = charts.evaluation(fig, photos, digits, probabilities, curves, points_map=points_map)
        draw(fig)
        assert ax.get_gid() == ("points map" if points_map else "confusion")
        assert {"noise curve", "rotation curve", "wrong photo"} <= set(gids(fig))
        not_ready = {"rotation curve", "points map"} if points_map and points_map["points"] is None else {"rotation curve"}
        assert {ax.get_gid() for ax in fig.axes if crossed_out(ax)} == not_ready  # still computing: crossed out
        if points_map and points_map["points"] is not None:
            charts.point_label(ax, points_map["points"][0], photos[0], "true 0", "red")

    fig = Figure(layout="constrained")
    charts.evaluation(fig, photos, digits, np.eye(10)[digits], curves)  # all right: no wrong photo to show
    draw(fig)
    assert "No mistakes!" in [text.get_text() for ax in fig.axes if crossed_out(ax) for text in ax.texts]


def test_map_photos():
    assert charts.map_photos(200).tolist() == list(range(200))
    chosen = charts.map_photos(10_000)  # t-SNE on 10,000 photos would take minutes
    assert len(set(chosen)) == charts.MAP_PHOTOS and chosen.max() < 10_000
    assert np.array_equal(chosen, charts.map_photos(10_000))  # always the same ones


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
    names = ["input", "matrix", "bars", "photo"]
    assert [axes[name].get_gid() for name in names] == names

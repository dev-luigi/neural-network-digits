"""The network: softmax, backpropagation, learning, saving."""
import copy

import numpy as np
import pytest

from neural_net.network import ACTIVATIONS, NeuralNetwork, softmax, softmax_steps


def test_softmax_gives_probabilities_even_with_huge_numbers():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(axis=1), 1)


def test_softmax_steps_same_as_softmax():
    z = np.random.default_rng(0).normal(size=10)
    shifted, e, p = softmax_steps(z)
    assert np.allclose(p, softmax(z))
    assert shifted.max() == 0 and e.max() == 1


def test_temperature_changes_the_confidence_not_the_answer():
    z = np.array([1.0, 2.0, 3.0])
    cold, normal, hot = (softmax_steps(z, t)[2] for t in (0.1, 1.0, 10.0))
    assert cold.max() > normal.max() > hot.max()
    assert cold.argmax() == normal.argmax() == hot.argmax()


def test_backpropagation_same_as_numerical_gradient():
    """The corrections computed by backpropagation must match the "brute force" ones:
    I move a weight by a tiny amount and look at how much the loss changes."""
    rng = np.random.default_rng(1)
    net = NeuralNetwork((5, 4, 3), "tanh", seed=1)
    for values in (net.weights, net.biases, net.weight_velocity, net.bias_velocity):  # float64: more precise math
        values[:] = [v.astype(np.float64) for v in values]
    X, Y = rng.normal(size=(8, 5)), np.eye(3)[rng.integers(0, 3, 8)]
    before = copy.deepcopy(net)
    lr = 1e-3
    net.train_step(X, Y, lr, momentum=0, l2=0)
    for layer in range(2):
        gradient = (before.weights[layer] - net.weights[layer]) / lr  # plain SGD: W_new = W - lr * gradient
        for i, j in [(0, 0), (1, 2), (3, 1)]:
            trial = copy.deepcopy(before)
            trial.weights[layer][i, j] += 1e-6
            plus = trial.loss(trial.predict(X), Y)
            trial.weights[layer][i, j] -= 2e-6
            minus = trial.loss(trial.predict(X), Y)
            assert gradient[i, j] == pytest.approx((plus - minus) / 2e-6, rel=1e-4, abs=1e-7)


@pytest.mark.parametrize("activation", list(ACTIVATIONS))
def test_learns_an_easy_problem(activation):
    rng = np.random.default_rng(0)
    X = rng.random((200, 784), dtype=np.float32)
    digits = (X[:, :392].sum(axis=1) > X[:, 392:].sum(axis=1)).astype(int)  # more light on the left or on the right?
    Y = np.eye(10, dtype=np.float32)[digits]
    net = NeuralNetwork((784, 16, 10), activation, seed=0)
    before = net.loss(net.predict(X), Y)
    for _ in range(150):
        net.train_step(X, Y, lr=0.1)
    assert net.loss(net.predict(X), Y) < before / 2


def test_save_and_load(tmp_path):
    net = NeuralNetwork((784, 8, 10), "tanh", seed=3)
    X = np.random.default_rng(0).random((4, 784), dtype=np.float32)
    net.save(tmp_path / "model.npz")
    loaded = NeuralNetwork.load(tmp_path / "model.npz")
    assert loaded.layers == net.layers and loaded.activation == "tanh"
    assert np.allclose(loaded.predict(X), net.predict(X))

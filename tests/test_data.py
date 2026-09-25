"""Preparing the photos: normalization, noise, alterations, drawings made with the mouse."""
import numpy as np
from PIL import Image, ImageDraw

from neural_net.data import add_noise, alter, augment, normalize, one_hot, prepare_drawing

PHOTOS = np.random.default_rng(0).integers(0, 256, (5, 28, 28), dtype=np.uint8)


def test_normalize():
    X = normalize(PHOTOS)
    assert X.shape == (5, 784) and X.dtype == np.float32
    assert 0 <= X.min() and X.max() <= 1


def test_one_hot():
    assert one_hot(np.array([3, 0])).tolist() == [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def test_noise():
    X, rng = normalize(PHOTOS), np.random.default_rng(0)
    assert add_noise(X, 0, rng) is X
    noisy = add_noise(X, 0.3, rng)
    assert noisy.shape == X.shape and 0 <= noisy.min() and noisy.max() <= 1
    assert not np.allclose(noisy, X)


def test_augment_and_alter_keep_the_shape():
    assert augment(PHOTOS, np.random.default_rng(0)).shape == PHOTOS.shape
    for thickness in (-2, 0, 2):
        assert alter(PHOTOS, rotation=20, thickness=thickness).shape == PHOTOS.shape


def test_prepare_drawing():
    board = Image.new("L", (280, 280))
    assert prepare_drawing(board) is None  # empty board
    ImageDraw.Draw(board).line((20, 20, 60, 120), fill=255, width=20)  # a stroke in a corner
    photo = prepare_drawing(board)
    assert photo.shape == (28, 28)
    pixels = photo.astype(float)
    rows, columns = np.indices(pixels.shape)
    # like in MNIST, the center of mass of the digit ends up in the middle
    assert abs((rows * pixels).sum() / pixels.sum() - 13.5) < 1
    assert abs((columns * pixels).sum() / pixels.sum() - 13.5) < 1

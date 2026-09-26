"""The photos: saving and reading them, normalization, noise, alterations, drawings made with the mouse."""
import os
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from nn_digits.neural_net import data
from nn_digits.neural_net.data import (add_noise, alter, augment, load_photos, normalize, one_hot,
                                       prepare_drawing, save_collection)

PHOTOS = np.random.default_rng(0).integers(0, 256, (5, 28, 28), dtype=np.uint8)
RNG = np.random.default_rng(1)
MNIST = {"x_train": RNG.integers(0, 256, (60, 28, 28), dtype=np.uint8), "y_train": np.arange(60, dtype=np.uint8) % 10,
         "x_test": RNG.integers(0, 256, (20, 28, 28), dtype=np.uint8), "y_test": np.arange(20, dtype=np.uint8) % 10}


@pytest.fixture
def photos_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PHOTOS_DIR", tmp_path)
    return tmp_path


def test_collection_size():
    size = data.collection_size()
    assert 40e6 < size["mnist"] + size["photos"] < 45e6 and size["disk"] == size["mnist"] + size["photos"]
    assert 290e6 < data.collection_size(block=4096)["disk"] < 330e6  # each small file takes a whole 4 KB block


def test_save_collection(photos_dir):
    fractions = []
    assert save_collection(MNIST, 3, None, progress=fractions.append) == (30, 20)  # None = all the photos
    assert fractions[0] == 0 and fractions[-1] == 1
    photos, digits = load_photos("train")  # from the quick copy
    (photos_dir / "train.npz").unlink()
    from_png = load_photos("train")  # from the PNG files: the same photos, in the same order
    assert np.array_equal(photos, from_png[0]) and np.array_equal(digits, from_png[1])
    assert (photos_dir / "train.npz").exists()  # the copy is made again
    assert np.bincount(digits).tolist() == [3] * 10
    assert sorted(load_photos("test")[1].tolist()) == sorted(MNIST["y_test"].tolist())


def test_the_quick_copy_is_used_instead_of_the_png_files(photos_dir, monkeypatch):
    save_collection(MNIST, 2, 1)
    monkeypatch.setattr(data.Image, "open", None)  # reading a PNG would fail
    assert len(load_photos("train")[0]) == 20


def test_photos_added_by_hand_are_read(photos_dir):
    save_collection(MNIST, 2, 1)
    before = time.time() - 10  # as if the photos had been saved 10 seconds ago
    for path in [photos_dir / "train.npz", photos_dir / "train", *(photos_dir / "train").iterdir()]:
        os.utime(path, (before, before))
    Image.new("L", (28, 28)).save(photos_dir / "train" / "3" / "mine.png")
    photos, digits = load_photos("train")
    assert len(photos) == 21 and np.sum(digits == 3) == 3


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

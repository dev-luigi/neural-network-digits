"""
The digit photos: download, loading and preparation for the network.

Each photo is 28x28 pixels in grayscale (0 = black, 255 = white),
with the white digit on a black background, like in the famous MNIST dataset.
"""
import shutil
import urllib.request

import numpy as np
from PIL import Image, ImageFilter

from i18n import tr
from neural_net.storage import MNIST_FILE, PHOTOS_DIR

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


# ---------------------------------------------------------------- download and saving

def download_mnist(progress=None):
    """Downloads MNIST (70,000 handwritten digits, ~11 MB) and returns its photos.
    It downloads it only once, then uses the copy saved in data/photos/mnist.npz.
    progress(fraction), if given, is called during the download with a number from 0 to 1."""
    if not MNIST_FILE.exists():
        MNIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        partial = MNIST_FILE.with_suffix(".partial")  # if the download stops, no broken file is left behind
        with urllib.request.urlopen(MNIST_URL, timeout=60) as response, open(partial, "wb") as file:
            total = int(response.headers.get("Content-Length", 0))
            while block := response.read(65536):
                file.write(block)
                if progress and total:
                    progress(file.tell() / total)
        partial.replace(MNIST_FILE)
    with np.load(MNIST_FILE) as mnist:
        return {name: mnist[name] for name in ("x_train", "y_train", "x_test", "y_test")}


def save_photos(photos, digits, split, per_digit, rng):
    """Picks `per_digit` random photos of each digit and saves them as PNG in data/photos/<split>/<digit>/."""
    folder = PHOTOS_DIR / split
    shutil.rmtree(folder, ignore_errors=True)  # away with the old photos
    for digit in range(10):
        (folder / str(digit)).mkdir(parents=True)
        available = np.flatnonzero(digits == digit)
        for index in rng.choice(available, min(per_digit, len(available)), replace=False):
            Image.fromarray(photos[index]).save(folder / str(digit) / f"mnist_{index:05d}.png")


def count_photos(split):
    """How many photos there are in data/photos/<split>/  (split = "train" or "test")."""
    return len(list((PHOTOS_DIR / split).glob("*/*.png")))


def load_photos(split):
    """Reads the "train" or "test" photos.
    Returns (photos, digits): photos is an array [N, 28, 28] with values 0-255, digits says which digit each one is."""
    photos, digits = [], []
    for digit in range(10):
        for file in sorted((PHOTOS_DIR / split / str(digit)).glob("*.png")):
            photos.append(np.array(Image.open(file).convert("L")))
            digits.append(digit)
    if not photos:
        raise FileNotFoundError(tr("No photos in {folder}: download them first.", folder=PHOTOS_DIR / split))
    return np.stack(photos), np.array(digits)


# ---------------------------------------------------------------- preparation for the network

def normalize(photos):
    """From [N, 28, 28] with values 0-255 to [N, 784] with values 0-1: each photo becomes a row of numbers."""
    return photos.reshape(len(photos), -1).astype(np.float32) / 255


def one_hot(digits):
    """The right answer in the network's format: 3 becomes [0,0,0,1,0,0,0,0,0,0]."""
    return np.eye(10, dtype=np.float32)[digits]


def augment(photos, rng, rotation=12, shift=2):
    """Data augmentation: rotates and shifts each photo a tiny bit, at random (at most by `rotation`
    degrees and by `shift` pixels). This way, at every round the network sees slightly different digits
    and learns the shape, not the single pixels."""
    shift = int(shift)
    if rotation == 0 and shift == 0:
        return photos
    result = []
    for photo in photos:
        angle = rng.uniform(-rotation, rotation)
        dx, dy = rng.integers(-shift, shift + 1, size=2)
        rotated = Image.fromarray(photo).rotate(angle, resample=Image.Resampling.BILINEAR,
                                                translate=(int(dx), int(dy)))
        result.append(np.array(rotated))
    return np.stack(result)


def alter(photos, rotation=0.0, thickness=0):
    """Spoils the photos on purpose to put the network to the test: rotates them by `rotation` degrees and
    thickens (thickness > 0) or thins (thickness < 0) the stroke, one pass for each unit."""
    thickness = int(thickness)
    if rotation == 0 and thickness == 0:
        return photos
    image_filter = ImageFilter.MaxFilter(3) if thickness > 0 else ImageFilter.MinFilter(3)
    result = []
    for photo in photos:
        image = Image.fromarray(photo).rotate(rotation, resample=Image.Resampling.BILINEAR)
        for _ in range(abs(thickness)):
            image = image.filter(image_filter)
        result.append(np.array(image))
    return np.stack(result)


def add_noise(X, sigma, rng):
    """Gaussian noise: to each pixel (values 0-1) we add a random number drawn from a gaussian
    as wide as sigma. The bigger sigma is, the "dirtier" the photo. With sigma = 0 nothing changes."""
    if sigma <= 0:
        return X
    return np.clip(X + rng.normal(0, sigma, X.shape).astype(np.float32), 0, 1)


def prepare_drawing(drawing):
    """Turns a drawing made with the mouse into a 28x28 photo similar to the MNIST ones:
    crops the digit, shrinks it to 20x20 and centers it on its center of mass.
    Returns None if the drawing is empty."""
    box = drawing.getbbox()  # the rectangle that contains the stroke
    if box is None:
        return None
    digit = drawing.crop(box)
    scale = 20 / max(digit.size)
    digit = digit.resize((max(1, round(digit.width * scale)), max(1, round(digit.height * scale))),
                         Image.Resampling.LANCZOS)
    photo = Image.new("L", (28, 28))
    photo.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))

    # Moves the digit so that its center of mass (the "balance point" of the pixels) falls in the middle
    pixels = np.asarray(photo, dtype=np.float32)
    if pixels.sum() == 0:
        return None
    rows, columns = np.indices(pixels.shape)
    dy = round(13.5 - (rows * pixels).sum() / pixels.sum())
    dx = round(13.5 - (columns * pixels).sum() / pixels.sum())
    return np.array(photo.transform((28, 28), Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy)))

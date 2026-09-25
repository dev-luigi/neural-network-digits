"""
Where the program saves the files it creates: everything goes in the project's "data" folder.

    data/photos/          the digit photos (train and test) as PNG, their quick copy (train.npz and test.npz)
                          and the copy of MNIST
    data/model.npz        the trained network (that is, its weights)
    data/charts/          the charts saved by the terminal commands
    data/settings.json    the program preferences (reset does not delete them)

To start again from scratch, just delete these files: that is what reset() does.
"""
import json
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PHOTOS_DIR = DATA_DIR / "photos"
MNIST_FILE = PHOTOS_DIR / "mnist.npz"
MODEL_FILE = DATA_DIR / "model.npz"
CHARTS_DIR = DATA_DIR / "charts"
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_SETTINGS = {"check_updates": True, "skipped_version": "", "language": ""}


def settings():
    """The saved preferences (if the file is missing or broken, the default ones)."""
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)


def save_setting(name, value):
    values = settings()
    values[name] = value
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(values, indent=2), encoding="utf-8")


def to_delete(include_photos=False):
    """What reset would delete (only the files and folders that really exist)."""
    paths = [MODEL_FILE, CHARTS_DIR] + ([PHOTOS_DIR] if include_photos else [])
    return [path for path in paths if path.exists()]


def reset(include_photos=False):
    """Deletes model and charts; with include_photos=True the photos too. The code is never touched."""
    for path in to_delete(include_photos):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def save_chart(fig, name):
    """Saves a matplotlib figure in data/charts/<name> and returns the path."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=110)
    return path

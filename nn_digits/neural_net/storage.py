"""
Where the program saves the files it creates: everything goes in one "data" folder.

    data/photos/          the digit photos (train and test) as PNG, their quick copy (train.npz and test.npz)
                          and the copy of MNIST
    data/model.npz        the trained network (that is, its weights)
    data/charts/          the charts saved by the terminal commands
    data/settings.json    the program preferences (reset does not delete them)

Where that folder is depends on how the program was installed:
  - downloaded as a zip or with git: "data" next to start.py, in the program folder;
  - installed with pip: the program folder is not ours to write in, so the data goes in the folder of the user
    (Windows: %APPDATA%\\neural-network-digits, macOS: ~/Library/Application Support/neural-network-digits,
    Linux: ~/.local/share/neural-network-digits).

To start again from scratch, just delete these files: that is what reset() does.
"""
import json
import os
import shutil
import sys
from pathlib import Path

PROGRAM_DIR = Path(__file__).resolve().parents[2]  # the folder that contains the nn_digits package
# Zip and git copies have the launchers next to the package; an installation made by pip does not
PORTABLE = all((PROGRAM_DIR / name).is_file() for name in ("start.py", "start.bat", "start.sh"))


def user_data_dir():
    """The folder where each system keeps the data of the programs of the user."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "neural-network-digits"


DATA_DIR = PROGRAM_DIR / "data" if PORTABLE else user_data_dir()
PHOTOS_DIR = DATA_DIR / "photos"
MNIST_FILE = PHOTOS_DIR / "mnist.npz"
MODEL_FILE = DATA_DIR / "model.npz"
CHARTS_DIR = DATA_DIR / "charts"
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_SETTINGS = {"check_updates": True, "skipped_version": "", "language": "",
                    "assistant": False, "hints": True}  # assistant = the panel is open


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

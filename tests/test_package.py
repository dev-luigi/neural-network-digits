"""How the program is installed: as a zip or with git (start.py next to nn_digits/), or with pip (the package alone)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from nn_digits import project
from nn_digits.neural_net import storage

ROOT = Path(__file__).resolve().parent.parent


def test_the_repository_keeps_its_data_inside():
    assert storage.PORTABLE and storage.PROGRAM_DIR == ROOT
    assert storage.DATA_DIR == ROOT / "data"


def test_the_data_folder_of_pip(monkeypatch, tmp_path):
    """Installed with pip, the data goes in the folder that each system keeps for the programs of the user."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert storage.user_data_dir() == tmp_path / "Roaming" / "neural-network-digits"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert storage.user_data_dir() == tmp_path / "Library" / "Application Support" / "neural-network-digits"
    monkeypatch.setattr(sys, "platform", "linux")
    assert storage.user_data_dir() == tmp_path / ".local" / "share" / "neural-network-digits"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert storage.user_data_dir() == tmp_path / "xdg" / "neural-network-digits"


def test_the_old_versions_find_the_program_in_the_new_zip():
    """Up to 1.1.1 the updater takes as the program the folder of the zip with project.py and start.py:
    it must be the main folder, and only that one (otherwise nn_digits/ would end up in the wrong place)."""
    folders = [ROOT, *(path for path in (ROOT / "nn_digits").rglob("*") if path.is_dir())]
    assert [f for f in folders if (f / "project.py").exists() and (f / "start.py").exists()] == [ROOT]


def test_the_old_project_py_still_works():
    spec = importlib.util.spec_from_file_location("old_project", ROOT / "project.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VERSION == project.VERSION


def test_pip_installs_the_command_and_the_same_libraries():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11 and newer
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [line for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith("#")]
    assert pyproject["dependencies"] == requirements
    assert pyproject["name"] == project.PYPI_NAME
    assert pyproject["scripts"] == {project.PYPI_NAME: "nn_digits.cli:main"}

"""Updates: comparing versions, changes from the CHANGELOG, installing a zip."""
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from nn_digits import updater


def test_version_comparison():
    assert updater.version_numbers("v1.10.2") == (1, 10, 2)
    assert updater.is_newer("1.10.0", "1.9.3")  # as numbers, not as text
    assert updater.is_newer("v2.0.0", "1.99.99")
    assert not updater.is_newer("1.0.0", "1.0.0")


def test_release_notes(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [1.1.0] - 2026-10-01\n\n- new tab\n\n## [1.0.0] - 2026-09-25\n\n"
                         "- first version\n", encoding="utf-8")
    assert updater.release_notes("1.1.0", changelog) == "- new tab"
    assert updater.release_notes("1.0.0", changelog) == "- first version"
    assert updater.release_notes("9.9.9", changelog) == ""
    assert updater.release_notes("1.0.0", tmp_path / "missing.md") == ""


@pytest.fixture
def program(tmp_path, monkeypatch):
    """A fake installation of the program, version 1.0.0, with the user's photos and model."""
    folder = tmp_path / "program"
    (folder / "data").mkdir(parents=True)
    (folder / "data" / "model.npz").write_text("my model")
    (folder / "nn_digits").mkdir()
    (folder / "nn_digits" / "old.py").write_text("old")
    (folder / "nn_digits" / "project.py").write_text('VERSION = "1.0.0"')
    (folder / "start.py").write_text("old")
    (folder / "requirements.txt").write_text("numpy\n")
    monkeypatch.setattr(updater, "APP_DIR", folder)
    return folder


def make_zip(path, files):
    """A zip like the release ones: everything inside a folder."""
    with zipfile.ZipFile(path, "w") as z:
        for name, content in files.items():
            z.writestr(f"neural-network-digits-v2.0.0/{name}", content)
    return path.as_uri()  # urllib can read local files too (file://...)


VERSION_2 = {"start.py": "new", "nn_digits/project.py": 'VERSION = "2.0.0"', "nn_digits/new.py": "new",
             "requirements.txt": "numpy\npillow\n", "data/model.npz": "must NOT overwrite"}


def test_install_replaces_the_program_but_not_data(program, tmp_path):
    libraries_changed = updater.install(make_zip(tmp_path / "v2.zip", VERSION_2))
    assert libraries_changed
    assert (program / "start.py").read_text() == "new"
    assert (program / "nn_digits" / "new.py").exists()
    assert not (program / "nn_digits" / "old.py").exists()  # the code folders are replaced
    assert (program / "data" / "model.npz").read_text() == "my model"


def test_update_from_a_version_without_nn_digits(program, tmp_path):
    """Up to 1.1.1 the code was next to start.py: what is left of it goes away, the data stays."""
    shutil.rmtree(program / "nn_digits")
    for name in ("gui/tab_draw.py", "neural_net/network.py", "locales/it.json", "i18n.py", "updater.py",
                 "__pycache__/i18n.cpython-313.pyc"):
        (program / name).parent.mkdir(exist_ok=True)
        (program / name).write_text("old")
    (program / "project.py").write_text('VERSION = "1.1.1"')
    updater.install(make_zip(tmp_path / "v2.zip", {**VERSION_2, "project.py": "from nn_digits.project import *"}))
    assert sorted(p.name for p in program.iterdir()) == ["data", "nn_digits", "project.py", "requirements.txt",
                                                         "start.py"]
    assert (program / "data" / "model.npz").read_text() == "my model"


def test_the_zip_of_an_old_version_is_refused(program, tmp_path):
    """A zip without nn_digits/ (version 1.1.1 or older) is not the program for this updater."""
    with pytest.raises(ValueError):
        updater.install(make_zip(tmp_path / "old.zip", {"start.py": "x", "project.py": 'VERSION = "1.1.1"'}))
    assert (program / "start.py").read_text() == "old"


def test_same_libraries(program, tmp_path):
    assert not updater.install(make_zip(tmp_path / "v2.zip", {**VERSION_2, "requirements.txt": "numpy\n"}))


@pytest.mark.skipif(os.name == "nt", reason="Windows has no executable bit")
def test_executable_files_stay_executable(program, tmp_path):
    path = tmp_path / "v2.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, content in VERSION_2.items():
            z.writestr(f"neural-network-digits-v2.0.0/{name}", content)
        script = zipfile.ZipInfo("neural-network-digits-v2.0.0/start.sh")
        script.external_attr = 0o100755 << 16  # like git archive does for the executable files
        z.writestr(script, "#!/bin/sh\n")
    updater.install(path.as_uri())
    assert os.access(program / "start.sh", os.X_OK)
    assert not os.access(program / "start.py", os.X_OK)


@pytest.mark.parametrize("content", [{"../outside.txt": "x"}, {"readme.txt": "no program"}])
def test_refuses_wrong_zips(program, tmp_path, content):
    with pytest.raises(ValueError):
        updater.install(make_zip(tmp_path / "strange.zip", content))
    assert (program / "start.py").read_text() == "old"


def test_if_something_goes_wrong_puts_the_old_files_back(program, tmp_path, monkeypatch):
    move = shutil.move

    def broken_move(src, dst):
        if Path(src).name == "requirements.txt" and "extracted" in str(src):
            raise OSError("disk full")
        return move(src, dst)

    monkeypatch.setattr(shutil, "move", broken_move)
    with pytest.raises(OSError):
        updater.install(make_zip(tmp_path / "v2.zip", VERSION_2))
    assert (program / "start.py").read_text() == "old"
    assert (program / "nn_digits" / "old.py").exists()
    assert not (program / "nn_digits" / "new.py").exists()
    assert (program / "requirements.txt").read_text() == "numpy\n"


def test_installed_with_pip(monkeypatch):
    monkeypatch.setattr(updater.storage, "PORTABLE", False)
    assert updater.installed_with_pip()
    assert not updater.installed_with_git()
    started = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda command, **options: started.append(command))
    updater.restart()
    assert started == [[sys.executable, "-m", "nn_digits"]]  # there is no start.py: it starts as a module

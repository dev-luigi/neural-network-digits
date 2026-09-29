"""Updates: comparing versions, changes from the CHANGELOG, installing a zip."""
import os
import shutil
import subprocess
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


def git(folder, *args):
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@test", *args], cwd=folder, check=True,
                   capture_output=True)


def commit(folder, files):
    for name, content in files.items():
        (folder / name).write_text(content)
    git(folder, "add", ".")
    git(folder, "commit", "-m", "version")


@pytest.fixture
def git_copy(tmp_path, monkeypatch):
    """A copy made with git clone, and the repository it comes from ("GitHub")."""
    github = tmp_path / "github"
    github.mkdir()
    git(github, "init")
    commit(github, {"start.py": "old", "requirements.txt": "numpy\n"})
    git(tmp_path, "clone", str(github), "program")
    monkeypatch.setattr(updater, "APP_DIR", tmp_path / "program")
    monkeypatch.setattr(updater.storage, "PORTABLE", True)
    return github, tmp_path / "program"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_git_copy_updates_itself_with_git_pull(git_copy):
    github, program = git_copy
    assert updater.install_method() == "git"
    assert updater.update_command() == ["git", "pull", "--ff-only"]
    commit(github, {"start.py": "new", "requirements.txt": "numpy\npillow\n"})
    assert updater.update("no zip for git copies")  # the libraries changed
    assert (program / "start.py").read_text() == "new"
    assert not updater.update("no zip for git copies")  # already up to date


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_pull_does_not_touch_files_changed_by_hand(git_copy):
    github, program = git_copy
    (program / "start.py").write_text("mine")
    commit(github, {"start.py": "new"})
    with pytest.raises(RuntimeError, match="git pull --ff-only"):  # the error, and the command to run by hand
        updater.update("no zip for git copies")
    assert (program / "start.py").read_text() == "mine"


def test_a_missing_tool_says_what_to_do(program, monkeypatch):
    def missing(*args, **options):
        raise FileNotFoundError("pipx")

    monkeypatch.setattr(updater.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="pipx upgrade neural-network-digits"):
        updater.run(["pipx", "upgrade", "neural-network-digits"])


@pytest.fixture
def pip_copy(tmp_path, monkeypatch):
    """A copy installed from PyPI, in its own environment (sys.prefix) with nothing inside yet."""
    monkeypatch.setattr(updater.storage, "PORTABLE", False)
    monkeypatch.setattr(updater.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(updater, "APP_DIR", tmp_path / "site-packages")
    monkeypatch.setattr(updater.shutil, "which", lambda program: f"/bin/{program}")  # every tool is there
    return tmp_path


@pytest.mark.parametrize("note, method, command", [
    ("pipx_metadata.json", "pipx", ["pipx", "upgrade", "neural-network-digits"]),
    ("uv-receipt.toml", "uv tool", ["uv", "tool", "upgrade", "neural-network-digits"]),
    (None, "pip", [sys.executable, "-m", "pip", "install", "--upgrade", "neural-network-digits"]),
])
def test_it_updates_with_the_tool_that_installed_it(pip_copy, note, method, command):
    if note:
        (pip_copy / note).write_text("")  # the note that pipx and uv leave in the environment of the program
    assert updater.install_method() == method
    assert updater.update_command() == command
    assert updater.release_site() == "PyPI"


def test_pip_in_the_folder_of_the_user(pip_copy, monkeypatch):
    monkeypatch.setattr(updater.site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(updater.site, "getusersitepackages", lambda: str(pip_copy / "site-packages"))
    monkeypatch.setattr(updater.sysconfig, "get_path", lambda name: str(pip_copy / "stdlib"))
    assert updater.install_method() == "pip --user"
    assert updater.update_command()[-2:] == ["--user", "neural-network-digits"]
    (pip_copy / "stdlib").mkdir()
    (pip_copy / "stdlib" / "EXTERNALLY-MANAGED").write_text("")  # Debian, Ubuntu: PEP 668
    assert updater.update_command()[-3:] == ["--user", "--break-system-packages", "neural-network-digits"]


def test_installed_with_uv_pip(pip_copy):
    info = pip_copy / "site-packages" / "neural_network_digits-1.0.0.dist-info"
    info.mkdir(parents=True)
    (info / "INSTALLER").write_text("uv\n")
    assert updater.install_method() == "uv pip"
    assert updater.update_command()[:3] == ["uv", "pip", "install"]


def test_pipx_without_its_command_uses_the_pip_of_its_environment(pip_copy, monkeypatch):
    (pip_copy / "pipx_metadata.json").write_text("")
    monkeypatch.setattr(updater.shutil, "which", lambda program: None)
    assert updater.update_command()[:3] == [sys.executable, "-m", "pip"]
    # The pipx environments made with uv have no pip: better the pipx command, even if it is not found
    monkeypatch.setattr(updater.importlib.util, "find_spec", lambda name: None)
    assert updater.update_command() == ["pipx", "upgrade", "neural-network-digits"]


@pytest.mark.parametrize("works", [True, False])
def test_on_windows_the_running_exe_goes_aside(pip_copy, monkeypatch, works):
    """Windows cannot replace a running .exe, but it can rename it."""
    scripts = pip_copy / "Scripts"
    scripts.mkdir()
    (scripts / "neural-network-digits.exe").write_text("old")
    monkeypatch.setattr(updater, "os", type("FakeOs", (), {"name": "nt"}))
    monkeypatch.setattr(updater.sys, "argv", [str(scripts / "neural-network-digits")])  # pip removes ".exe"

    def pip(command):
        assert not (scripts / "neural-network-digits.exe").exists()  # free for the new one
        if not works:
            raise RuntimeError("no internet")
        (scripts / "neural-network-digits.exe").write_text("new")

    monkeypatch.setattr(updater, "run", pip)
    if works:
        updater.update(None)
        assert (scripts / "neural-network-digits.exe").read_text() == "new"
        assert (scripts / "neural-network-digits.exe.old").read_text() == "old"
    else:
        with pytest.raises(RuntimeError):
            updater.update(None)
        assert (scripts / "neural-network-digits.exe").read_text() == "old"  # put back


def test_zip_copy(program):
    assert updater.install_method() == "zip"
    assert updater.update_command() is None  # it downloads the zip by itself
    assert updater.release_site() == "GitHub"


GITHUB_ANSWER = {"tag_name": "v1.3.0", "body": "- news", "html_url": "https://github.com/release",
                 "assets": [], "zipball_url": "https://github.com/zip"}


def test_the_copies_from_pypi_ask_pypi(pip_copy, monkeypatch):
    """pip takes the new version from PyPI, where it arrives a few minutes after GitHub."""
    answers = {updater.LATEST_RELEASE_API: GITHUB_ANSWER, updater.PYPI_API: {"info": {"version": "1.2.9"}}}
    monkeypatch.setattr(updater, "read_json", lambda url, timeout, accept="": answers[url])
    assert updater.latest_release()["version"] == "1.2.9"  # not yet the 1.3.0 of GitHub
    answers[updater.PYPI_API] = {"info": {"version": "1.3.0"}}
    assert updater.latest_release()["notes"] == "- news"  # same version: the news come from GitHub


def test_the_other_copies_ask_github(program, monkeypatch):
    monkeypatch.setattr(updater, "read_json", lambda url, timeout, accept="": {
        updater.LATEST_RELEASE_API: GITHUB_ANSWER}[url])
    assert updater.latest_release() == {"version": "1.3.0", "notes": "- news", "page": "https://github.com/release",
                                        "zip": "https://github.com/zip"}

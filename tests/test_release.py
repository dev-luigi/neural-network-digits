"""The release tool: the CI/CD checks and preparing a new version."""
import importlib.util
import shutil
from pathlib import Path

import pytest

from nn_digits.project import VERSION

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release", ROOT / "tools" / "release.py")
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


def test_the_current_version_has_its_changes():
    """If it fails: CHANGELOG.md is missing the paragraph of the version written in nn_digits/project.py."""
    release.check(f"v{VERSION}")


def test_tag_different_from_the_version():
    with pytest.raises(SystemExit):
        release.check("v0.0.1")


@pytest.fixture
def project_copy(tmp_path, monkeypatch):
    shutil.copy(ROOT / "nn_digits" / "project.py", tmp_path / "project.py")
    shutil.copy(ROOT / "CHANGELOG.md", tmp_path / "CHANGELOG.md")
    monkeypatch.setattr(release, "PROJECT_FILE", tmp_path / "project.py")
    monkeypatch.setattr(release, "CHANGELOG", tmp_path / "CHANGELOG.md")
    return tmp_path


def test_prepare_a_new_version(project_copy):
    major, minor, _ = map(int, VERSION.split("."))
    new = f"{major}.{minor + 1}.0"
    release.prepare(new)
    assert release.current_version() == new
    changelog = (project_copy / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.index(f"## [{new}]") < changelog.index(f"## [{VERSION}]")  # the newest at the top
    with pytest.raises(SystemExit):  # the changes have not been written yet
        release.check(f"v{new}")
    changelog = changelog.replace(release.TODO_NOTE, "- something new")
    (project_copy / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    release.check(f"v{new}")


@pytest.mark.parametrize("wrong", ["0.9.0", VERSION, "1.2", "abc"])
def test_prepare_refuses_wrong_versions(project_copy, wrong):
    with pytest.raises(SystemExit):
        release.prepare(wrong)

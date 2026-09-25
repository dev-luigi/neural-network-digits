"""The interface opens, every tab shows up without errors, the update dialog works.
It needs a screen: without one (for example on a server) these tests are skipped."""
import tkinter as tk

import pytest

from neural_net import storage
from project import VERSION


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)  # the settings of the tests do not touch the real ones
    monkeypatch.setattr(storage, "SETTINGS_FILE", tmp_path / "settings.json")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no screen available")
    errors = []
    root.report_callback_exception = lambda *error: errors.append(error)
    from gui.window import MainWindow
    window = MainWindow(root, check_updates=False)
    window.errors = errors
    yield window
    root.destroy()


def test_every_tab_opens(window):
    for index in range(len(window.all_tabs)):
        window.tabs.select(index)
        window.root.update()
    assert not window.errors
    assert VERSION in window.root.title()


def test_readable_markdown():
    from gui.tab_info import readable_markdown
    markdown = "First.\n\n- **Data**: a line\n  that goes on\n  - under `point`\n- other"
    assert readable_markdown(markdown) == "First.\n\n•  Data: a line that goes on\n    •  under point\n•  other"


def test_update_dialog(window):
    info = window.info_tab
    assert info._handle_release({"version": VERSION, "notes": "", "page": "", "zip": ""}, silent=False) is None
    assert "latest version" in info.status.cget("text")

    release = {"version": "99.0.0", "notes": "- lots of things", "page": "", "zip": ""}
    dialog = info._handle_release(release, silent=False)
    assert dialog is not None and "99.0.0" in info.status.cget("text")
    dialog.skip()  # "Skip this version": at startup it is not offered anymore...
    assert storage.settings()["skipped_version"] == "99.0.0"
    assert info._handle_release(release, silent=True) is None
    assert info._handle_release(release, silent=False) is not None  # ...but from the Info tab it is
    assert not window.errors

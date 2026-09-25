"""The interface opens, every tab shows up without errors, the update dialog works, the assistant and Pick work.
It needs a screen: without one (for example on a server) these tests are skipped."""
import gc
import tkinter as tk

import pytest

from assistant import knowledge, rules
from gui import base
from gui.assistant import WIDTH
from neural_net import storage
from project import VERSION


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)  # the settings of the tests do not touch the real ones
    monkeypatch.setattr(storage, "SETTINGS_FILE", tmp_path / "settings.json")
    root = None
    for _attempt in range(3):  # on Windows a new Tk right after the one of the previous test can fail once
        try:
            root = tk.Tk()
            break
        except tk.TclError as error:
            problem = error
            gc.collect()
    if root is None:
        pytest.skip(f"no screen available ({str(problem).splitlines()[0]})")
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


def test_the_state_seen_by_the_assistant(window):
    from gui.app_state import app_state
    for index in range(len(window.all_tabs)):
        window.tabs.select(index)
        window.root.update()
        state = app_state(window)
        assert state["tab"] == knowledge.TABS[index]
        assert state["training"]["params"]["lr"] == 0.05
        assert window.assistant.brain().answer("how is it going?", state).found
    assert not window.errors


def test_assistant_panel(window):
    panel = window.assistant
    window.root.update()
    width = window.root.winfo_width()
    room = window.root.winfo_screenwidth() >= width + WIDTH  # small screens (like the ones of the CI) have none
    window.toggle_assistant(True)
    window.root.update()
    assert panel.visible and storage.settings()["assistant"]
    assert "Hi!" in panel.chat.get("1.0", "end")
    if room:  # the window gets wider, so the tabs keep their space...
        assert window.root.winfo_width() == width + WIDTH
    panel.ask("what is the learning rate?")
    assert "How big each correction of the weights is" in panel.chat.get("1.0", "end")
    window.toggle_assistant(False)
    window.root.update()
    assert not panel.visible and not storage.settings()["assistant"]
    if room:  # ...and goes back as it was when the panel closes
        assert window.root.winfo_width() == width
    assert not window.errors


def test_pick_explains_a_control_without_pressing_it(window):
    window.tabs.select(1)
    pressed = []
    button = base.button(window.training_tab.frame, "Try me", lambda: pressed.append(True),
                         explanation="A button to try Pick.")
    button.place(x=0, y=0)  # on top of the tab: packed at the bottom it would not fit on a small screen
    window.root.update()

    def click():
        for event in ("<Enter>", "<Button-1>", "<ButtonRelease-1>"):
            button.event_generate(event, x=5, y=5)
        window.root.update()

    window.picker.set(True)
    assert window.assistant.visible  # the explanations of Pick appear in the panel
    click()
    assert pressed == []
    assert "Pick: Try me\nTry me\nA button to try Pick." in window.assistant.chat.get("1.0", "end")
    window.picker.show(button)  # the orange frame around the control
    assert window.picker.target is button and all(border.place_info() for border in window.picker.borders)
    window.picker.set(False)
    assert window.picker.target is None and not any(border.place_info() for border in window.picker.borders)
    click()
    assert pressed == [True]
    assert not window.errors


def test_hints_and_their_fix(window):
    panel = window.assistant
    state = {"tab": "training", "photos": {"train": 5000, "test": 1000}, "model": True,
             "training": {"params": {"noise": 0.5}}}
    [hint] = panel.check_hints(state)
    assert hint.key == "noise" and panel.check_hints(state) == []  # a hint already written is not repeated
    assert window.assistant_button.cget("text") == "Assistant · 1 new"  # the panel is closed: it counts them
    panel.apply(hint)
    window.root.update()
    assert window.training_tab.params["noise"] == pytest.approx(0.1)
    assert "Done: Noise σ 0.10." in panel.chat.get("1.0", "end")

    panel.apply(rules.Hint("photos", "", (("tab", "data"),), "Go to tab 1", "mnist"))
    window.root.update()
    assert window.tabs.index("current") == 0
    window.toggle_assistant(True)
    assert window.assistant_button.cget("text") == "Assistant (F2)"
    panel.hints_on.set(False)
    assert panel.check_hints(dict(state, training={"params": {"rotation": 40}})) == []
    assert not window.errors


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

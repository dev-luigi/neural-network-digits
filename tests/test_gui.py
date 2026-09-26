"""The interface opens, every tab shows up without errors, the update dialog works, the assistant and Pick work.
It needs a screen: without one (for example on a server) these tests are skipped."""
import gc
import time
import tkinter as tk

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image, ImageColor

from nn_digits.assistant import knowledge, rules
from nn_digits.gui import base
from nn_digits.gui.assistant import FRAME, MOST_FRAMES, THINKING, WIDTH
from nn_digits.neural_net import charts, storage
from nn_digits.project import VERSION


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
    from nn_digits.gui.window import MainWindow
    window = MainWindow(root, check_updates=False)
    window.errors = errors
    yield window
    root.destroy()


def answered(window, limit=3.0):
    """Waits until the assistant has written its answer (after the "thinking" animation). Returns the seconds."""
    start = time.perf_counter()
    while window.assistant.waiting is not None and time.perf_counter() - start < limit:
        window.root.update()
        time.sleep(0.005)
    return time.perf_counter() - start


def test_every_tab_opens(window):
    for index in range(len(window.all_tabs)):
        window.tabs.select(index)
        window.root.update()
    assert not window.errors
    assert VERSION in window.root.title()


def test_the_state_seen_by_the_assistant(window):
    from nn_digits.gui.app_state import app_state
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
    answered(window)
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
    answered(window)
    assert pressed == []
    assert "Pick: Try me\nTry me\nA button to try Pick." in window.assistant.chat.get("1.0", "end")
    window.picker.show(button)  # the orange frame around the control
    assert window.picker.target is button and all(border.place_info() for border in window.picker.borders)
    window.picker.set(False)
    assert window.picker.target is None and not any(border.place_info() for border in window.picker.borders)
    click()
    assert pressed == [True]
    assert not window.errors


def test_the_answer_comes_after_a_short_animation(window):
    panel = window.assistant
    window.toggle_assistant(True)
    assert MOST_FRAMES * FRAME <= 600  # the wait is never longer than 0.6 seconds
    frames = {frame for animation in THINKING for frame in animation}

    panel.ask("what is the learning rate?")
    last_line = panel.chat.get("end-2l", "end").strip()
    assert panel.waiting is not None and last_line in frames  # first the animation, then the answer
    assert "How big each correction" not in panel.chat.get("1.0", "end")
    assert panel.check_hints({"tab": "training", "training": {"params": {"noise": 0.5}}}) == []  # hints wait too

    panel.ask("what is overfitting?")  # a new question: the answer of the one before is written first
    chat = panel.chat.get("1.0", "end")
    assert chat.index("How big each correction") < chat.index("what is overfitting?")
    assert answered(window) < 2  # 0.6 seconds, plus the slowness of the test machines
    chat = panel.chat.get("1.0", "end")
    assert "Overfitting" in chat and chat.strip().splitlines()[-1] not in frames  # the animation is gone
    assert not window.errors


def test_pick_chooses_one_chart_of_a_figure(window):
    tab, widget = window.evaluation_tab, window.evaluation_tab.canvas.get_tk_widget()
    tab.needs_update = False  # the tab must not draw the saved model (if there is one): the test draws its charts
    window.tabs.select(2)
    window.root.update()
    # A figure without charts (empty, or only a message): Pick takes the whole figure
    assert widget.pick_parts(10, 10) == base.Part("Evaluation charts", widget.pick_text, None,
                                                  (0, 0, widget.winfo_width(), widget.winfo_height()))

    rng = np.random.default_rng(0)
    photos, digits, probabilities = rng.random((40, 28, 28)), np.arange(40) % 10, rng.dirichlet(np.ones(10), 40)
    charts.evaluation(tab.fig, photos, digits, probabilities, {"noise": (np.linspace(0, 0.6, 13), rng.random(13)),
                                                               "rotation": None})
    tab.canvas.draw()
    window.root.update()

    def middle(box):  # from pixels of the figure (y going up) to pixels of the widget (y going down)
        scale = tab.fig.bbox.width / widget.winfo_width()
        return round((box.x0 + box.width / 2) / scale), round((tab.fig.bbox.height - box.y0 - box.height / 2) / scale)

    def chart(gid):
        return next(ax for ax in tab.fig.axes if ax.get_gid() == gid)

    x, y = middle(chart("wrong photo").bbox)
    part = widget.pick_parts(x, y)
    assert part.name == "A wrong photo" and part.value() == " ".join(chart("wrong photo").get_title().split())
    left, top, width, height = part.box
    assert left <= x <= left + width and top <= y <= top + height and width < widget.winfo_width() / 4
    assert widget.pick_parts(*middle(chart("confusion").bbox)).name == "Confusion matrix"
    assert widget.pick_parts(*middle(chart("rotation curve").bbox)).name == "Accuracy with more rotation"
    assert widget.pick_parts(*middle(tab.fig.subfigs[1]._suptitle.get_window_extent())) is None  # between charts

    # With Pick on the frame follows the mouse from chart to chart, and a click explains only that chart
    window.picker.set(True)
    x, y = middle(chart("noise curve").bbox)
    for event in ("<Enter>", "<Motion>", "<Button-1>", "<ButtonRelease-1>"):
        widget.event_generate(event, x=x, y=y)
    window.root.update()
    assert window.picker.target is widget and window.picker.part.name == "Accuracy with more noise"
    assert all(border.place_info() for border in window.picker.borders)
    answered(window)
    assert "Pick: Accuracy with more noise\nAccuracy with more noise\nThe accuracy on the test photos as the noise" \
        in window.assistant.chat.get("1.0", "end")
    x, y = middle(chart("confusion").bbox)
    widget.event_generate("<Motion>", x=x, y=y)
    assert window.picker.part.name == "Confusion matrix"
    window.picker.set(False)
    assert not window.errors


def test_pick_chooses_one_tile(window):
    window.tabs.select(1)
    window.root.update()
    tiles = window.training_tab.dashboard
    assert base.control_name(tiles.frame) == "Numbers of the last epoch"  # not the name of the last tile
    tile = tiles.values["validation accuracy"].master
    part = tiles.frame.pick_parts(tile.winfo_x() + 5, tile.winfo_y() + 5)
    assert part.name == "Validation accuracy" and part.value() == "-" and part.text == tiles.explanation
    assert part.box == (tile.winfo_x(), tile.winfo_y(), tile.winfo_width(), tile.winfo_height())


def test_the_empty_charts_are_crossed_out(window):
    border = ImageColor.getrgb(base.BORDER)
    bars = base.Bars(tk.Frame(window.root))
    bars.show([None] * 10, [base.ACCENT] * 10)  # no answer yet: a thin X on the bars
    assert len(bars.canvas.find_withtag("empty")) == 2
    bars.show([0.1] * 10, [base.ACCENT] * 10)
    assert not bars.canvas.find_withtag("empty")

    image = base.crossed_out_image(Image.new("L", (20, 10)))  # a grey photo: the X keeps its color
    assert image.getpixel((0, 0)) == image.getpixel((19, 9)) == image.getpixel((19, 0)) == border
    tab = window.training_tab
    tab.examples = None  # no photos: the preview is crossed out too
    tab._show_preview()
    assert tuple(int(v) for v in window.root.tk.splitlist(window.root.tk.call(str(tab._preview_photo), "get", 0, 0))) \
        == border

    fig = Figure()
    base.message(fig, "Nothing to show")  # a whole figure with nothing to show
    assert [text.get_text() for text in fig.texts] == ["Nothing to show"] and len(fig.artists) == 2
    assert not window.errors


def test_the_charts_can_be_found_by_the_assistant(window):
    reply = window.assistant.brain().answer("what does the gaussian of the weights show?", {"tab": "training"})
    assert reply.title == "Gaussian of the weights"


def test_the_language_is_chosen_next_to_pick(window, monkeypatch):
    window.root.update()
    languages = window.pick_button.master.winfo_children()[0]
    buttons = {button.cget("text"): button for button in languages.winfo_children()
               if isinstance(button, tk.Radiobutton)}
    assert list(buttons) == ["EN", "IT"] and window.language.get() == "en"
    assert languages.winfo_x() + languages.winfo_width() <= window.pick_button.winfo_x()  # on its left
    asked = []
    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *question: asked.append(question) or False)
    buttons["IT"].invoke()
    assert storage.settings()["language"] == "it" and asked  # saved, and it offers to restart (here: not now)
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
    from nn_digits.gui.tab_info import readable_markdown
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

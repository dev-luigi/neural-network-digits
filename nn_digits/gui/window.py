"""
The main window: 5 tabs, one for each step, plus the Info tab, and the assistant on the right.

The tabs "talk" to each other with two notices:
  - photos_changed()  after a download or a reset: all the tabs must be redone
  - model_changed()   after a training: Evaluation, Draw and Inside the network must be redone
To start quickly, each tab draws itself only when you open it (see base.Tab).
Shortly after opening, a background check looks on GitHub for a new version.

At the top right: the language, Pick (F1, see pick.py) and the Assistant panel (F2, see assistant.py).
"""
import re
import tkinter as tk
import traceback
from tkinter import messagebox, ttk

from nn_digits import i18n, updater
from nn_digits.gui import base
from nn_digits.gui.assistant import WIDTH, AssistantPanel
from nn_digits.gui.pick import Picker
from nn_digits.gui.tab_data import DataTab
from nn_digits.gui.tab_draw import DrawTab
from nn_digits.gui.tab_evaluation import EvaluationTab
from nn_digits.gui.tab_info import InfoTab
from nn_digits.gui.tab_inside import InsideTab
from nn_digits.gui.tab_training import TrainingTab
from nn_digits.i18n import tr
from nn_digits.neural_net import storage
from nn_digits.neural_net.data import count_photos, load_photos
from nn_digits.project import VERSION


class MainWindow:
    def __init__(self, root, check_updates=True):
        self.root = root
        root.title(tr("Neural network - digit recognition") + f"  ·  v{VERSION}")
        root.geometry("1600x980+10+0")
        root.minsize(1500, 900)
        base.apply_theme(root)
        self._photos = {}  # photos already read from disk, so they are not read again every time
        self._counts = {}  # how many photos there are, for the photos not read yet

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        self.data_tab = DataTab(self)
        self.training_tab = TrainingTab(self)
        self.evaluation_tab = EvaluationTab(self)
        self.draw_tab = DrawTab(self)
        self.inside_tab = InsideTab(self)
        self.info_tab = InfoTab(self)
        self.all_tabs = [self.data_tab, self.training_tab, self.evaluation_tab, self.draw_tab, self.inside_tab,
                         self.info_tab]

        # At the height of the tabs: the language, then Pick and the assistant (the panel on the right)
        self.picker = Picker(root, lambda control, part: self.assistant.explain_control(control, part),
                             self._pick_changed)
        self.assistant = AssistantPanel(self, self._unread_changed)
        self._size_before_panel = None
        buttons = tk.Frame(root, bg=base.BACKGROUND)
        buttons.place(in_=self.tabs, relx=1, x=-16, y=8, anchor="ne")
        self.language = tk.StringVar(value=i18n.LANGUAGE)
        languages = base.choice_buttons(buttons, "", [(code.upper(), code) for code in i18n.LANGUAGES], self.language,
                                        self._language_changed, explanation=tr(
            "The language of the program: EN = English, IT = Italian. It is applied when the program starts again: "
            "it offers to restart it right away."))
        languages.pack_configure(padx=(0, 12))
        languages.pick_name = tr("Language")
        self.pick_button = base.button(buttons, tr("Pick (F1)"), self.picker.toggle)
        self.pick_button.pack(side="left", padx=(0, 6))
        self.assistant_button = base.button(buttons, tr("Assistant (F2)"), self.toggle_assistant)
        self.assistant_button.pack(side="left")
        root.bind_all("<F1>", lambda _: self.picker.toggle())
        root.bind_all("<F2>", lambda _: self.toggle_assistant())

        self.tabs.bind("<<NotebookTabChanged>>", lambda _: self._tab_changed())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.current_tab().shown()
        if storage.settings()["assistant"]:
            self.toggle_assistant(True)
        self.assistant.start()
        if check_updates and storage.settings()["check_updates"]:
            # after opening, to start quickly
            root.after(1500, lambda: self.info_tab.check_for_updates(silent=True))

    def current_tab(self):
        return self.all_tabs[self.tabs.index("current")]

    def _tab_changed(self):
        self.picker.hide()
        self.current_tab().shown()
        if self.assistant.visible:
            self.assistant.tab_changed()

    def photos(self, split):
        """The "train" or "test" photos, read from disk only once."""
        if split not in self._photos:
            self._photos[split] = load_photos(split)
        return self._photos[split]

    def photo_count(self, split):
        """How many "train" or "test" photos there are, without reading them (the assistant asks often)."""
        if split in self._photos:
            return len(self._photos[split][1])
        if split not in self._counts:
            self._counts[split] = count_photos(split)
        return self._counts[split]

    def photos_changed(self):
        self._photos.clear()
        self._counts.clear()
        self.refresh(*self.all_tabs)

    def model_changed(self):
        self.refresh(self.evaluation_tab, self.draw_tab, self.inside_tab)

    def refresh(self, *tabs):
        """Marks the tabs to redraw: the open one updates right away, the others when you open them."""
        for tab in tabs:
            tab.needs_update = True
        self.current_tab().shown()

    # ------------------------------------------------------------------ the assistant

    def toggle_assistant(self, show=None):
        """Opens or closes the assistant panel (show=None: the opposite of now)."""
        show = not self.assistant.visible if show is None else show
        if show == self.assistant.visible:
            return
        if show:
            self.assistant.frame.pack(side="right", fill="y", before=self.tabs)
        else:
            self.assistant.frame.pack_forget()
        self._make_room(show)
        self.assistant.set_visible(show)
        self._paint(self.assistant_button, show)
        storage.save_setting("assistant", show)

    def _make_room(self, panel_open):
        """With the panel open the window gets wider (as far as the screen allows), so the tabs keep their space;
        when it closes, the window goes back as it was. A maximized window stays as it is."""
        if self.root.state() != "normal":
            return
        self.root.update_idletasks()
        size = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", self.root.geometry())  # "1600x980+10+0"
        if size is None:
            return
        width, height, x, y = map(int, size.groups())
        if panel_open:
            self._size_before_panel = (width, x)
            screen = self.root.winfo_screenwidth()
            width = min(width + WIDTH, max(screen, width))
            x = max(0, min(x, screen - width))
        elif self._size_before_panel:
            width, x = self._size_before_panel
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _language_changed(self):
        """Saves the chosen language: it is applied at the next start, so it offers to restart now."""
        code = self.language.get()
        storage.save_setting("language", code)
        if code != i18n.LANGUAGE and messagebox.askyesno(tr("Language"), tr("Restart now to apply the language?")):
            updater.restart()
            self.close()

    def _pick_changed(self, active):
        self._paint(self.pick_button, active)
        if active:
            self.toggle_assistant(True)  # the explanations of Pick appear in the panel
            self.assistant.write_note(tr("Pick is on: click a control to find out what it does. F1 or the right "
                                         "mouse button to stop."))

    def _unread_changed(self, n):
        self.assistant_button.config(text=tr("Assistant · {n} new", n=n) if n else tr("Assistant (F2)"))
        self._paint(self.assistant_button, self.assistant.visible or n > 0)

    @staticmethod
    def _paint(button, on):
        """Orange when on (Pick active, panel open or hints waiting), grey when off."""
        color, text = (base.ACCENT, base.BACKGROUND) if on else (base.PANEL, base.TEXT)
        button.config(bg=color, fg=text, activebackground=base.mix(color, "#ffffff", 0.2), activeforeground=text)

    def close(self):
        self.training_tab.stop_training()
        self.root.destroy()


def run():
    """Opens the window. Started from start.bat there is no terminal: errors are shown in a small window."""
    root = tk.Tk()
    root.report_callback_exception = lambda *error: messagebox.showerror(
        tr("Error"), "".join(traceback.format_exception(*error)))
    try:
        MainWindow(root)
    except Exception:
        messagebox.showerror(tr("Error at startup"), traceback.format_exc())
        raise
    root.mainloop()

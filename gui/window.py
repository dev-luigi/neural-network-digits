"""
The main window: 5 tabs, one for each step, plus the Info tab.

The tabs "talk" to each other with two notices:
  - photos_changed()  after a download or a reset: all the tabs must be redone
  - model_changed()   after a training: Evaluation, Draw and Inside the network must be redone
To start quickly, each tab draws itself only when you open it (see base.Tab).
Shortly after opening, a background check looks on GitHub for a new version.
"""
import tkinter as tk
import traceback
from tkinter import messagebox, ttk

from gui import base
from gui.tab_data import DataTab
from gui.tab_draw import DrawTab
from gui.tab_evaluation import EvaluationTab
from gui.tab_info import InfoTab
from gui.tab_inside import InsideTab
from gui.tab_training import TrainingTab
from i18n import tr
from neural_net import storage
from neural_net.data import load_photos
from project import VERSION


class MainWindow:
    def __init__(self, root, check_updates=True):
        self.root = root
        root.title(tr("Neural network - digit recognition") + f"  ·  v{VERSION}")
        root.geometry("1600x980+10+0")
        root.minsize(1500, 900)
        base.apply_theme(root)
        self._photos = {}  # photos already read from disk, so they are not read again every time

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
        self.tabs.bind("<<NotebookTabChanged>>", lambda _: self.current_tab().shown())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.current_tab().shown()
        if check_updates and storage.settings()["check_updates"]:
            # after opening, to start quickly
            root.after(1500, lambda: self.info_tab.check_for_updates(silent=True))

    def current_tab(self):
        return self.all_tabs[self.tabs.index("current")]

    def photos(self, split):
        """The "train" or "test" photos, read from disk only once."""
        if split not in self._photos:
            self._photos[split] = load_photos(split)
        return self._photos[split]

    def photos_changed(self):
        self._photos.clear()
        self.refresh(*self.all_tabs)

    def model_changed(self):
        self.refresh(self.evaluation_tab, self.draw_tab, self.inside_tab)

    def refresh(self, *tabs):
        """Marks the tabs to redraw: the open one updates right away, the others when you open them."""
        for tab in tabs:
            tab.needs_update = True
        self.current_tab().shown()

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

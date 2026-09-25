"""
Tab 4 · Draw & edit: the drawing board (drawing_board.py) and the lab (lab.py) together.
"""
import tkinter as tk

from gui import base
from gui.drawing_board import DrawingBoard
from gui.lab import Lab
from i18n import tr
from neural_net.data import load_photos
from neural_net.network import NeuralNetwork
from neural_net.storage import MODEL_FILE


def build(parent, load_test, on_save=None):
    """Puts the drawing board and the lab side by side inside `parent`."""
    board = DrawingBoard(parent, load_test)
    return Lab(parent, board, load_test, on_save)


class DrawTab(base.Tab):
    title = tr("4 · Draw & edit")

    def __init__(self, window):
        super().__init__(window)
        self.lab = None
        self.content = tk.Frame(self.frame, bg=base.BACKGROUND)
        self.notice = tk.Label(self.frame, text=tr("No trained model.\n\nTrain the network in the tab "
                                                   "\"2 · Training\"."), bg=base.BACKGROUND, fg=base.TEXT_SOFT,
                               font=(base.FONT, 13))

    def refresh(self):
        """Loads the last saved model (or shows a notice if there is none)."""
        if not MODEL_FILE.exists():
            self.content.pack_forget()
            return self.notice.pack(pady=160)
        self.notice.pack_forget()
        self.content.pack(anchor="n")
        if self.lab is None:
            self.lab = build(self.content, lambda: self.window.photos("test"),
                             lambda: self.window.refresh(self.window.evaluation_tab, self.window.inside_tab))
        self.lab.set_net(NeuralNetwork.load())


def open_drawing_window():
    """Drawing board and lab in a window of their own (command:  start.bat draw)."""
    root = tk.Tk()
    root.title(tr("Neural network - draw & edit"))
    base.apply_theme(root)
    build(root, lambda: load_photos("test")).set_net(NeuralNetwork.load())
    root.mainloop()

"""
Tab 1 · Data: download the photos, look at them before training (pre-training) and reset.
"""
from tkinter import messagebox, ttk

import numpy as np

from gui import base
from i18n import tr
from neural_net import charts, storage
from neural_net.data import download_mnist, save_photos


class DataTab(base.Tab):
    title = tr("1 · Data")

    def __init__(self, window):
        super().__init__(window)
        c = base.column(self.frame)
        base.title(c, tr("The dataset"))
        base.label(c, tr("28x28 photos of handwritten digits, from the MNIST dataset. Looking at the data before "
                         "training (pre-training) helps to understand what the network will have to learn."))
        self.numbers = base.Tiles(c, ("training photos", "test photos", "black pixels", "average pixel value"),
                                  columns=2, size=12)

        base.section(c, tr("Photos to download for each digit"))
        r = base.row(c)
        self.per_digit = base.NumberField(r, "training", 10, 1000, 100, 10)
        self.test_per_digit = base.NumberField(r, "test", 5, 500, 20, 5)
        self.download_button = base.button(c, tr("Download the photos"), self.download, primary=True)
        self.download_button.pack(fill="x", pady=(12, 6))
        self.progress = ttk.Progressbar(c, maximum=1.0)
        self.progress.pack(fill="x")
        self.status = base.label(c, "", base.TEXT)
        base.label(c, tr("More photos = a more accurate network, but slower training. MNIST is downloaded only once "
                         "(~11 MB): after that the photos can be extracted again even without internet."))

        base.section(c, tr("Start from scratch"))
        base.label(c, tr("Deletes the trained model and the saved charts and, if you want, the photos too. "
                         "It is the same as:  start.bat reset"))
        self.reset_button = base.button(c, tr("Reset project..."), self.reset)
        self.reset_button.pack(fill="x", pady=(10, 0))

        self.fig, self.canvas = base.figure(base.right_area(self.frame))

    def refresh(self):
        """Numbers and charts of the dataset (or a message if there are no photos)."""
        try:
            photos, digits = self.window.photos("train")
            test_photos, test_digits = self.window.photos("test")
        except FileNotFoundError:
            self.numbers.clear()
            self.status.config(text=tr("No photos: press \"Download the photos\"."))
            base.message(self.fig, tr("Download the photos to see the dataset"))
        else:
            self.numbers.show({"training photos": len(photos), "test photos": len(test_photos),
                               "black pixels": f"{np.mean(photos == 0):.0%}",
                               "average pixel value": f"{photos.mean():.0f} / 255"})
            self.status.config(text="")
            charts.exploration(self.fig, photos, digits, test_photos, test_digits)
        self.canvas.draw()

    def _busy(self):
        """While the network is training the photos cannot be changed."""
        if self.window.training_tab.in_progress:
            messagebox.showinfo(tr("Training in progress"), tr("Stop the training first (tab 2)."))
        return self.window.training_tab.in_progress

    def download(self):
        if self._busy():
            return
        per_digit, test_per_digit = self.per_digit.value(), self.test_per_digit.value()
        for widget in (self.download_button, self.reset_button):
            widget.config(state="disabled")
        self.status.config(text=tr("Downloading..."))

        def job(send):  # runs in a separate thread: the window stays free
            mnist = download_mnist(progress=lambda fraction: send("progress", fraction))
            send("saving")
            rng = np.random.default_rng(42)
            save_photos(mnist["x_train"], mnist["y_train"], "train", per_digit, rng)
            save_photos(mnist["x_test"], mnist["y_test"], "test", test_per_digit, rng)

        base.in_background(self.frame, job, self._download_news)

    def _download_news(self, kind, data):
        if kind == "progress":
            self.progress["value"] = data
            self.status.config(text=tr("MNIST download: {fraction:.0%}", fraction=data))
        elif kind == "saving":
            self.progress["value"] = 1
            self.status.config(text=tr("Saving the photos as PNG..."))
        else:  # "done" or "error"
            for widget in (self.download_button, self.reset_button):
                widget.config(state="normal")
            if kind == "error":
                messagebox.showerror(tr("Download failed"), str(data))
            self.window.photos_changed()

    def reset(self):
        if self._busy():
            return
        choice = messagebox.askyesnocancel(
            tr("Reset project"),
            tr("The trained model and the saved charts will be deleted.\n\n"
               "Do you also want to delete the downloaded photos?\n\n"
               "Yes = delete everything      No = keep the photos      Cancel = do nothing"))
        if choice is None:
            return
        storage.reset(include_photos=choice)
        self.progress["value"] = 0
        self.window.photos_changed()

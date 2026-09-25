"""
Tab 1 · Data: download the photos, look at them before training (pre-training) and reset.
"""
from tkinter import messagebox, ttk

import numpy as np

from gui import base
from i18n import tr
from neural_net import charts, storage
from neural_net.data import download_mnist, save_photos
from project import LAUNCHER


class DataTab(base.Tab):
    title = tr("1 · Data")

    def __init__(self, window):
        super().__init__(window)
        c = base.column(self.frame)
        base.title(c, tr("The dataset"))
        base.label(c, tr("28x28 photos of handwritten digits, from the MNIST dataset. Looking at the data before "
                         "training (pre-training) helps to understand what the network will have to learn."))
        self.numbers = base.Tiles(c, ("training photos", "test photos", "black pixels", "average pixel value"),
                                  columns=2, size=12, explanation=tr(
            "The photos downloaded, and how their pixels are: most of them are black (0), the stroke is white "
            "(255)."))

        base.section(c, tr("Photos to download for each digit"))
        r = base.row(c)
        self.per_digit = base.NumberField(r, "training", 10, 1000, 100, 10, explanation=tr(
            "How many photos of each digit the network learns from. More photos = a more accurate network, "
            "but every epoch takes longer."))
        self.test_per_digit = base.NumberField(r, "test", 5, 500, 20, 5, explanation=tr(
            "How many photos of each digit are kept aside to test the network: it never learns from them, so "
            "they tell how it does with photos it has never seen."))
        self.download_button = base.button(c, tr("Download the photos"), self.download, primary=True,
                                           explanation=tr(
            "Downloads MNIST (only the first time) and saves the chosen photos as PNG in data/photos/, in place "
            "of the ones already there."))
        self.download_button.pack(fill="x", pady=(12, 6))
        self.progress = ttk.Progressbar(c, maximum=1.0)
        self.progress.pack(fill="x")
        self.status = base.label(c, "", base.TEXT)
        base.label(c, tr("More photos = a more accurate network, but slower training. MNIST is downloaded only once "
                         "(~11 MB): after that the photos can be extracted again even without internet."))

        base.section(c, tr("Start from scratch"))
        base.label(c, tr("Deletes the trained model and the saved charts and, if you want, the photos too. "
                         "It is the same as:  {command}", command=f"{LAUNCHER} reset"))
        self.reset_button = base.button(c, tr("Reset project..."), self.reset, explanation=tr(
            "Before deleting anything it asks whether to delete the photos too. The code and the settings are "
            "never touched."))
        self.reset_button.pack(fill="x", pady=(10, 0))

        right = base.right_area(self.frame)
        base.explanation_box(right, self.frame, tr("Move the mouse over a control, a value or a chart "
                                                   "to find out what it means."), 1220).pack(fill="x", pady=(0, 6))
        self.fig, self.canvas = base.figure(right)
        base.explain(self.canvas.get_tk_widget(), tr(
            "The dataset before training: some photos of each digit, how many photos there are per digit, the "
            "\"average digit\" (all the photos of a digit on top of each other) and how the pixel values are "
            "spread out."))

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

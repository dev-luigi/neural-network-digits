"""
Tab 1 · Data: download the photos, look at them before training (pre-training) and reset.
"""
import os
from tkinter import messagebox, ttk

import numpy as np

from nn_digits.gui import base
from nn_digits.i18n import tr
from nn_digits.neural_net import charts, storage
from nn_digits.neural_net.data import MNIST_PHOTOS, collection_size, download_mnist, save_collection
from nn_digits.project import LAUNCHER


class DataTab(base.Tab):
    title = tr("1 · Data")

    def __init__(self, window):
        super().__init__(window)
        c = base.column(self.frame)
        base.title(c, tr("The dataset"))
        base.label(c, tr("28x28 photos of handwritten digits, from the MNIST dataset. Looking at the data before "
                         "training (pre-training) helps to understand what the network will have to learn."))
        self.numbers = base.Tiles(c, ("training photos", "test photos", "black pixels", "average pixel value"),
                                  columns=2, size=12, name=tr("The photos in numbers"), explanation=tr(
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
            "Downloads MNIST (only the first time) and saves the chosen photos as PNG in {folder}, in place of "
            "the ones already there.", folder=storage.PHOTOS_DIR))
        self.download_button.pack(fill="x", pady=(12, 6))
        self.all_button = base.button(c, tr("Download the whole collection..."), self.download_all, explanation=tr(
            "All the photos of MNIST: 60000 for training and 10000 for test. Before starting it tells you how much "
            "space they take."))
        self.all_button.pack(fill="x", pady=(0, 6))
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
            "spread out."), tr("Dataset charts"))
        base.explain_charts(self.canvas, {
            "samples": (tr("Sample photos"), tr(
                "Ten photos of each digit, one row per digit, taken from the training photos: the same digit is "
                "written in many different ways, and the network has to recognize all of them.")),
            "per digit": (tr("Photos per digit"), tr(
                "How many photos there are of each digit: blue the training ones, orange the test ones. The digits "
                "are more or less equally many, so the network does not learn to prefer one of them.")),
            "average digit": (tr("The average digit"), tr(
                "All the photos of a digit laid on top of each other: the bright part is where almost everybody "
                "draws the stroke, the dark part where nobody does. It is a first rough idea of what the network "
                "has to learn.")),
            "pixel values": (tr("Pixel values"), tr(
                "How many pixels have each value, on a logarithmic scale: almost all are black (0) and the stroke "
                "is close to 255. Before entering the network the values are divided by 255, so they go from 0 "
                "to 1."))})

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
        self._download(self.per_digit.value(), self.test_per_digit.value())

    def download_all(self):
        """The whole collection: first a window says how much space it takes."""
        if self._busy():
            return
        try:
            # the disk block (Linux, macOS), asked to the first folder that exists on the way to the data
            existing = next(folder for folder in (storage.DATA_DIR, *storage.DATA_DIR.parents) if folder.exists())
            block = os.statvfs(existing).f_frsize
        except AttributeError:  # Windows: NTFS keeps the small files inside its table of files, without blocks
            block = None
        megabytes = {name: size / 1e6 for name, size in collection_size(block).items()}
        text = tr("All the photos of MNIST: {photos} ({train} for training and {test} for test).\n\n"
                  "Total size: about {total:.0f} MB, in {folder}\n"
                  "  •  the MNIST file: {mnist:.0f} MB ({state})\n"
                  "  •  the photos as PNG, with their quick copy: {png:.0f} MB\n\n",
                  photos=sum(MNIST_PHOTOS.values()), train=MNIST_PHOTOS["train"], test=MNIST_PHOTOS["test"],
                  total=megabytes["mnist"] + megabytes["photos"], mnist=megabytes["mnist"], png=megabytes["photos"],
                  state=tr("already downloaded") if storage.MNIST_FILE.exists() else tr("to download"),
                  folder=storage.PHOTOS_DIR)
        if megabytes["disk"] > 1.5 * (megabytes["mnist"] + megabytes["photos"]):
            text += tr("On this disk they take about {disk:.0f} MB, because each of the 70000 small files takes at "
                       "least {block} KB.\n\n", disk=megabytes["disk"], block=block // 1024)
        text += tr("Saving them takes about a minute, and the training becomes much slower: every epoch takes "
                   "seconds instead of a fraction of a second.\n\nDownload the whole collection?")
        if messagebox.askyesno(tr("Download the whole collection"), text):
            self._download(None, None)

    def _download(self, per_digit, test_per_digit):
        """Downloads MNIST and saves `per_digit` photos of each digit (None = all of them)."""
        if self._busy():
            return
        for widget in (self.download_button, self.all_button, self.reset_button):
            widget.config(state="disabled")
        self.status.config(text=tr("Downloading..."))

        def job(send):  # runs in a separate thread: the window stays free
            mnist = download_mnist(progress=lambda fraction: send("progress", fraction))
            save_collection(mnist, per_digit, test_per_digit, progress=lambda fraction: send("saving", fraction))

        base.in_background(self.frame, job, self._download_news)

    def _download_news(self, kind, data):
        if kind == "progress":
            self.progress["value"] = data
            self.status.config(text=tr("MNIST download: {fraction:.0%}", fraction=data))
        elif kind == "saving":
            self.progress["value"] = data
            self.status.config(text=tr("Saving the photos as PNG: {fraction:.0%}", fraction=data))
        else:  # "done" or "error"
            for widget in (self.download_button, self.all_button, self.reset_button):
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

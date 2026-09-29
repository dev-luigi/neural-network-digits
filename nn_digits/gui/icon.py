"""
The icon of the window and of the taskbar: a tiny neural network (3 columns of dots joined by lines), drawn with
Pillow, so there is no image file to ship. On Windows the program also gets its own identity: otherwise the
taskbar shows the generic Python icon and groups the window with every other Python program.
"""
import ctypes
import sys
import tkinter as tk

from PIL import Image, ImageDraw, ImageTk

from nn_digits.gui import base

APP_ID = "dev-luigi.neural-network-digits"
LAYERS = (2, 3, 2)      # dots in each column
SIZES = (256, 64, 32, 16)


def draw_icon(size):
    """The icon as a square image `size` pixels wide (drawn 4 times bigger, then shrunk to look smooth)."""
    big = size * 4
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=big // 5, fill=base.BACKGROUND)

    margin = big * 0.22
    columns = [margin + (big - 2 * margin) * i / (len(LAYERS) - 1) for i in range(len(LAYERS))]
    dots = []
    for x, count in zip(columns, LAYERS):
        dots.append([(x, big / 2 + big * 0.26 * (row - (count - 1) / 2)) for row in range(count)])

    for left, right in zip(dots, dots[1:]):
        for a in left:
            for b in right:
                draw.line((*a, *b), fill=base.TEXT_SOFT, width=max(big // 40, 1))
    radius = big * 0.075
    for column, dots_in_column in enumerate(dots):
        color = base.ACCENT if column == len(dots) - 1 else base.TEXT
        for x, y in dots_in_column:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image.resize((size, size), Image.LANCZOS)


def apply_app_id():
    """On Windows, tells the system this is its own program, not python.exe. To be called before the window exists."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except (AttributeError, OSError):
            pass


def apply(root):
    """Gives the window its icon."""
    try:
        images = [ImageTk.PhotoImage(draw_icon(size), master=root) for size in SIZES]
        root.iconphoto(True, *images)
        root._icon_images = images  # Tk does not keep them alive by itself
    except tk.TclError:
        pass  # no icon is not worth a crash

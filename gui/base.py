"""
Building blocks shared by all the tabs of the interface:
  - the colors of the dark theme
  - ready-made interface pieces: labels, buttons, sliders, number boxes, tiles...
  - the explanation box, which describes the control under the mouse (the assistant reuses the explanations)
  - the base class of the tabs and a helper for long jobs (without freezing the window)
"""
import queue
import threading
import tkinter as tk
import webbrowser
from collections import namedtuple
from tkinter import ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import ImageDraw

from i18n import tr
from neural_net import charts

# ---------------------------------------------------------------- colors (dark theme)
BACKGROUND = "#14171c"
PANEL = "#1d2129"
BORDER = "#3a3f48"
OFF = "#2b313b"         # neuron switched off
TEXT = "#e6e6e6"
TEXT_SOFT = "#8b93a1"
ACCENT = "#ffb347"      # orange: active, positive, chosen value
BLUE = "#4da3ff"        # blue: negative
GREEN = "#5fd39a"
RED = "#e05d6f"
FONT = "Segoe UI"
CONTROLS_WIDTH = 300
PICK = "Pick"   # the binding tag of the controls with an explanation (see gui/pick.py)
EXPLAINED = []  # the controls with an explanation: Pick can choose them, the assistant can search them
# A piece of a control that Pick chooses on its own: one chart of a figure, one tile. value() = what it shows now
# (or None), box = (x, y, width, height) in pixels of the control.
Part = namedtuple("Part", "name text value box")


def power(v):
    """10 to the power of v, rounded to 2 digits (for "logarithmic" sliders: the value used is the one shown)."""
    return float(f"{10 ** v:.2g}")


def mix(color1, color2, t):
    """The color halfway between: t=0 gives color1, t=1 gives color2."""
    t = min(max(float(t), 0.0), 1.0)
    rgb1, rgb2 = _rgb(color1), _rgb(color2)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(rgb1, rgb2))


def _rgb(color):
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


def color_map(values):
    """Colors a table of numbers: positive in orange, negative in blue, zero = the panel color.
    Returns an RGB image (array of integers 0-255) to show with PIL."""
    t = np.clip(np.abs(values) / (np.abs(values).max() + 1e-9), 0, 1)[..., None]
    color = np.where(values[..., None] > 0, _rgb(ACCENT), _rgb(BLUE))
    return (_rgb(PANEL) + (color - _rgb(PANEL)) * t).astype(np.uint8)


def apply_theme(root):
    """Dark colors for the window, the tabs and the charts."""
    root.configure(bg=BACKGROUND)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BACKGROUND, foreground=TEXT, bordercolor=BACKGROUND, lightcolor=BACKGROUND,
                    darkcolor=BACKGROUND, focuscolor=BACKGROUND)
    style.configure("TNotebook", borderwidth=0, tabmargins=(16, 10, 16, 0))
    style.configure("TNotebook.Tab", background=PANEL, foreground=TEXT_SOFT, padding=(18, 8),
                    font=(FONT, 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", BORDER), ("active", OFF)], foreground=[("selected", TEXT)])
    style.configure("TProgressbar", background=ACCENT, troughcolor=PANEL, bordercolor=PANEL,
                    lightcolor=ACCENT, darkcolor=ACCENT, thickness=8)
    style.configure("Vertical.TScrollbar", background=BORDER, troughcolor=PANEL, bordercolor=PANEL,
                    lightcolor=BORDER, darkcolor=BORDER, arrowcolor=TEXT_SOFT)
    matplotlib.rcParams.update({
        "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "axes.facecolor": PANEL, "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_SOFT, "axes.titlecolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT_SOFT, "ytick.color": TEXT_SOFT, "grid.color": BORDER,
        "legend.facecolor": PANEL, "legend.edgecolor": BORDER, "legend.labelcolor": TEXT,
        "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ACCENT, GREEN, RED, "#b18cff"]),
    })


# ---------------------------------------------------------------- explanations

def explanation_box(parent, container, text, width, lines=3):
    """The box where the explanation of the control under the mouse appears, for all the controls
    inside `container`. Returns the label: it must then be placed with pack or grid."""
    box = tk.Label(parent, text=text, bg=PANEL, fg=TEXT_SOFT, font=(FONT, 9), justify="left",
                   anchor="nw", wraplength=width - 24, height=lines, padx=12, pady=6)
    # If the window gets narrower (for example when the assistant opens) the text goes to a new line earlier
    box.bind("<Configure>", lambda event: box.config(wraplength=min(width, event.width) - 24), add="+")
    container.explanations = box
    return box


def explain(widget, text, name="", value=None):
    """When the mouse goes over the widget (or one of its pieces), the explanation box shows `text`.
    name = the name of the control and value() = what it is worth now: the assistant shows them with Pick."""
    if not text:
        return
    widget.pick_text, widget.pick_name, widget.pick_value = text, name, value
    EXPLAINED.append(widget)

    def show(_event):
        container = widget
        while container is not None and not hasattr(container, "explanations"):
            container = container.master  # go up until I find who has the box
        if container is not None:
            container.explanations.config(text=text)

    to_bind = [widget]
    while to_bind:
        w = to_bind.pop()
        w.bind("<Enter>", show, add="+")
        if PICK not in w.bindtags():  # first of all the tags: while Pick is on, the clicks stop there
            w.bindtags((PICK,) + w.bindtags())
        to_bind.extend(w.winfo_children())


def explain_charts(canvas, charts):
    """Pick chooses the single charts of a matplotlib figure, not the whole figure.
    charts = {gid: (name, text)} or {gid: (name, text, value)}: the gid is the one that neural_net/charts.py gives
    to each chart (ax.set_gid) and value(ax) says what that chart shows now. If the figure has none of these charts
    (for example it only writes a message) Pick takes the whole figure, with the explanation of explain()."""
    widget = canvas.get_tk_widget()
    places = []  # [(chart, its plot area, the area with title and labels too)] in pixels of the figure

    def forget(_event):  # after every drawing the charts may have moved
        places.clear()

    def measure():
        renderer = canvas.get_renderer()
        for ax in canvas.figure.axes:
            if ax.get_gid() in charts and ax.get_visible():
                places.append((ax, ax.bbox.frozen(), ax.get_tightbbox(renderer)))

    def part(x, y):
        """The chart at the point (x, y) of the widget, None if the point is between two charts."""
        if not places:
            measure()
        width, height = max(widget.winfo_width(), 1), max(widget.winfo_height(), 1)
        if not places:
            return Part(widget.pick_name, widget.pick_text, widget.pick_value, (0, 0, width, height))
        # The figure has y going up and may be drawn with more pixels than the widget (screens with zoom)
        scale = canvas.figure.bbox.width / width
        x, y = x * scale, canvas.figure.bbox.height - y * scale
        # First the plot areas, which never overlap; then also titles and labels
        found = (next((p for p in places if p[1].contains(x, y)), None)
                 or next((p for p in places if p[2].contains(x, y)), None))
        if found is None:
            return None
        ax, _, box = found
        name, text, value = (charts[ax.get_gid()] + (None,))[:3]
        left, top = max(box.x0 / scale, 0), max((canvas.figure.bbox.height - box.y1) / scale, 0)
        right, bottom = min(box.x1 / scale, width), min((canvas.figure.bbox.height - box.y0) / scale, height)
        return Part(name, text, value and (lambda: value(ax)),
                    (round(left), round(top), round(right - left), round(bottom - top)))

    canvas.mpl_connect("draw_event", forget)
    widget.pick_parts = part
    widget.pick_entries = [(name, text) for name, text, *_ in charts.values()]


def chart_title(ax):
    """The title of a chart on one line (value for explain_charts)."""
    return " ".join(ax.get_title().split())


def chart_legend(ax):
    """The texts of the legend of a chart (value for explain_charts)."""
    legend = ax.get_legend()
    return "   ".join(text.get_text() for text in legend.get_texts()) if legend else None


def explained_controls():
    """The controls with an explanation that still exist (some are recreated when the network changes)."""
    EXPLAINED[:] = [w for w in EXPLAINED if _exists(w)]
    return list(EXPLAINED)


def _exists(widget):
    try:
        return bool(widget.winfo_exists())
    except tk.TclError:  # its whole window has been closed
        return False


def control_name(widget):
    """The name of a control with an explanation: the one it was given, or the text written on it."""
    try:
        return widget.pick_name or widget.cget("text")
    except tk.TclError:  # widgets without text, like a frame or a canvas
        return widget.pick_name


# ---------------------------------------------------------------- interface pieces

def column(parent):
    """The column on the left of a tab, where the controls are."""
    frame = tk.Frame(parent, bg=BACKGROUND, width=CONTROLS_WIDTH)
    frame.pack(side="left", fill="y", padx=(20, 8), pady=16)
    frame.pack_propagate(False)  # fixed width, even if the content is narrow
    return frame


def right_area(parent):
    """The space on the right of the controls, for charts and values."""
    area = tk.Frame(parent, bg=BACKGROUND)
    area.pack(side="left", fill="both", expand=True, padx=(8, 20), pady=16)
    return area


def row(parent, pady=0):
    """A container to put several things next to each other."""
    frame = tk.Frame(parent, bg=BACKGROUND)
    frame.pack(fill="x", pady=pady)
    return frame


def label(parent, text="", color=TEXT_SOFT, size=9, bold=False, pady=(2, 0), width=CONTROLS_WIDTH - 10):
    """A piece of text: if it is long it wraps by itself."""
    widget = tk.Label(parent, text=text, bg=BACKGROUND, fg=color, justify="left", wraplength=width,
                      font=(FONT, size, "bold" if bold else "normal"))
    widget.pack(anchor="w", pady=pady)
    return widget


def title(parent, text, pady=(0, 8)):
    return label(parent, text, TEXT, 12, True, pady)


def section(parent, text):
    return label(parent, text, ACCENT, 9, True, (14, 0))


def link(parent, text, url, size=10):
    """Clickable blue text that opens `url` in the browser (underlined under the mouse)."""
    widget = tk.Label(parent, text=text, bg=BACKGROUND, fg=BLUE, cursor="hand2", font=(FONT, size))
    widget.bind("<Button-1>", lambda _: webbrowser.open(url))
    widget.bind("<Enter>", lambda _: widget.config(font=(FONT, size, "underline")), add="+")
    widget.bind("<Leave>", lambda _: widget.config(font=(FONT, size)), add="+")
    return widget


def button(parent, text, command, primary=False, explanation=""):
    color = ACCENT if primary else PANEL
    widget = tk.Button(parent, text=text, command=command, bg=color, fg=BACKGROUND if primary else TEXT,
                       activebackground=mix(color, "#ffffff", 0.2),
                       activeforeground=BACKGROUND if primary else TEXT, disabledforeground=TEXT_SOFT,
                       relief="flat", bd=0, padx=10, pady=5, cursor="hand2",
                       font=(FONT, 9, "bold" if primary else "normal"))
    explain(widget, explanation)
    return widget


def slider(parent, text, from_, to, step, value, on_change=None, show=None, explanation=""):
    """A slider control: name and current value on top, slider below.
    on_change(value) is called at every move; show(value) decides how to write the value."""
    frame = tk.Frame(parent, bg=BACKGROUND)
    frame.pack(fill="x", pady=(6, 0))
    frame.columnconfigure(1, weight=1)
    tk.Label(frame, text=text, bg=BACKGROUND, fg=TEXT, font=(FONT, 9)).grid(row=0, column=0, sticky="w")
    value_label = tk.Label(frame, bg=BACKGROUND, fg=ACCENT, font=(FONT, 9, "bold"))
    value_label.grid(row=0, column=1, sticky="e")

    def changed(v):
        v = float(v)
        value_label.config(text=show(v) if show else f"{v:g}")
        if on_change:
            on_change(v)

    scale = tk.Scale(frame, from_=from_, to=to, resolution=step, orient="horizontal", showvalue=False,
                     command=changed, bg="#aab2bf", troughcolor=PANEL, activebackground=ACCENT,
                     highlightthickness=0, bd=0, sliderrelief="flat", sliderlength=14, width=12)
    scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
    scale.set(value)
    # Tk calls `command` only when the value changes: the first time I call it myself
    frame.after_idle(lambda: changed(scale.get()))
    explain(frame, explanation, text, lambda: value_label.cget("text"))
    return scale


class NumberField:
    """Number box with the up/down arrows; .value() always returns a valid integer."""

    def __init__(self, parent, text, from_, to, value, increment=1, explanation=""):
        self.from_, self.to, self.default = from_, to, value
        cell = tk.Frame(parent, bg=BACKGROUND)
        cell.pack(side="left", padx=(0, 12), pady=(6, 0))
        tk.Label(cell, text=text, bg=BACKGROUND, fg=TEXT_SOFT, font=(FONT, 9)).pack(anchor="w")
        self.var = tk.StringVar()
        tk.Spinbox(cell, from_=from_, to=to, increment=increment, textvariable=self.var, width=6, justify="center",
                   bg=PANEL, fg=TEXT, buttonbackground=PANEL, insertbackground=TEXT, relief="flat",
                   font=(FONT, 10), highlightthickness=1, highlightbackground=BORDER,
                   highlightcolor=ACCENT).pack(anchor="w", pady=(2, 0))
        self.var.set(str(value))  # after creating it: otherwise the Spinbox overwrites it
        explain(cell, explanation, text, self.var.get)

    def value(self):
        try:
            return min(max(int(float(self.var.get())), self.from_), self.to)
        except ValueError:
            return self.default


class Choice:
    """Drop-down menu; .value() returns the chosen option."""

    def __init__(self, parent, text, options, value, explanation=""):
        frame = row(parent, pady=(6, 0))
        tk.Label(frame, text=text, bg=BACKGROUND, fg=TEXT, font=(FONT, 9)).pack(side="left")
        self.var = tk.StringVar(value=value)
        menu = tk.OptionMenu(frame, self.var, *options)
        menu.config(bg=PANEL, fg=TEXT, activebackground=BORDER, activeforeground=TEXT, highlightthickness=0,
                    relief="flat", bd=0, font=(FONT, 9), width=10)
        menu["menu"].config(bg=PANEL, fg=TEXT, activebackground=ACCENT, activeforeground=BACKGROUND)
        menu.pack(side="right")
        explain(frame, explanation, text, self.var.get)

    def value(self):
        return self.var.get()


def choice_buttons(parent, text, options, variable, on_change, explanation=""):
    """A row of buttons where only one stays pressed (in orange), like a switch with several
    positions. options = [(label, value), ...]; `variable` holds the chosen value."""
    frame = tk.Frame(parent, bg=BACKGROUND)
    frame.pack(side="left", padx=(0, 22))
    tk.Label(frame, text=text, bg=BACKGROUND, fg=TEXT_SOFT, font=(FONT, 9)).pack(side="left", padx=(0, 6))
    buttons = {}
    for option_label, value in options:
        buttons[value] = tk.Radiobutton(frame, text=option_label, value=value, variable=variable, command=on_change,
                                        indicatoron=False, bg=PANEL, selectcolor=ACCENT, activebackground=BORDER,
                                        relief="flat", offrelief="flat", bd=0, padx=10, pady=4, font=(FONT, 9),
                                        cursor="hand2")
        buttons[value].pack(side="left", padx=1)

    def paint(*_):  # dark text on the orange button, light on the others
        for value, widget in buttons.items():
            chosen = str(value) == str(variable.get())
            widget.config(fg=BACKGROUND if chosen else TEXT, activeforeground=BACKGROUND if chosen else TEXT)

    variable.trace_add("write", paint)
    paint()
    explain(frame, explanation, text, lambda: next((option_label for option_label, value in options
                                                    if str(value) == str(variable.get())), ""))
    return frame


def checkbox(parent, text, on_change, explanation=""):
    """A box to tick. Returns the variable (true/false)."""
    var = tk.BooleanVar()
    box = tk.Checkbutton(parent, text=text, variable=var, command=on_change, bg=BACKGROUND, fg=TEXT,
                         selectcolor=PANEL, activebackground=BACKGROUND, activeforeground=TEXT,
                         font=(FONT, 9), highlightthickness=0, bd=0, cursor="hand2")
    box.pack(anchor="w", pady=(6, 0))
    explain(box, explanation, text, lambda: tr("on") if var.get() else tr("off"))
    return var


class Tiles:
    """Tiles with a small name and a big value: the "dashboard" of the important numbers.
    The names are English keys: each tile shows the name translated with tr()."""

    def __init__(self, parent, names, columns=None, size=13, explanation="", name=""):
        """name = what the tiles are as a whole (the assistant shows it with Pick)."""
        self.frame = tk.Frame(parent, bg=BACKGROUND)
        self.frame.pack(fill="x", pady=(8, 0))
        columns = columns or len(names)
        self.values = {}
        for k, tile_name in enumerate(names):  # not "name": it would cover the name of the tiles as a whole
            tile = tk.Frame(self.frame, bg=PANEL, padx=10, pady=4)
            tile.grid(row=k // columns, column=k % columns, sticky="nsew", padx=(0, 6), pady=(0, 6))
            tk.Label(tile, text=tr(tile_name), bg=PANEL, fg=TEXT_SOFT, font=(FONT, 8)).pack(anchor="w")
            self.values[tile_name] = tk.Label(tile, text="-", bg=PANEL, fg=TEXT, font=(FONT, size, "bold"))
            self.values[tile_name].pack(anchor="w")
        for c in range(columns):
            self.frame.columnconfigure(c, weight=1, uniform="tiles")
        explain(self.frame, explanation, name or " · ".join(tr(n) for n in names[:3]),
                lambda: "   ".join(f"{tr(n)}: {value.cget('text')}" for n, value in self.values.items()))
        self.frame.pick_parts = self._tile_at  # Pick chooses one tile at a time
        self.explanation = explanation

    def _tile_at(self, x, y):
        """The tile at the point (x, y) of the tiles, as a Part for Pick (None between two tiles)."""
        for tile_name, value in self.values.items():
            tile = value.master
            box = (tile.winfo_x(), tile.winfo_y(), tile.winfo_width(), tile.winfo_height())
            if box[0] <= x < box[0] + box[2] and box[1] <= y < box[1] + box[3]:
                title = tr(tile_name)
                return Part(title[:1].upper() + title[1:], self.explanation, lambda label=value: label.cget("text"),
                            box)
        return None

    def show(self, values):
        """values = {name: text}. The tiles not named stay as they are."""
        for name, value in values.items():
            self.values[name].config(text=value)

    def clear(self):
        self.show({name: "-" for name in self.values})


class Bars:
    """Ten horizontal bars, one per digit, with the percentage on the right."""

    def __init__(self, parent, length=150, height=30):
        self.length, self.height = length, height
        self.canvas = tk.Canvas(parent, width=length + 76, height=10 * height, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(anchor="w", pady=(4, 0))
        self.bars = []
        for digit in range(10):
            y = digit * height + height / 2
            self.canvas.create_text(8, y, text=str(digit), fill=TEXT, font=(FONT, 11, "bold"))
            self.canvas.create_rectangle(24, y - 8, 24 + length, y + 8, fill=PANEL, outline="")
            bar = self.canvas.create_rectangle(24, y - 8, 24, y + 8, fill=ACCENT, outline="")
            text = self.canvas.create_text(32 + length, y, anchor="w", fill=TEXT_SOFT, font=(FONT, 9))
            self.bars.append((bar, text, y))

    def show(self, values, colors):
        """values: 10 numbers between 0 and 1 (None = empty bar); colors: the color of each bar.
        All empty: the bars are crossed out."""
        for (bar, text, y), value, color in zip(self.bars, values, colors):
            self.canvas.coords(bar, 24, y - 8, 24 + self.length * (value or 0), y + 8)
            self.canvas.itemconfig(bar, fill=color)
            self.canvas.itemconfig(text, text="" if value is None else f"{value:.0%}")
        self.canvas.delete("empty")
        if all(value is None for value in values):
            cross_out(self.canvas, 24, self.height / 2 - 8, 24 + self.length, 9.5 * self.height + 8)


def figure(parent):
    """A matplotlib figure inside the window; returns (figure, canvas to redraw)."""
    fig = Figure(figsize=(9, 7), dpi=100, layout="constrained")
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().configure(bg=BACKGROUND, highlightthickness=0)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return fig, canvas


def message(fig, text):
    """Empties the figure and writes only a message in the middle, on a thin X: there is nothing to show."""
    fig.clear()
    charts.crossed_out(fig, text, color=TEXT_SOFT, fontsize=13)


def cross_out(canvas, x0, y0, x1, y1, frame=False):
    """The empty state of a chart drawn in Tk, like charts.crossed_out(): a thin X from corner to corner of the
    rectangle (and its edge, if frame=True). Everything has the tag "empty", to delete it when there is data."""
    for line in ((x0, y0, x1, y1), (x0, y1, x1, y0)):
        canvas.create_line(*line, fill=BORDER, tags="empty")
    if frame:
        canvas.create_rectangle(x0, y0, x1, y1, outline=BORDER, tags="empty")


def crossed_out_image(image):
    """The same X on an image (PIL) that has nothing to show. Returns a crossed out copy, in color (in a grey image
    the X would lose its color)."""
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for line in ((0, 0, width - 1, height - 1), (0, height - 1, width - 1, 0)):
        draw.line(line, fill=BORDER)
    return image


# ---------------------------------------------------------------- tabs and long jobs

class Tab:
    """Common base of the tabs. To start quickly, a tab redraws itself only when
    you open it, and only if something changed in the meantime (needs_update = True)."""
    title = ""

    def __init__(self, window):
        self.window = window
        self.frame = tk.Frame(window.tabs, bg=BACKGROUND)
        window.tabs.add(self.frame, text=f"  {self.title}  ")
        self.needs_update = True

    def shown(self):
        if self.needs_update:
            self.needs_update = False
            self.refresh()

    def refresh(self):
        """Each tab redefines it: it redraws its content."""


def in_background(widget, job, on_news):
    """Runs job(send) in a separate thread, so the window does not freeze.

    The thread must NOT touch the window (Tkinter does not allow it): it sends news with
    send(kind, data). Every 100 ms the window collects them and calls on_news(kind, data).
    At the end ("done", None) always arrives, or ("error", exception) if something goes wrong.
    """
    news = queue.Queue()

    def in_thread():
        try:
            job(lambda kind, data=None: news.put((kind, data)))
            news.put(("done", None))
        except Exception as error:
            news.put(("error", error))

    def check():
        while not news.empty():
            kind, data = news.get()
            on_news(kind, data)
            if kind in ("done", "error"):
                return
        widget.after(100, check)

    threading.Thread(target=in_thread, daemon=True).start()
    check()

"""
Pick: press F1 (or the Pick button), move the mouse over the controls and click one: the assistant explains
what it does and what it is worth now. While Pick is on, the clicks do not reach the controls, so nothing
starts by mistake. F1 again, or the right mouse button, turns it off.

How it works: base.explain() gives every control that has an explanation (and all its pieces) an extra
"binding tag", base.PICK, in front of the others. Tk runs the bindings of the tags in order: while Pick is on,
the bindings of PICK answer first and return "break", so the bindings of the control never run.
The orange frame around the control under the mouse is made of 4 thin rectangles placed on top of the window.

Some controls are made of pieces that Pick chooses one by one: the charts of a figure (base.explain_charts)
and the tiles of the numbers (base.Tiles). They have pick_parts(x, y), which says which piece is at that point:
the frame goes around that piece only, and following the mouse it jumps from one chart to the next.
"""
import tkinter as tk

from gui import base

THICKNESS = 2


class Picker:
    def __init__(self, root, on_pick, on_toggle=None):
        """on_pick(control, part) is called when a control is clicked (part = the chart or the tile clicked, None
        for the controls made of one piece); on_toggle(active) when Pick turns on or off."""
        self.root, self.on_pick, self.on_toggle = root, on_pick, on_toggle
        self.active = False
        self.target = self.part = None  # the control framed right now, and which of its pieces
        self.borders = [tk.Frame(root, bg=base.ACCENT) for _ in range(4)]
        for border in self.borders:  # leaving the frame is like leaving the control
            border.bind("<Leave>", self._leave)
        for sequence, handler in (("<Enter>", self._enter), ("<Motion>", self._motion), ("<Leave>", self._leave),
                                  ("<Button-1>", self._click), ("<Button-3>", self._cancel),
                                  ("<Button-2>", self._swallow), ("<ButtonRelease-1>", self._swallow),
                                  ("<ButtonRelease-2>", self._swallow), ("<ButtonRelease-3>", self._swallow),
                                  ("<B1-Motion>", self._swallow), ("<Double-Button-1>", self._swallow)):
            root.bind_class(base.PICK, sequence, handler)

    def set(self, active):
        self.active = active
        if not active:
            self.hide()
        if self.on_toggle:
            self.on_toggle(active)

    def toggle(self):
        self.set(not self.active)

    # ------------------------------------------------------------------ the mouse

    def _control(self, widget):
        """The control with an explanation that contains this widget (itself, or the closest container)."""
        if isinstance(widget, str):  # some events carry only the name of the widget
            try:
                widget = self.root.nametowidget(widget)
            except KeyError:
                return None
        while widget is not None and not hasattr(widget, "pick_text"):
            widget = widget.master
        if widget is not None and widget.winfo_toplevel() is self.root:
            return widget
        return None

    def _under(self, widget, x_root, y_root):
        """(control, part) at a point of the screen: part = the chart or the tile at that point, for the controls
        made of pieces (None for the others). Between two pieces there is nothing to choose: (None, None)."""
        control = self._control(widget)
        parts = getattr(control, "pick_parts", None)
        if parts is None:
            return control, None
        part = parts(x_root - control.winfo_rootx(), y_root - control.winfo_rooty())
        return (control, part) if part is not None else (None, None)

    def _enter(self, event):
        if self.active:
            self.show(*self._under(event.widget, event.x_root, event.y_root))

    def _motion(self, event):
        """Over a figure the mouse goes from one chart to the next without leaving the control.
        No "break": the charts keep answering to the mouse (the photo of a point, the cells of the matrix)."""
        if not self.active or not hasattr(self._control(event.widget), "pick_parts"):
            return
        control, part = self._under(event.widget, event.x_root, event.y_root)
        if control is not self.target or (part and part.box) != (self.part and self.part.box):
            self.show(control, part)

    def _leave(self, _event):
        if self.active:  # the mouse may have gone onto another control: I check where it is a moment later
            self.root.after(40, self._check_pointer)

    def _check_pointer(self):
        if not self.active:
            return
        x, y = self.root.winfo_pointerxy()
        try:
            under = self.root.winfo_containing(x, y)
        except KeyError:  # a piece of a widget made directly in Tk, unknown to Python
            under = None
        if under in self.borders:  # the mouse is on the orange frame itself: nothing changes
            return
        if under is None:
            return self.hide()
        self.show(*self._under(under, x, y))

    def _click(self, event):
        if not self.active:
            return None
        control, part = self._under(event.widget, event.x_root, event.y_root)
        if control is not None:
            self.show(control, part)
            self.on_pick(control, part)
        return "break"

    def _cancel(self, _event):
        if not self.active:
            return None
        self.set(False)
        return "break"

    def _swallow(self, _event):
        return "break" if self.active else None

    # ------------------------------------------------------------------ the orange frame

    def show(self, control, part=None):
        """Frames the control, or only its piece `part` (None = no frame)."""
        if control is None or not control.winfo_viewable():
            return self.hide()
        self.target, self.part = control, part
        x = control.winfo_rootx() - self.root.winfo_rootx()
        y = control.winfo_rooty() - self.root.winfo_rooty()
        width, height, t = control.winfo_width(), control.winfo_height(), THICKNESS
        if part is not None:
            x, y, width, height = x + part.box[0], y + part.box[1], part.box[2], part.box[3]
        # Just outside the control: the mouse on the control is never on the frame
        places = ((x - 2 * t, y - 2 * t, width + 4 * t, t), (x - 2 * t, y + height + t, width + 4 * t, t),
                  (x - 2 * t, y - 2 * t, t, height + 4 * t), (x + width + t, y - 2 * t, t, height + 4 * t))
        for border, (bx, by, bw, bh) in zip(self.borders, places):
            border.place(x=bx, y=by, width=bw, height=bh)
            border.lift()

    def hide(self):
        self.target = self.part = None
        for border in self.borders:
            border.place_forget()

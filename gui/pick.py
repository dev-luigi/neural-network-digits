"""
Pick: press F1 (or the Pick button), move the mouse over the controls and click one: the assistant explains
what it does and what it is worth now. While Pick is on, the clicks do not reach the controls, so nothing
starts by mistake. F1 again, or the right mouse button, turns it off.

How it works: base.explain() gives every control that has an explanation (and all its pieces) an extra
"binding tag", base.PICK, in front of the others. Tk runs the bindings of the tags in order: while Pick is on,
the bindings of PICK answer first and return "break", so the bindings of the control never run.
The orange frame around the control under the mouse is made of 4 thin rectangles placed on top of the window.
"""
import tkinter as tk

from gui import base

THICKNESS = 2


class Picker:
    def __init__(self, root, on_pick, on_toggle=None):
        """on_pick(control) is called when a control is clicked; on_toggle(active) when Pick turns on or off."""
        self.root, self.on_pick, self.on_toggle = root, on_pick, on_toggle
        self.active = False
        self.target = None  # the control framed right now
        self.borders = [tk.Frame(root, bg=base.ACCENT) for _ in range(4)]
        for sequence, handler in (("<Enter>", self._enter), ("<Leave>", self._leave), ("<Button-1>", self._click),
                                  ("<Button-3>", self._cancel), ("<Button-2>", self._swallow),
                                  ("<ButtonRelease-1>", self._swallow), ("<ButtonRelease-2>", self._swallow),
                                  ("<ButtonRelease-3>", self._swallow), ("<B1-Motion>", self._swallow),
                                  ("<Double-Button-1>", self._swallow)):
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

    def _enter(self, event):
        if self.active:
            self.show(self._control(event.widget))

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
        self.show(self._control(under) if under is not None else None)

    def _click(self, event):
        if not self.active:
            return None
        control = self._control(event.widget)
        if control is not None:
            self.show(control)
            self.on_pick(control)
        return "break"

    def _cancel(self, _event):
        if not self.active:
            return None
        self.set(False)
        return "break"

    def _swallow(self, _event):
        return "break" if self.active else None

    # ------------------------------------------------------------------ the orange frame

    def show(self, control):
        """Frames the control (None = no frame)."""
        if control is None or not control.winfo_viewable():
            return self.hide()
        self.target = control
        x = control.winfo_rootx() - self.root.winfo_rootx()
        y = control.winfo_rooty() - self.root.winfo_rooty()
        width, height, t = control.winfo_width(), control.winfo_height(), THICKNESS
        # Just outside the control: the mouse on the control is never on the frame
        places = ((x - 2 * t, y - 2 * t, width + 4 * t, t), (x - 2 * t, y + height + t, width + 4 * t, t),
                  (x - 2 * t, y - 2 * t, t, height + 4 * t), (x + width + t, y - 2 * t, t, height + 4 * t))
        for border, (bx, by, bw, bh) in zip(self.borders, places):
            border.place(x=bx, y=by, width=bw, height=bh)
            border.lift()

    def hide(self):
        self.target = None
        for border in self.borders:
            border.place_forget()

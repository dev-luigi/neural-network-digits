"""
The assistant panel, on the right of the window (F2 opens and closes it): a small chat.

  - Questions: written at the bottom, or clicked among the suggested ones. assistant/brain.py finds the answer
    looking at the real state of the program (gui/app_state.py).
  - Hints: every second the rules of assistant/rules.py look at the state. A new problem shows up in the chat
    with a link that fixes it with one click; with the panel closed, the Assistant button counts the new ones.
  - Pick (F1, see gui/pick.py): the clicked control ends up here, explained, with what it is worth now.
Before every answer a short "thinking" animation (at most 0.6 seconds, different every time): the answer is
already ready, the pause only makes it easier to see where the new text starts.
"""
import random
import tkinter as tk
import traceback
from tkinter import ttk

from assistant import knowledge, rules
from assistant.brain import Assistant, Document
from gui import base
from gui.app_state import app_state
from i18n import tr
from neural_net import storage

WIDTH = 360
CHECK_EVERY = 1000  # milliseconds between one look at the hints and the next
FRAME = 100         # milliseconds of each frame of the "thinking" animation...
MOST_FRAMES = 6     # ...and at most 6 frames: the wait is never longer than 0.6 seconds
THINKING = (("●  ○  ○", "○  ●  ○", "○  ○  ●", "○  ●  ○"),  # the animations, one chosen at random each time
            ("•", "•  •", "•  •  •", "•  •"),
            ("◐", "◓", "◑", "◒"),
            ("▁ ▃ ▅", "▃ ▅ ▇", "▅ ▇ ▅", "▇ ▅ ▃", "▅ ▃ ▁", "▃ ▁ ▃"),
            ("◜", "◝", "◞", "◟"))


class AssistantPanel:
    def __init__(self, window, on_unread=None):
        self.window = window
        self.on_unread = on_unread  # on_unread(n): n hints arrived while the panel was closed
        self.visible = False
        self.unread = 0
        self.shown_hints = set()    # the hints already in the chat: they are written again only if they come back
        self.links = 0              # every link in the chat has its own tag: link1, link2...
        self.waiting = None         # the reply ready to be written at the end of the animation
        self.next_frame = self.next_check = None

        self.frame = tk.Frame(window.root, bg=base.PANEL, width=WIDTH)
        self.frame.pack_propagate(False)  # fixed width, whatever the length of the texts
        top = tk.Frame(self.frame, bg=base.PANEL)
        top.pack(fill="x", padx=14, pady=(12, 0))
        tk.Label(top, text=tr("Assistant"), bg=base.PANEL, fg=base.TEXT, font=(base.FONT, 12, "bold")).pack(
            side="left")
        close = tk.Label(top, text="×", bg=base.PANEL, fg=base.TEXT_SOFT, font=(base.FONT, 14), cursor="hand2")
        close.pack(side="right")
        close.bind("<Button-1>", lambda _: window.toggle_assistant(False))
        tk.Label(self.frame, text=tr("Ask me something, or press F1 and click a control."), bg=base.PANEL,
                 fg=base.TEXT_SOFT, font=(base.FONT, 8), justify="left", wraplength=WIDTH - 28).pack(anchor="w",
                                                                                                     padx=14)
        self.hints_on = tk.BooleanVar(value=storage.settings()["hints"])
        tk.Checkbutton(self.frame, text=tr("Hints: tell me when something looks wrong"), variable=self.hints_on,
                       command=self._hints_changed, bg=base.PANEL, fg=base.TEXT, selectcolor=base.BACKGROUND,
                       activebackground=base.PANEL, activeforeground=base.TEXT, font=(base.FONT, 9),
                       highlightthickness=0, bd=0, cursor="hand2").pack(anchor="w", padx=12, pady=(6, 0))

        # At the bottom (packed before the chat, so the chat takes only the space that is left):
        # the box to write the question and the suggested questions
        bottom = tk.Frame(self.frame, bg=base.PANEL)
        bottom.pack(side="bottom", fill="x", padx=14, pady=(6, 12))
        self.entry = tk.Entry(bottom, bg=base.BACKGROUND, fg=base.TEXT, insertbackground=base.TEXT, relief="flat",
                              font=(base.FONT, 10), highlightthickness=1, highlightbackground=base.BORDER,
                              highlightcolor=base.ACCENT)
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        # Without the tag of the window: the keys of the drawing board (E, Delete, Esc) do not fire while you write
        self.entry.bindtags((str(self.entry), "Entry", "all"))
        self.entry.bind("<Return>", lambda _: self._send())
        base.button(bottom, tr("Ask"), self._send, primary=True).pack(side="left", padx=(6, 0))
        self.suggested = tk.Frame(self.frame, bg=base.PANEL)
        self.suggested.pack(side="bottom", fill="x", padx=14, pady=(6, 0))

        chat = tk.Frame(self.frame, bg=base.PANEL)
        chat.pack(fill="both", expand=True, padx=(14, 6), pady=(10, 0))
        self.chat = tk.Text(chat, bg=base.PANEL, fg=base.TEXT, relief="flat", wrap="word", font=(base.FONT, 9),
                            highlightthickness=0, cursor="arrow", padx=2, pady=2, spacing1=1, spacing3=2)
        scrollbar = ttk.Scrollbar(chat, command=self.chat.yview)
        self.chat.config(yscrollcommand=scrollbar.set, state="disabled")
        scrollbar.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)
        for tag, options in (
                ("you", dict(foreground=base.TEXT_SOFT, justify="right", lmargin1=40, lmargin2=40, spacing1=12)),
                ("title", dict(foreground=base.ACCENT, font=(base.FONT, 10, "bold"), spacing1=8)),
                ("text", dict(foreground=base.TEXT)),
                ("label", dict(foreground=base.TEXT_SOFT, font=(base.FONT, 9, "bold"))),
                ("hint title", dict(foreground=base.RED, font=(base.FONT, 9, "bold"), spacing1=8)),
                ("hint", dict(foreground=base.TEXT, lmargin1=10, lmargin2=10)),
                ("link", dict(foreground=base.BLUE, lmargin1=10, lmargin2=10)),
                ("small", dict(foreground=base.TEXT_SOFT, font=(base.FONT, 8))),
                ("thinking", dict(foreground=base.ACCENT, font=(base.FONT, 14), spacing1=6))):
            self.chat.tag_config(tag, **options)
        self.frame.bind("<Destroy>", self._stop_timers)

    # ------------------------------------------------------------------ opening, tabs, hints

    def set_visible(self, visible):
        """Called by the window when it opens or closes the panel."""
        self.visible = visible
        if visible:
            if not self.chat.get("1.0", "end").strip():  # the first time: a greeting
                self.show(self.brain().hello(self.state()))
            self.unread = 0
            if self.on_unread:
                self.on_unread(0)
            self.tab_changed()
            self.entry.focus_set()

    def tab_changed(self):
        """New tab, new suggested questions."""
        for old in self.suggested.winfo_children():
            old.destroy()
        for question in knowledge.suggestions(self.state()["tab"]):
            chip = tk.Label(self.suggested, text="→ " + question, bg=base.PANEL, fg=base.BLUE, cursor="hand2",
                            font=(base.FONT, 9), justify="left", anchor="w", wraplength=WIDTH - 32)
            chip.pack(anchor="w")
            chip.bind("<Button-1>", lambda _, q=question: self.ask(q))

    def start(self):
        """Starts looking at the hints, every second, until the window closes."""
        def loop():
            try:
                self.check_hints()
            except Exception:  # an error here must not open a window every second: I write it and stop
                traceback.print_exc()
                return
            self.next_check = self.frame.after(CHECK_EVERY, loop)
        self.next_check = self.frame.after(CHECK_EVERY, loop)

    def _stop_timers(self, event):
        """The window closes: the hints loop and the animation must not call a panel that no longer exists."""
        if event.widget is self.frame:
            for timer in (self.next_check, self.next_frame):
                if timer:
                    self.frame.after_cancel(timer)

    def check_hints(self, state=None):
        """Writes in the chat the hints that were not there before. Returns the new ones.
        During the animation of an answer it waits: they will be written at the next look."""
        if not self.hints_on.get() or self.waiting:
            return []
        active = rules.hints(state or self.state())
        new = [hint for hint in active if hint.key not in self.shown_hints]
        self.shown_hints = {hint.key for hint in active}
        for hint in new:
            self._hint(hint)
        if new:
            self._scroll()
            if not self.visible:
                self.unread += len(new)
                if self.on_unread:
                    self.on_unread(self.unread)
        return new

    def _hints_changed(self):
        storage.save_setting("hints", self.hints_on.get())
        self.shown_hints.clear()  # turned on again: the problems still there are written again

    # ------------------------------------------------------------------ questions and answers

    def state(self):
        return app_state(self.window)

    def brain(self):
        """A new Assistant every time: this way it also knows the controls created in the meantime,
        and each chart of the figures (see base.explain_charts)."""
        controls = []
        for widget in base.explained_controls():
            tab = self.tab_of(widget)
            if tab:
                controls.append((base.control_name(widget) or knowledge.tab_name(tab), widget.pick_text, tab))
                controls += [(name, text, tab) for name, text in getattr(widget, "pick_entries", ())]
        return Assistant(controls)

    def tab_of(self, widget):
        """The tab that contains a widget (None if it is outside the tabs)."""
        frames = {str(tab.frame): key for tab, key in zip(self.window.all_tabs, knowledge.TABS)}
        while widget is not None:
            if str(widget) in frames:
                return frames[str(widget)]
            widget = widget.master
        return None

    def _send(self):
        question = self.entry.get()
        self.entry.delete(0, "end")
        self.ask(question)

    def ask(self, question):
        question = question.strip()
        if not question:
            return
        self.finish_thinking()
        self._write(question + "\n", "you")
        self.think(self.brain().answer(question, self.state()))

    def explain_control(self, widget, part=None):
        """Pick: the explanation of the clicked control, with what it is worth now and the related concepts.
        part = the single piece that was clicked (a chart of a figure, a tile: see base.Part), if any."""
        state, brain = self.state(), self.brain()
        tab = self.tab_of(widget) or state["tab"]
        if part is not None:
            name, text, value = part.name, part.text, part.value
        else:
            name = base.control_name(widget) or knowledge.tab_name(tab)
            text, value = widget.pick_text, widget.pick_value
        try:
            value = value() if value else None
        except tk.TclError:
            value = None
        related = [(score, document) for score, document in brain.search(f"{name} {text}", tab)
                   if document.kind == "concept"]
        self.finish_thinking()
        self._write(tr("Pick: {name}", name=name) + "\n", "you")
        self.think(brain.document_reply(Document("control", "", name, text, tab, ""), state, related=related,
                                        value=value))

    # ------------------------------------------------------------------ the "thinking" animation

    def think(self, reply):
        """Shows the reply after a short animation: every time a different one, for 0.3 to 0.6 seconds."""
        self.finish_thinking()
        self.waiting = reply
        frames, count = random.choice(THINKING), random.randint(3, MOST_FRAMES)
        first = random.randrange(len(frames))
        self.chat.mark_set("thinking", "end-1c")
        self.chat.mark_gravity("thinking", "left")  # it stays before the frames written after it

        def step(k):
            if k == count:
                return self.finish_thinking()
            self._clear_thinking()
            self._write(frames[(first + k) % len(frames)] + "\n", "thinking")
            self._scroll()
            self.next_frame = self.frame.after(FRAME, step, k + 1)
        step(0)

    def finish_thinking(self):
        """Stops the animation (if there is one) and writes the reply that was waiting."""
        if self.waiting is None:
            return
        self.frame.after_cancel(self.next_frame)
        self._clear_thinking()
        reply, self.waiting = self.waiting, None
        self.show(reply)

    def _clear_thinking(self):
        self.chat.config(state="normal")
        self.chat.delete("thinking", "end-1c")  # the animation is always the last thing in the chat
        self.chat.config(state="disabled")

    # ------------------------------------------------------------------ writing in the chat

    def show(self, reply):
        """Writes a reply of brain.py: title, text, notes, hints and questions to click."""
        self._write(reply.title + "\n", "title")
        if reply.text:
            self._write(reply.text + "\n", "text")
        for label, text in reply.notes:
            self._write(label + ": ", "label")
            self._write(text + "\n", "text")
        for hint in reply.hints:
            self._hint(hint)
        for question in reply.chips:
            self._link("→ " + question, lambda q=question: self.ask(q))
            self._write("\n")
        self._scroll()

    def _hint(self, hint):
        self._write(tr("Hint") + "\n", "hint title")
        self._write(hint.text + "\n", "hint")
        if hint.fix:
            self._link(tr("Apply: {fix}", fix=hint.fix_label), lambda: self.apply(hint))
            self._write("   ")
        self._link(tr("Why?"), lambda: self.explain_concept(hint.topic))
        self._write("\n")

    def explain_concept(self, key):
        brain = self.brain()
        self.think(brain.document_reply(brain.concept(key), self.state()))

    def apply(self, hint):
        """Does the steps of the fix of a hint (see assistant/rules.py)."""
        self.finish_thinking()
        training = self.window.training_tab
        for step in hint.fix:
            if step[0] == "set":
                training.set_control(step[1], step[2])
            elif step[0] == "tab":
                self.window.tabs.select(knowledge.TABS.index(step[1]))
            elif step[0] == "new network":
                if training.in_progress:
                    self._write(tr("First stop the training (tab 2), then press New network.") + "\n", "small")
                    return self._scroll()
                training.new_network()
            elif step[0] == "reset lab" and self.window.draw_tab.lab:
                self.window.draw_tab.lab.reset()
        self._write(tr("Done: {fix}.", fix=hint.fix_label) + "\n", "small")
        self._scroll()

    def write_note(self, text):
        """A small grey line in the chat (for example: Pick is on)."""
        self.finish_thinking()
        self._write(text + "\n", "small")
        self._scroll()

    def _write(self, text, tag=()):
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.config(state="disabled")

    def _link(self, text, command):
        """Blue text that does `command` when clicked."""
        self.links += 1
        tag = f"link{self.links}"
        self._write(text, ("link", tag))
        self.chat.tag_bind(tag, "<Button-1>", lambda _: command())
        self.chat.tag_bind(tag, "<Enter>", lambda _: self.chat.config(cursor="hand2"))
        self.chat.tag_bind(tag, "<Leave>", lambda _: self.chat.config(cursor="arrow"))

    def _scroll(self):
        self.chat.see("end")

"""
Info tab: who made the program, the version, the updates, the credits and the changelog.
(The language is chosen at the top right of the window, see window.py.)

Here there is also the small window that offers the update when a new version comes out on GitHub
(the check starts by itself at every start, see window.py; the logic is in updater.py).
"""
import re
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

from nn_digits import updater
from nn_digits.gui import base
from nn_digits.i18n import tr
from nn_digits.neural_net import storage
from nn_digits.project import AUTHOR, GITHUB_PROFILE, NAME, PROJECT_PAGE, RELEASES_PAGE, VERSION, WEBSITE

TEXT_WIDTH = 560


def readable_markdown(markdown):
    """The news (written in Markdown in the CHANGELOG) made readable in a text box:
    away with ** and `, the "- " become bullets and the lines broken in the middle of a sentence are joined."""
    lines = []
    for line in markdown.replace("**", "").replace("`", "").splitlines():
        text, indent = line.strip(), len(line) - len(line.lstrip())
        if text.startswith("- "):
            lines.append(" " * indent * 2 + "•  " + text[2:])
        elif text and lines and lines[-1]:
            lines[-1] += " " + text  # continues the line before
        else:
            lines.append(text)
    return "\n".join(lines)


def changelog_page(parent):
    """The Changelog page: all the versions of CHANGELOG.md (newest first), with highlighted titles."""
    text = tk.Text(parent, width=110, height=30, bg=base.PANEL, fg=base.TEXT, relief="flat", wrap="word",
                   font=(base.FONT, 9), padx=16, pady=12, highlightthickness=0)
    text.tag_config("heading", foreground=base.ACCENT, font=(base.FONT, 12, "bold"))
    try:
        changelog = updater.CHANGELOG.read_text(encoding="utf-8")
    except OSError:
        changelog = ""
    # Splitting on the "## [x.y.z] - date" lines keeps them too: [intro, title 1, text 1, title 2, text 2, ...]
    parts = re.split(r"^## (\[.*)$", changelog, flags=re.MULTILINE)
    for heading, body in zip(parts[1::2], parts[2::2]):
        text.insert("end", heading.replace("[", "").replace("]", "") + "\n", "heading")
        text.insert("end", readable_markdown(body).strip() + "\n\n")
    if len(parts) == 1:
        text.insert("end", tr("(no notes)"))
    text.config(state="disabled")
    return text


class InfoTab(base.Tab):
    title = tr("Info")

    def __init__(self, window):
        super().__init__(window)
        # ---- at the top, the switch between the two pages
        self.page = tk.StringVar(value="about")
        top = tk.Frame(self.frame, bg=base.BACKGROUND)
        top.pack(anchor="nw", padx=48, pady=(20, 0))
        base.choice_buttons(top, "", [(tr("About"), "about"), (tr("Changelog"), "changelog")], self.page,
                            self._show_page)
        about = tk.Frame(self.frame, bg=base.BACKGROUND)
        self.pages = {"about": about, "changelog": changelog_page(self.frame)}
        left = tk.Frame(about, bg=base.BACKGROUND, width=620)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 60))
        right = tk.Frame(about, bg=base.BACKGROUND)
        right.grid(row=0, column=1, sticky="nw")

        # ---- the project and the author
        tk.Label(left, text=tr(NAME), bg=base.BACKGROUND, fg=base.ACCENT, font=(base.FONT, 26, "bold")).pack(anchor="w")
        base.label(left, tr("version {version}", version=VERSION), base.TEXT_SOFT, 12, width=TEXT_WIDTH)
        base.label(left, tr(
            "A small neural network written from scratch in Python and NumPy, without artificial intelligence "
            "libraries, that learns to recognize handwritten digits. Every tab shows one step: the data, the "
            "training, the evaluation, the network at work and the math it does inside, number by number."),
            base.TEXT, 10, pady=(14, 0), width=TEXT_WIDTH)

        base.section(left, tr("Author"))
        tk.Label(left, text=AUTHOR, bg=base.BACKGROUND, fg=base.TEXT, font=(base.FONT, 18, "bold")).pack(anchor="w",
                                                                                                       pady=(4, 6))
        for name, url in ((tr("Website"), WEBSITE), ("GitHub", GITHUB_PROFILE), (tr("Project code"), PROJECT_PAGE)):
            r = base.row(left, pady=2)
            tk.Label(r, text=name, bg=base.BACKGROUND, fg=base.TEXT_SOFT, font=(base.FONT, 10), width=18,
                     anchor="w").pack(side="left")
            base.link(r, url.removeprefix("https://"), url).pack(side="left")

        base.section(left, tr("License and credits"))
        base.label(left, tr(
            "Code under the MIT license: you can use it, change it and redistribute it crediting the author.\n"
            "Digit photos: MNIST dataset by Yann LeCun, Corinna Cortes and Christopher J. C. Burges "
            "(CC BY-SA 3.0 license).\n"
            "Made with Python, NumPy, Matplotlib, Pillow and Tkinter."), width=TEXT_WIDTH)
        r = base.row(left, pady=(6, 0))
        base.link(r, tr("Full license"), f"{PROJECT_PAGE}/blob/main/LICENSE", 9).pack(side="left")
        base.link(r, tr("Version history"), f"{PROJECT_PAGE}/blob/main/CHANGELOG.md", 9).pack(side="left", padx=16)

        # ---- updates
        base.section(right, tr("Updates"))
        self.status = base.label(right, tr("Installed version: {version}", version=VERSION), base.TEXT, 10,
                                 width=TEXT_WIDTH)
        r = base.row(right, pady=(8, 0))
        self.check_button = base.button(r, tr("Check now"), self.check_for_updates, primary=True)
        self.check_button.pack(side="left")
        base.button(r, tr("All versions"), lambda: webbrowser.open(RELEASES_PAGE)).pack(side="left", padx=6)
        self.at_start = base.checkbox(right, tr("Check for updates at every start"), lambda: (
            storage.save_setting("check_updates", self.at_start.get())))
        self.at_start.set(storage.settings()["check_updates"])
        self._show_page()

    def _show_page(self):
        """Shows the page chosen with the switch at the top: About or Changelog."""
        for page in self.pages.values():
            page.pack_forget()
        self.pages[self.page.get()].pack(anchor="nw", fill="both", expand=True, padx=48, pady=(16, 36))

    # ------------------------------------------------ checking for updates

    def check_for_updates(self, silent=False):
        """Asks GitHub for the latest version (without blocking the window). If it is newer it opens the
        small update window. silent=True (at start): no error on screen if internet is missing,
        and no small window for a version that you chose to skip."""
        self.check_button.config(state="disabled")
        self.status.config(text=tr("Checking on GitHub..."), fg=base.TEXT)

        def job(send):
            send("release", updater.latest_release())

        def on_news(kind, data):
            if kind == "release":
                self._handle_release(data, silent)
            elif kind == "error":
                self.status.config(text=tr("I can't reach GitHub: are you connected to the internet?\n({error})",
                                           error=data), fg=base.TEXT_SOFT)
            if kind in ("done", "error"):
                self.check_button.config(state="normal")

        base.in_background(self.frame, job, on_news)

    def _handle_release(self, release, silent):
        if release is None:
            return self.status.config(text=tr("Installed version: {version}. There are no published versions on "
                                              "GitHub yet.", version=VERSION))
        if not updater.is_newer(release["version"]):
            return self.status.config(text=tr("You have the latest version ({version}).", version=VERSION),
                                      fg=base.GREEN)
        self.status.config(text=tr("Version {new} is available (you have {version}).", new=release["version"],
                                   version=VERSION), fg=base.ACCENT)
        if not (silent and storage.settings()["skipped_version"] == release["version"]):
            return UpdateDialog(self.window, release)


class UpdateDialog:
    """The small window "Version X is available": what's new, and the buttons to update."""

    def __init__(self, window, release):
        self.window, self.release = window, release
        self.dialog = tk.Toplevel(window.root, bg=base.BACKGROUND, padx=26, pady=22)
        self.dialog.title(tr("Update available"))
        self.dialog.transient(window.root)
        self.dialog.resizable(False, False)

        base.label(self.dialog, tr("Version {version} is available", version=release["version"]), base.ACCENT, 16,
                   True, width=TEXT_WIDTH)
        base.label(self.dialog, tr("You have version {version}. When you update, your photos and your model stay "
                                   "as they are.", version=VERSION), width=TEXT_WIDTH)
        base.section(self.dialog, tr("What's new"))
        text = tk.Text(self.dialog, width=76, height=12, bg=base.PANEL, fg=base.TEXT, relief="flat",
                       wrap="word", font=(base.FONT, 9), padx=12, pady=10, highlightthickness=0)
        text.insert("1.0", readable_markdown(release["notes"]) or tr("(no notes)"))
        text.config(state="disabled")
        text.pack(anchor="w", pady=(6, 0))
        self.bar = ttk.Progressbar(self.dialog, maximum=1.0)
        self.status = base.label(self.dialog, "", base.TEXT, width=TEXT_WIDTH)

        r = base.row(self.dialog, pady=(14, 0))
        self.buttons = [base.button(r, tr("Update now"), self.install, primary=True),
                        base.button(r, tr("See on GitHub"), lambda: webbrowser.open(release["page"])),
                        base.button(r, tr("Later"), self.dialog.destroy),
                        base.button(r, tr("Skip this version"), self.skip)]
        for button in self.buttons:
            button.pack(side="left", padx=(0, 6))

    def skip(self):
        """Don't offer this version again at start (from the Info tab you can always update)."""
        storage.save_setting("skipped_version", self.release["version"])
        self.dialog.destroy()

    def install(self):
        if updater.installed_with_git():
            return messagebox.showinfo(tr("Update"), tr("This copy of the program was downloaded with git: to "
                                                        "update it run  git pull  in its folder."), parent=self.dialog)
        if updater.installed_with_pip():
            return messagebox.showinfo(tr("Update"), tr("This copy of the program was installed with pip: to update "
                                                        "it run in the terminal\n{command}\nthen open it again.",
                                                        command=updater.UPGRADE_COMMAND), parent=self.dialog)
        if self.window.training_tab.in_progress and not messagebox.askyesno(
                tr("Update"), tr("A training is in progress: the program will restart and the training will "
                                 "stop. Update anyway?"), parent=self.dialog):
            return
        for button in self.buttons:
            button.config(state="disabled")
        self.bar.pack(fill="x", pady=(12, 0), before=self.status)
        version, url = self.release["version"], self.release["zip"]

        def job(send):
            try:
                requirements_changed = updater.install(url, lambda fraction: send("download", fraction))
            except Exception as error:
                raise RuntimeError(tr("Update failed, the program has not changed.\n{error}", error=error)) from error
            if requirements_changed:
                send("status", tr("Installing the new libraries..."))
                try:
                    updater.install_requirements()
                except Exception as error:
                    raise RuntimeError(tr("Program updated, but I couldn't install the new libraries: open the "
                                          "terminal in the program folder and run\n"
                                          "python -m pip install -r requirements.txt")) from error

        def on_news(kind, data):
            if kind == "download":
                self.bar["value"] = data
                self.status.config(text=tr("Downloading version {version}... {fraction:.0%}", version=version,
                                           fraction=data))
            elif kind == "status":
                self.status.config(text=data)
            elif kind == "done":
                messagebox.showinfo(tr("Update completed"), tr("Version {version} installed: the program "
                                                               "restarts.", version=version), parent=self.dialog)
                updater.restart()
                self.window.close()
            elif kind == "error":
                self.bar.pack_forget()
                self.status.config(text=str(data), fg=base.RED)
                for button in self.buttons:
                    button.config(state="normal")

        base.in_background(self.dialog, job, on_news)

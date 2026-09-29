"""
Updates.

At every start the program asks for the latest version (to PyPI if it was installed with pip, to GitHub
otherwise) and, if it is newer, offers it. "Update now" installs it with the same tool that installed it:

    zip   the zip of the GitHub release replaces the program files (data/ is never touched)
    git   git pull
    pipx  pipx upgrade neural-network-digits
    uv    uv tool upgrade neural-network-digits
    pip   python -m pip install --upgrade neural-network-digits

The releases are created by GitHub's CI/CD (.github/workflows/release.yml), which also publishes them on PyPI.
"""
import importlib.util
import json
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from nn_digits.i18n import tr
from nn_digits.neural_net import storage
from nn_digits.project import GITHUB_REPO, GITHUB_USER, PYPI_NAME, RELEASES_PAGE, VERSION

APP_DIR = storage.PROGRAM_DIR  # the folder with start.py (site-packages when installed with pip)
# In the package installed by pip the CHANGELOG travels inside nn_digits/, in the zip and in git it is next to it
CHANGELOG = next((path for path in (Path(__file__).resolve().parent / "CHANGELOG.md", APP_DIR / "CHANGELOG.md")
                  if path.exists()), APP_DIR / "CHANGELOG.md")
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/releases/latest"
PYPI_API = f"https://pypi.org/pypi/{PYPI_NAME}/json"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # Windows: no terminal window pops up
KEEP = {"data", ".git", ".venv", "venv"}  # never replaced by an update
# Up to version 1.1.1 the code was next to start.py, not inside nn_digits/: an update removes what is left of it
# (__pycache__ too: now nothing is imported from the main folder)
OLD_LAYOUT = ("assistant", "gui", "neural_net", "locales", "i18n.py", "updater.py", "__pycache__")


def version_numbers(version):
    """ "v1.2.0" -> (1, 2, 0): this way versions are compared as numbers (1.10 comes after 1.9)."""
    return tuple(int(n) for n in re.findall(r"\d+", version)[:3])


def is_newer(version, than=VERSION):
    return version_numbers(version) > version_numbers(than)


def release_notes(version=VERSION, changelog=CHANGELOG):
    """What is new in a version, taken from its "## [x.y.z]" paragraph of CHANGELOG.md ("" if missing)."""
    if not changelog.exists():
        return ""
    lines, inside = [], False
    for line in changelog.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line.startswith(f"## [{version}]")
        elif inside:
            lines.append(line)
    return "\n".join(lines).strip()


def installed_with_git():
    return storage.PORTABLE and (APP_DIR / ".git").exists()


def installed_with_pip():
    """Installed from PyPI (with pip, pipx or uv): there is no start.py next to the package."""
    return not storage.PORTABLE


def release_site():
    return "PyPI" if installed_with_pip() else "GitHub"


def install_method():
    """How this copy was installed: "zip", "git", "pipx", "uv tool", "uv pip", "pip --user" or "pip".
    Nothing to guess: pipx and uv leave a file in the environment, pip and uv write their name in INSTALLER."""
    if not installed_with_pip():
        return "git" if installed_with_git() else "zip"
    if Path(sys.prefix, "pipx_metadata.json").exists():
        return "pipx"
    if Path(sys.prefix, "uv-receipt.toml").exists():
        return "uv tool"
    # Not with importlib.metadata: it would open (and keep open) the .exe that started the program, see update()
    installer = next(APP_DIR.glob("neural_network_digits-*.dist-info/INSTALLER"), None)
    if installer and installer.read_text().strip() == "uv":
        return "uv pip"
    if site.ENABLE_USER_SITE and APP_DIR == Path(site.getusersitepackages()).resolve():
        return "pip --user"
    return "pip"


def update_command(method=None):
    """The command that installs the new version (None for the zip copies: see install())."""
    method = method or install_method()
    if method == "zip":
        return None
    if method == "git":
        return ["git", "pull", "--ff-only"]
    # pipx and uv use their command; if it is not in the PATH, the pip of the environment, when there is one
    has_pip = importlib.util.find_spec("pip") is not None
    if method == "pipx" and (shutil.which("pipx") or not has_pip):
        return ["pipx", "upgrade", PYPI_NAME]
    if method == "uv tool":
        return ["uv", "tool", "upgrade", PYPI_NAME]
    if method == "uv pip" and (shutil.which("uv") or not has_pip):
        return ["uv", "pip", "install", "--upgrade", "--python", sys.executable, PYPI_NAME]
    options = []
    if method == "pip --user":
        options.append("--user")  # otherwise the old copy, in the user folder, would keep coming first
        # Debian, Ubuntu, Homebrew...: pip refuses to install outside a venv (PEP 668). This copy got here
        # with --break-system-packages, so the update needs it too (it only touches the user folder)
        if Path(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED").exists():
            options.append("--break-system-packages")
    return [sys.executable, "-m", "pip", "install", "--upgrade", *options, PYPI_NAME]


def command_text(command):
    """The command as you would type it in the terminal."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def read_json(url, timeout, accept="application/json"):
    request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": f"{GITHUB_REPO}/{VERSION}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def github_release(timeout=5):
    """The latest GitHub release: {"version", "notes", "page", "zip"}, or None if there is none yet."""
    try:
        release = read_json(LATEST_RELEASE_API, timeout, "application/vnd.github+json")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    # The zip prepared by the CI/CD; if missing, the one GitHub creates by itself with all the code
    ready_zip = [a["browser_download_url"] for a in release.get("assets", []) if a["name"].endswith(".zip")]
    return {"version": release["tag_name"].lstrip("v"), "notes": release.get("body") or "",
            "page": release["html_url"], "zip": ready_zip[0] if ready_zip else release["zipball_url"]}


def pypi_version(timeout=5):
    try:
        return read_json(PYPI_API, timeout)["info"]["version"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def latest_release(timeout=5):
    """Like github_release(). The pip copies ask PyPI, where the version arrives a few minutes after GitHub:
    offering it earlier, pip would not find it yet. Without internet it raises an error."""
    if not installed_with_pip():
        return github_release(timeout)
    version = pypi_version(timeout)
    if version is None:
        return None
    try:
        release = github_release(timeout)  # only for the news
    except (OSError, ValueError):
        release = None
    if release is None or release["version"] != version:
        release = {"version": version, "notes": "", "page": RELEASES_PAGE, "zip": None}
    return release


def install(zip_url, progress=None):
    """Downloads the new version and replaces the program files (data/ excluded).
    If something goes wrong it puts the old files back. Returns True if the required
    libraries (requirements.txt) changed: in that case reinstall them with install_requirements()."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        zip_file = temp / "new_version.zip"
        with urllib.request.urlopen(zip_url, timeout=60) as response, open(zip_file, "wb") as file:
            total = int(response.headers.get("Content-Length", 0))
            while block := response.read(65536):
                file.write(block)
                if progress and total:
                    progress(file.tell() / total)

        with zipfile.ZipFile(zip_file) as archive:
            for name in archive.namelist():  # safety: no file may end up outside the folder
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    raise ValueError(tr("Invalid zip: {name}", name=name))
            archive.extractall(temp / "extracted")
            for info in archive.infolist():  # extractall forgets which files are executable (start.sh on Linux)
                mode = info.external_attr >> 16
                if mode & 0o111:
                    (temp / "extracted" / info.filename).chmod(mode & 0o777)
        # The zip contains a folder with the program inside: I recognize it from start.py next to nn_digits/
        roots = [p.parent.parent for p in (temp / "extracted").rglob("project.py")
                 if p.parent.name == "nn_digits" and (p.parent.parent / "start.py").exists()]
        if not roots:
            raise ValueError(tr("The downloaded zip does not contain the program."))
        new_files = [p for p in roots[0].iterdir() if p.name not in KEEP]

        requirements_before = requirements()
        old_files = temp / "old_version"
        old_files.mkdir()
        moved, placed = [], []
        try:
            for new in new_files:
                destination = APP_DIR / new.name
                if destination.exists():
                    shutil.move(str(destination), str(old_files / new.name))
                    moved.append(new.name)
                shutil.move(str(new), str(destination))
                placed.append(new.name)
        except Exception:
            for name in placed:  # something went wrong: I remove the new files and put back the old ones
                path = APP_DIR / name
                shutil.rmtree(path) if path.is_dir() else path.unlink()
            for name in moved:
                shutil.move(str(old_files / name), str(APP_DIR / name))
            raise
        for name in OLD_LAYOUT:  # the code of the old versions that the new one no longer has
            if name not in placed and (APP_DIR / name).exists():
                shutil.move(str(APP_DIR / name), str(old_files / name))
        return requirements() not in ("", requirements_before)


def run(command):
    """Runs the command without opening terminal windows. If it fails, the error also says how to do it by hand."""
    by_hand = tr("To update by hand, run in the terminal:\n{command}", command=command_text(command))
    try:
        result = subprocess.run(command, cwd=APP_DIR, capture_output=True, encoding="utf-8", errors="replace",
                                creationflags=NO_WINDOW)
    except FileNotFoundError:
        raise RuntimeError(tr("{program} is not on this computer (or not in the PATH).", program=command[0])
                           + "\n" + by_hand) from None
    if result.returncode != 0:
        lines = (result.stderr.strip() or result.stdout.strip()).splitlines()
        reason = [line for line in lines if line.lower().startswith(("error", "fatal"))] or lines[-6:]
        raise RuntimeError("\n".join(reason) + "\n\n" + by_hand)


def update(zip_url, progress=None):
    """Installs the new version with the same tool that installed this copy. Returns True if requirements.txt
    changed (zip and git only: pip, pipx and uv install the libraries by themselves)."""
    command = update_command()
    if command is None:
        return install(zip_url, progress)
    requirements_before = requirements()
    # Windows does not let pip replace the .exe that started the program while it runs (pip fails and leaves
    # the program half removed), but it lets it be renamed: it goes aside and the update puts a new one in its
    # place. pip's launcher removes ".exe" from argv[0], and pipx may start it from a link to the real one
    exe = Path(sys.argv[0]).with_suffix(".exe").resolve()
    aside = exe.with_name(exe.name + ".old")
    move = os.name == "nt" and exe.name.lower() == f"{PYPI_NAME}.exe" and exe.exists()
    if move:
        try:
            aside.unlink()  # left by the previous update
        except OSError:
            pass
        exe.replace(aside)
    try:
        run(command)
    except Exception:
        if move and not exe.exists():
            aside.replace(exe)
        raise
    return requirements() not in ("", requirements_before)


def requirements():
    path = APP_DIR / "requirements.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def install_requirements():
    """pip install -r requirements.txt, without opening terminal windows."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(APP_DIR / "requirements.txt")],
                   check=True, capture_output=True, creationflags=NO_WINDOW)


def restart():
    """Opens the program again (the new version just installed). The caller then closes the old one."""
    if installed_with_pip():
        subprocess.Popen([sys.executable, "-m", "nn_digits"])
    else:
        subprocess.Popen([sys.executable, str(APP_DIR / "start.py")], cwd=APP_DIR)

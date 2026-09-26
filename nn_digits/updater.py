"""
Updates from GitHub.

At every start the program asks GitHub which is the latest published version (the latest "release").
If it is newer than this one, it offers it. Updating means: downloading the zip of the release and
putting the new files in place of the old ones. The data/ folder (photos, model, settings)
is never touched.

Only the copies downloaded as a zip update themselves like this. A copy made with git is updated with
"git pull", one installed with pip with "pipx upgrade neural-network-digits" (or pip install -U).

The releases are created automatically by GitHub's CI/CD (.github/workflows/release.yml).
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from nn_digits.i18n import tr
from nn_digits.neural_net import storage
from nn_digits.project import GITHUB_REPO, GITHUB_USER, PYPI_NAME, VERSION

APP_DIR = storage.PROGRAM_DIR  # where the program is installed: the folder with start.py
# In the package installed by pip the CHANGELOG travels inside nn_digits/, in the zip and in git it is next to it
CHANGELOG = next((path for path in (Path(__file__).resolve().parent / "CHANGELOG.md", APP_DIR / "CHANGELOG.md")
                  if path.exists()), APP_DIR / "CHANGELOG.md")
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/releases/latest"
UPGRADE_COMMAND = f"pipx upgrade {PYPI_NAME}"  # for the copies installed with pip
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
    """If the program is a git working copy, it is updated with "git pull" and not from here."""
    return storage.PORTABLE and (APP_DIR / ".git").exists()


def installed_with_pip():
    """Installed with pip (or pipx): it is updated with UPGRADE_COMMAND and not from here."""
    return not storage.PORTABLE


def latest_release(timeout=5):
    """Asks GitHub for the latest release. Returns {"version", "notes", "page", "zip"},
    or None if none has been published yet. Without internet it raises an error."""
    request = urllib.request.Request(LATEST_RELEASE_API, headers={"Accept": "application/vnd.github+json",
                                                                  "User-Agent": f"{GITHUB_REPO}/{VERSION}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            release = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:  # repository without releases (or not published yet)
            return None
        raise
    # The zip prepared by the CI/CD; if missing, the one GitHub creates by itself with all the code
    ready_zip = [a["browser_download_url"] for a in release.get("assets", []) if a["name"].endswith(".zip")]
    return {"version": release["tag_name"].lstrip("v"), "notes": release.get("body") or "",
            "page": release["html_url"], "zip": ready_zip[0] if ready_zip else release["zipball_url"]}


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

        requirements = APP_DIR / "requirements.txt"
        requirements_before = requirements.read_text(encoding="utf-8") if requirements.exists() else ""
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
        return requirements.exists() and requirements.read_text(encoding="utf-8") != requirements_before


def install_requirements():
    """pip install -r requirements.txt, without opening terminal windows."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(APP_DIR / "requirements.txt")],
                   check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def restart():
    """Opens the program again (the new version just installed). The caller then closes the old one."""
    if installed_with_pip():
        subprocess.Popen([sys.executable, "-m", "nn_digits"])
    else:
        subprocess.Popen([sys.executable, str(APP_DIR / "start.py")], cwd=APP_DIR)

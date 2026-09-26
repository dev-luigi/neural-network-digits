"""
Publishing a new version, in two commands:

    python tools/release.py prepare 1.1.0    changes the version and prepares the paragraph in CHANGELOG.md
    (write the changes in CHANGELOG.md)
    python tools/release.py publish          checks, runs the tests, commit, tag v1.1.0 and push to GitHub

When the tag is pushed, GitHub's CI/CD (.github/workflows/release.yml) runs the tests again, creates the zip,
publishes the release and (after your approval) publishes the package on PyPI: from that moment the installed
programs offer the update.

Commands used by the CI/CD:
    python tools/release.py check v1.1.0     does the tag match the version, and are the changes written?
    python tools/release.py notes 1.1.0      prints the changes (they become the text of the release)
"""
import datetime
import re
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from nn_digits import updater  # noqa: E402  (needs the path added just above)

PROJECT_FILE = APP_DIR / "nn_digits" / "project.py"
CHANGELOG = APP_DIR / "CHANGELOG.md"
TODO_NOTE = "- (write the changes here)"


def current_version():
    return re.search(r'VERSION = "([^"]+)"', PROJECT_FILE.read_text(encoding="utf-8")).group(1)


def fail(message):
    print(message)
    sys.exit(1)


def check(tag):
    version = current_version()
    if tag != f"v{version}":
        fail(f"The tag {tag} does not match the program version (nn_digits/project.py says {version}).")
    notes = updater.release_notes(version, CHANGELOG)
    if not notes or TODO_NOTE in notes:
        fail(f"CHANGELOG.md is missing the changes of version {version}.")
    print(f"All good for version {version}.")


def prepare(new):
    old = current_version()
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        fail("Write the version as MAJOR.MINOR.PATCH, for example 1.1.0")
    if not updater.is_newer(new, old):
        fail(f"The new version must be higher than the current one ({old}).")
    text = PROJECT_FILE.read_text(encoding="utf-8")
    PROJECT_FILE.write_text(text.replace(f'VERSION = "{old}"', f'VERSION = "{new}"'), encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    if f"## [{new}]" not in changelog:  # the new paragraph goes above the one of the previous version
        paragraph = f"## [{new}] - {datetime.date.today()}\n\n{TODO_NOTE}\n\n"
        start = changelog.find("## [")
        changelog = changelog + "\n" + paragraph if start < 0 else changelog[:start] + paragraph + changelog[start:]
        CHANGELOG.write_text(changelog, encoding="utf-8")
    print(f"Version {old} -> {new}.\nNow write the changes in CHANGELOG.md, then:  "
          "python tools/release.py publish")


def git(*args):
    return subprocess.run(["git", *args], cwd=APP_DIR, check=True, capture_output=True, text=True).stdout


def publish():
    version = current_version()
    check(f"v{version}")
    if git("branch", "--show-current").strip() != "main":
        fail("Releases are published only from the main branch: merge develop into main first.")
    if f"v{version}" in git("tag").split():
        fail(f"The tag v{version} already exists: prepare a new version first.")
    print("Running the tests...")
    if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=APP_DIR).returncode:
        fail("Some tests fail: no release.")
    print(f"\nAbout to commit all the changes, create the tag v{version} and push them to GitHub.")
    if input("Confirm? [y/N] ").strip().lower() not in ("y", "yes"):
        fail("Cancelled: nothing was changed.")
    git("add", "-A")
    if git("status", "--porcelain"):
        git("commit", "-m", f"Version {version}")
    git("tag", "-a", f"v{version}", "-m", f"Version {version}")
    git("push", "origin", "HEAD")
    git("push", "origin", f"v{version}")
    print(f"Done: the CI/CD is creating the release v{version}. You can find it in the repository's Actions tab.")


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["help"]
    if command == "check" and rest:
        check(rest[0])
    elif command == "notes":
        print(updater.release_notes(rest[0].lstrip("v") if rest else current_version(), CHANGELOG))
    elif command == "prepare" and rest:
        prepare(rest[0])
    elif command == "publish":
        publish()
    else:
        print(__doc__)

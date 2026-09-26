"""
Checks the package built for PyPI before it is published (the CI/CD runs it after  python -m build):

    python tools/check_package.py dist

  - the wheel contains the whole nn_digits package (translations and CHANGELOG too) and nothing else;
  - the source archive contains only the files listed in pyproject.toml;
  - version, command and libraries are the ones of the program (nn_digits/project.py and requirements.txt).
"""
import re
import sys
import tarfile
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from nn_digits.project import PYPI_NAME, VERSION  # noqa: E402  (needs the path added just above)

SDIST_FILES = {"nn_digits", "README.md", "CHANGELOG.md", "LICENSE", "requirements.txt", "pyproject.toml",
               "PKG-INFO", ".gitignore"}  # the last three are added by the build


def fail(message):
    print("ERROR:", message)
    sys.exit(1)


def program_files():
    """The files of the nn_digits package in the repository, as they must end up in the wheel."""
    package = APP_DIR / "nn_digits"
    return {path.relative_to(APP_DIR).as_posix() for path in package.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts}


def check_wheel(path):
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        info = f"{PYPI_NAME.replace('-', '_')}-{VERSION}.dist-info"
        metadata = wheel.read(f"{info}/METADATA").decode("utf-8")
        entry_points = wheel.read(f"{info}/entry_points.txt").decode("utf-8")
    code = {name for name in names if not name.startswith(info + "/")}
    expected = program_files() | {"nn_digits/CHANGELOG.md"}
    if code != expected:
        fail(f"the wheel is different from the nn_digits folder: missing {sorted(expected - code)}, "
             f"extra {sorted(code - expected)}")
    if not re.search(rf"^Version: {re.escape(VERSION)}$", metadata, re.MULTILINE):
        fail(f"the wheel does not have the version {VERSION}")
    requirements = sorted(line.strip() for line in (APP_DIR / "requirements.txt").read_text().splitlines()
                          if line.strip() and not line.startswith("#"))
    if sorted(re.findall(r"^Requires-Dist: (.+)$", metadata, re.MULTILINE)) != requirements:
        fail("the libraries in pyproject.toml are not the same as requirements.txt")
    if f"{PYPI_NAME} = nn_digits.cli:main" not in entry_points:
        fail(f"the wheel does not install the {PYPI_NAME} command")
    print(f"wheel ok: {path.name} ({len(code)} files)")


def check_sdist(path):
    with tarfile.open(path) as sdist:
        names = sdist.getnames()
    top = f"{PYPI_NAME.replace('-', '_')}-{VERSION}"
    outside = [name for name in names if name.split("/")[0] != top]
    extra = {name.split("/")[1] for name in names if name.count("/") >= 1} - SDIST_FILES
    if outside or extra:
        fail(f"unexpected files in the source archive: {sorted(outside) + sorted(extra)}")
    print(f"sdist ok: {path.name} ({len(names)} files)")


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels, sdists = sorted(folder.glob("*.whl")), sorted(folder.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"{folder} must contain one wheel and one .tar.gz, it has {len(wheels)} and {len(sdists)}")
    check_wheel(wheels[0])
    check_sdist(sdists[0])

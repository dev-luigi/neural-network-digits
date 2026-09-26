"""
Who made the program, which version it is, where it lives on GitHub and how it is started.

To publish a new version you change VERSION (see "Publishing a new version" in the README):
GitHub's CI/CD creates the release and the programs already installed offer it at startup.
"""
import os

NAME = "Neural network from scratch"  # shown translated, with tr(NAME)
VERSION = "1.1.1"  # MAJOR.MINOR.PATCH (semantic versioning)

AUTHOR = "Luigi Tanzillo"
WEBSITE = "https://luigitanzillo.it"
GITHUB_USER = "dev-luigi"
GITHUB_REPO = "neural-network-digits"

GITHUB_PROFILE = f"https://github.com/{GITHUB_USER}"
PROJECT_PAGE = f"{GITHUB_PROFILE}/{GITHUB_REPO}"
RELEASES_PAGE = f"{PROJECT_PAGE}/releases"

# How the terminal commands are started on this computer (for the messages that suggest one)
LAUNCHER = "start.bat" if os.name == "nt" else "./start.sh"

#!/bin/sh
# ==========================================================================
#  QUICK START (Linux and macOS)
#    ./start.sh                        opens the graphical interface
#    ./start.sh train --epochs 100     runs one step from the terminal
#  The first time it creates a virtual environment in .venv and installs
#  there the libraries it needs: the Python of the system is not touched.
# ==========================================================================
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python not found. Install it:"
    echo "  Linux:  with the package manager, for example  sudo apt install python3 python3-venv python3-tk"
    echo "  macOS:  from https://www.python.org"
    exit 1
fi
if ! python3 -c "import sys; sys.exit(sys.version_info < (3, 9))"; then
    echo "Python 3.9 or newer is needed, this is $(python3 --version 2>&1)."
    exit 1
fi

# Tkinter (the windows) is part of Python, but on Linux it is often a separate package
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "Tkinter is missing (it draws the windows). Install it with:"
    echo "  Ubuntu / Debian:  sudo apt install python3-tk"
    echo "  Fedora:           sudo dnf install python3-tkinter"
    echo "  Arch:             sudo pacman -S tk"
    echo "  macOS:            install Python from https://www.python.org (Tkinter included)"
    exit 1
fi
# The Python that comes with macOS has a very old Tk (8.5) that draws the windows badly
if ! python3 -c "import sys, tkinter; sys.exit(tkinter.TkVersion < 8.6)" 2>/dev/null; then
    echo "The Tkinter of this Python is too old (Tk 8.6 or newer is needed)."
    echo "On macOS install Python from https://www.python.org (Tkinter included)."
    exit 1
fi

# The libraries go in a virtual environment of the project (.venv), because many
# distributions do not let pip install them in the Python of the system
if [ ! -x .venv/bin/python ]; then
    echo "First time: creating the virtual environment in .venv..."
    if ! python3 -m venv .venv; then
        rm -rf .venv
        echo "The virtual environment could not be created."
        echo "On Ubuntu / Debian install it with:  sudo apt install python3-venv"
        exit 1
    fi
fi

# Quick check (without loading them) that the libraries are there; if they are missing I install them
if ! .venv/bin/python -c "import importlib.util as u, sys; sys.exit(not all(u.find_spec(m) for m in ('numpy', 'PIL', 'matplotlib')))"; then
    echo "First time: installing the required libraries..."
    .venv/bin/python -m pip install -r requirements.txt || exit 1
fi

exec .venv/bin/python start.py "$@"

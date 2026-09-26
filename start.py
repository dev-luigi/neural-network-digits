"""
START: the starting point of the program downloaded as a zip or with git.

    start.bat  (double-click)      opens the graphical interface  (./start.sh on Linux and macOS)
    python start.py                the same thing, from the terminal
    python start.py <command>      runs a single step from the terminal (also:  start.bat <command>)
    python start.py --help         the list of the commands

The program is in the nn_digits folder: the commands are in nn_digits/cli.py.
Installed with pip, the same commands are:  neural-network-digits <command>
"""
from nn_digits.cli import main

if __name__ == "__main__":
    main()

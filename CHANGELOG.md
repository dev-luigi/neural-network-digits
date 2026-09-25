# Changelog

All the versions of the program, newest first. The format is the one of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers follow
[semantic versioning](https://semver.org/): MAJOR.MINOR.PATCH.

The paragraph of each version becomes the text of its GitHub release and shows up in the program
when the update is offered.

## [1.0.1] - 2026-09-25

- **Linux and macOS**: new `start.sh` launcher. The first time it creates a virtual environment in `.venv`
  with the libraries, and if Tkinter or venv are missing it says what to install.
- The updates installed from inside the program keep `start.sh` executable.
- Data tab: every control and chart has its explanation when you hover it with the mouse.
- The messages that suggest a command show the right one for the system (`start.bat` or `./start.sh`).
- On macOS the right mouse button clears the drawing board too.
- The explanation of the initial weights says "LeCun" (for sigmoid and tanh) instead of "Xavier".
- README: macOS instructions and corrections.

## [1.0.0] - 2026-09-25

First public version.

- Neural network written from scratch with NumPy: forward, softmax, cross-entropy, backpropagation, SGD with
  momentum, L2, dropout, relu, leaky relu, sigmoid and tanh activations.
- Graphical interface with 5 tabs, plus the Info page:
  - **Data**: MNIST download, dataset exploration, reset;
  - **Training**: live, with loss curve, weight gaussians and dozens of knobs;
  - **Evaluation**: photos damaged on purpose, confidence threshold, confusion matrix and points map
    (PCA and t-SNE);
  - **Draw & edit**: drawing board and lab to change biases, weights and neurons;
  - **Inside the network**: the math of every layer and the softmax steps, with the temperature.
- Every step from the terminal too (`start.bat <command>`).
- English and Italian interface, with a language selector in the Info tab.
- A Changelog page in the Info tab.
- Update check at startup and one-click install of the new version.

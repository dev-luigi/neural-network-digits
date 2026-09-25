# Contributing

Thank you for wanting to help. You can:

- **report a bug**: open an [issue](https://github.com/dev-luigi/neural-network-digits/issues/new/choose)
  with the *Bug report* template;
- **propose an idea**: open an issue with the *Feature request* template;
- **change the code**: open a pull request (see below).

For a security problem do not open a public issue: follow [SECURITY.md](SECURITY.md).

## Setup

```bash
git clone https://github.com/dev-luigi/neural-network-digits.git
cd neural-network-digits
git checkout develop

pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
python start.py
```

On Linux and macOS let `./start.sh` create the `.venv` with the libraries, then use its Python:
`.venv/bin/python -m pip install -r requirements-dev.txt` and `.venv/bin/python -m pytest`.

## Pull requests

1. Start from the `develop` branch and open the pull request towards `develop`.
   `main` only receives the versions to publish.
2. Run `python -m pytest` before opening it. The CI runs the tests on Windows, Linux and macOS,
   with Python 3.9 and 3.13: the code has to work with Python 3.9 too.
3. In the description, write what the pull request changes and why.

## Rules of the code

- The logic in `neural_net/` does not depend on the interface: the tabs in `gui/` and the terminal
  commands in `start.py` both use it.
- The program is written in English. Every text shown to the user goes through `tr("...")` with a plain
  string (not an f-string or a variable), for example `tr("Epoch {n}", n=3)`.
- Every text passed to `tr()` needs its Italian translation in `locales/it.json`, with the same
  `{placeholders}`. The tests check it.

## License

By contributing, you agree that your contribution is distributed under the [MIT license](LICENSE) of the
project.

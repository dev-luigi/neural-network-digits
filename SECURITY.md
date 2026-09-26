# Security policy

## Supported versions

Only the latest version, the one on the [Releases](https://github.com/dev-luigi/neural-network-digits/releases/latest)
page, receives fixes. A fix is published as a new version, and the program offers it at the next start
(or with `start.bat update`, on Linux and macOS `./start.sh update`; installed with pip,
`pipx upgrade neural-network-digits`).

## Reporting a vulnerability

Do not open a public issue. Report it privately with
[Report a vulnerability](https://github.com/dev-luigi/neural-network-digits/security/advisories/new),
in the **Security** tab of the repository: only you and the maintainer see the report.

Write the version of the program (you find it in the **Info** tab), your operating system, what the
problem is and how to reproduce it.

## What the program does over the internet

- It downloads the MNIST photos from `storage.googleapis.com` (the public copy used by Keras/TensorFlow),
  only when you ask for it from the **Data** tab or with the `download` command from the terminal.
- At every start of the interface it asks GitHub (`api.github.com`) which is the latest version.
  You can turn this off in the **Info** tab.
- When you accept an update (only the copies downloaded as a zip: the ones installed with pip are updated
  by pip), it downloads the zip of the release from GitHub and replaces the program files. The `data/` folder is not touched, and a zip with files that would end up outside the program
  folder is refused.

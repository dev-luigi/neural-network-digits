"""
START: the starting point of the project.

    start.bat  (double-click)      opens the graphical interface
    python start.py                the same thing, from the terminal
    python start.py <command>      runs a single step from the terminal (also:  start.bat <command>)

        download  1. downloads the digit photos              --per-digit 100  --test-per-digit 20  --all
        explore   2. pre-training: charts about the dataset
        train     3. trains the network                      --epochs 60  --lr 0.05  --noise 0.2  ...
        evaluate  4. tests the network on the test photos    --noise 0.3  --rotation 20  --thickness -1
        draw      5. opens only the drawing board and the lab
        reset     deletes model and charts                   --all (the photos too)  --yes (no confirmation)
        update    checks on GitHub for a new version and installs it

    python start.py train --help       shows all the options of a command
    python start.py --version          shows the version

The charts of the terminal commands are saved in data/charts/.
"""
import argparse

import numpy as np

import updater
from i18n import tr
from neural_net import charts, storage
from neural_net.data import download_mnist, load_photos, save_collection
from neural_net.network import ACTIVATIONS, NeuralNetwork
from neural_net.training import Trainer, robustness, test_on
from project import AUTHOR, LAUNCHER, NAME, VERSION

YES = ("y", "yes", "s", "si", "sì")  # the answers that mean "yes" (in English and in Italian)


def download(args):
    """1. Downloads MNIST and saves a collection of PNG photos in data/photos/ (with --all the whole collection)."""
    mnist = download_mnist(progress=lambda fraction: print(
        "\r  " + tr("downloading MNIST: {fraction:.0%}", fraction=fraction), end="", flush=True))
    print()
    per_digit, test_per_digit = (None, None) if args.all else (args.per_digit, args.test_per_digit)
    counts = save_collection(mnist, per_digit, test_per_digit, progress=lambda fraction: print(
        "\r  " + tr("saving the photos: {fraction:.0%}", fraction=fraction), end="", flush=True))
    print()
    for split, n in zip(("train", "test"), counts):
        print("  " + tr("{n} photos saved in {folder}", n=n, folder=storage.PHOTOS_DIR / split))


def explore(args):
    """2. Pre-training: numbers and charts to understand what the dataset looks like."""
    import matplotlib.pyplot as plt  # imported only here: the graphical interface does not need it

    photos, digits = load_photos("train")
    test_photos, test_digits = load_photos("test")
    print(tr("Photos: {train} for training and {test} for test, each {h}x{w} = {pixels} pixels",
             train=len(photos), test=len(test_photos), h=photos.shape[1], w=photos.shape[2], pixels=photos[0].size))
    print(tr("Pixels: from {low} (black) to {high} (white); {black:.0%} completely black",
             low=photos.min(), high=photos.max(), black=np.mean(photos == 0)))
    print(tr("Photos per digit:"), np.bincount(digits, minlength=10).tolist())
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    charts.exploration(fig, photos, digits, test_photos, test_digits, title=tr("Pre-training: dataset exploration"))
    print(tr("Chart saved in"), storage.save_chart(fig, "1_exploration.png"))
    plt.show()


def train(args):
    """3. Trains the network (with live charts) and saves it in data/model.npz."""
    import matplotlib.pyplot as plt

    trainer = Trainer(args.hidden, args.activation, args.init_scale, args.seed)
    net = trainer.net
    print(tr("Photos: {train} to learn, {val} to check (validation)", train=len(trainer.X_train),
             val=len(trainer.X_val)))
    print(tr("Network: {layers}, activation {activation}, {n} parameters to learn",
             layers=" -> ".join(map(str, net.layers)), activation=net.activation, n=net.n_parameters) + "\n")

    fig = plt.figure(figsize=(16, 9), layout="constrained")

    def plot():
        charts.training(fig, trainer.snapshot(), args.noise, title=tr("Training"))

    if not args.no_plot:
        plot()
        plt.show(block=False)
    for epoch in range(1, args.epochs + 1):
        # The learning rate goes down slowly to zero ("cosine decay"):
        # big steps at the start, small and precise ones at the end
        lr = args.lr * 0.5 * (1 + np.cos(np.pi * (epoch - 1) / args.epochs))
        m = trainer.run_epoch(lr, args.noise, args.batch, args.momentum, args.l2, args.dropout,
                              args.rotation, args.shift)
        print(tr("Epoch {epoch:3d}/{epochs}  lr {lr:.4f}  |  loss train {train_loss:.3f}  val {val_loss:.3f}  |  "
                 "accuracy train {train_acc:6.1%}  val {val_acc:6.1%}", epoch=epoch, epochs=args.epochs, lr=lr, **m))
        if trainer.exploded:
            print("\n" + tr("The network exploded (infinite weights): the learning rate is too high, "
                            "try a lower --lr."))
            break
        if not args.no_plot:
            plot()
            fig.canvas.draw()
            fig.canvas.flush_events()

    plot()
    print("\n" + tr("Charts saved in"), storage.save_chart(fig, "2_training.png"))
    if not trainer.exploded:
        net.save()
        print(tr("Model saved in"), storage.MODEL_FILE)
    if not args.no_plot:
        plt.show()


def evaluate(args):
    """4. Puts the network to the test on the test photos, never seen during training."""
    import matplotlib.pyplot as plt

    if not storage.MODEL_FILE.exists():
        raise SystemExit(tr("No model: train the network first  ({command})", command=f"{LAUNCHER} train"))
    net = NeuralNetwork.load()
    photos, digits = load_photos("test")
    X, probabilities = test_on(net, photos, args.noise, args.rotation, args.thickness)
    right = probabilities.argmax(axis=1) == digits
    print(tr("Accuracy on the test set: {acc:.1%}  ({right}/{total} photos recognized)",
             acc=right.mean(), right=right.sum(), total=len(digits)))
    for digit in range(10):
        print("  " + tr("digit {digit}: {acc:6.1%}", digit=digit, acc=right[digits == digit].mean()))

    points_map = None
    if args.map:  # instead of the confusion matrix, the map of points of the last hidden layer
        chosen = charts.map_photos(len(digits))
        points, axes = charts.project_2d(net.forward(X[chosen])[-2], args.map)
        points_map = {"points": points, "axes": axes, "name": tr("layer {n}", n=len(net.layers) - 2),
                      "method": args.map, "photos": chosen}
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    charts.evaluation(fig, X.reshape(-1, 28, 28), digits, probabilities, robustness(net, photos, digits),
                      {"noise": args.noise, "rotation": args.rotation},
                      title=tr("Evaluation on the test set: accuracy {acc:.1%}", acc=right.mean()),
                      points_map=points_map)
    print(tr("Chart saved in"), storage.save_chart(fig, "3_evaluation.png"))
    plt.show()


def draw(args):
    """5. Drawing board and lab in a window of their own."""
    if not storage.MODEL_FILE.exists():
        raise SystemExit(tr("No model: train the network first  ({command})", command=f"{LAUNCHER} train"))
    from gui.tab_draw import open_drawing_window
    open_drawing_window()


def reset(args):
    """Deletes model and charts (with --all the photos too). The code is never touched."""
    paths = storage.to_delete(args.all)
    if not paths:
        return print(tr("Nothing to delete: the project is already clean."))
    print(tr("These will be deleted:"), *[f"  - {path}" for path in paths], sep="\n")
    if not args.yes and input(tr("Confirm? [y/N] ")).strip().lower() not in YES:
        return print(tr("Cancelled: nothing was deleted."))
    storage.reset(args.all)
    print(tr("Done: the project is clean again."))


def update(args):
    """Checks on GitHub for a new version and, if you confirm, installs it (data/ is not touched)."""
    release = updater.latest_release(timeout=10)
    if release is None:
        return print(tr("There are no published versions on GitHub yet."))
    if not updater.is_newer(release["version"]):
        return print(tr("You already have the latest version ({version}).", version=VERSION))
    print(tr("Version {new} is available (you have {current}).", new=release["version"], current=VERSION)
          + f"\n\n{release['notes']}\n")
    if updater.installed_with_git():
        return print(tr("This copy was downloaded with git: update it with  git pull"))
    if not args.yes and input(tr("Install it now? [y/N] ")).strip().lower() not in YES:
        return print(tr("Cancelled."))
    if updater.install(release["zip"], lambda fraction: print(
            "\r  " + tr("downloading: {fraction:.0%}", fraction=fraction), end="", flush=True)):
        print("\n  " + tr("installing the new libraries..."))
        updater.install_requirements()
    print("\n" + tr("Done: you now have version {version}.", version=release["version"]))


def main():
    parser = argparse.ArgumentParser(description=tr(
        "{name} v{version}, by {author}: a neural network that recognizes digits. "
        "Without a command it opens the graphical interface.", name=tr(NAME), version=VERSION, author=AUTHOR))
    parser.add_argument("--version", action="version", version=f"{tr(NAME)} {VERSION}")
    commands = parser.add_subparsers(dest="command", title=tr("commands"))

    p = commands.add_parser("download", help=tr("1. downloads the digit photos"))
    p.add_argument("--per-digit", type=int, default=100, help=tr("training photos for each digit"))
    p.add_argument("--test-per-digit", type=int, default=20, help=tr("test photos for each digit"))
    p.add_argument("--all", action="store_true", help=tr("the whole collection: all the 70000 photos of MNIST"))
    p.set_defaults(function=download)

    commands.add_parser("explore", help=tr("2. pre-training: charts about the dataset")).set_defaults(function=explore)

    p = commands.add_parser("train", help=tr("3. trains the network"))
    p.add_argument("--epochs", type=int, default=60, help=tr("how many rounds over all the photos"))
    p.add_argument("--lr", type=float, default=0.05, help=tr("initial learning rate"))
    p.add_argument("--momentum", type=float, default=0.9, help=tr("push of the corrections (0-0.99)"))
    p.add_argument("--l2", type=float, default=1e-4, help=tr("L2 regularization (0 = off)"))
    p.add_argument("--dropout", type=float, default=0.0, help=tr("fraction of neurons switched off at random (0-0.8)"))
    p.add_argument("--batch", type=int, default=32, help=tr("photos for each weight correction"))
    p.add_argument("--noise", type=float, default=0.0, help=tr("gaussian noise on the photos (sigma)"))
    p.add_argument("--rotation", type=float, default=12, help=tr("maximum random rotation (degrees)"))
    p.add_argument("--shift", type=int, default=2, help=tr("maximum random shift (pixels)"))
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help=tr("neurons of the hidden layers"))
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="relu", help=tr("activation function"))
    p.add_argument("--init-scale", type=float, default=1.0, help=tr("width of the initial weights (x)"))
    p.add_argument("--seed", type=int, default=0, help=tr("seed of the random numbers"))
    p.add_argument("--no-plot", action="store_true", help=tr("without live charts"))
    p.set_defaults(function=train)

    p = commands.add_parser("evaluate", help=tr("4. tests the network on the test photos"))
    p.add_argument("--noise", type=float, default=0.0, help=tr("gaussian noise on the test photos (sigma)"))
    p.add_argument("--rotation", type=float, default=0.0, help=tr("rotates the test photos (degrees)"))
    p.add_argument("--thickness", type=int, default=0, choices=range(-2, 3),
                   help=tr("thinner (<0) or thicker (>0) stroke"))
    p.add_argument("--map", choices=["PCA", "t-SNE"], help=tr("map of points instead of the confusion matrix"))
    p.set_defaults(function=evaluate)

    commands.add_parser("draw", help=tr("5. opens only the drawing board and the lab")).set_defaults(function=draw)

    p = commands.add_parser("reset", help=tr("deletes model and charts"))
    p.add_argument("--all", action="store_true", help=tr("deletes the photos too"))
    p.add_argument("--yes", action="store_true", help=tr("do not ask for confirmation"))
    p.set_defaults(function=reset)

    p = commands.add_parser("update", help=tr("checks for a new version and installs it"))
    p.add_argument("--yes", action="store_true", help=tr("installs without asking for confirmation"))
    p.set_defaults(function=update)

    args = parser.parse_args()
    if args.command is None:
        from gui.window import run  # imported only here: the terminal commands do not need it
        run()
    else:
        args.function(args)


if __name__ == "__main__":
    main()

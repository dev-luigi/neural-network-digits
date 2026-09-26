"""
What the assistant knows without looking at the program: the concepts of neural networks (a small glossary),
what every tab shows, an experiment to try in every tab and some questions to suggest.

The texts are functions and not constants: tr() is called when they are needed, in the language in use.
"""
from collections import namedtuple

from i18n import tr

# key: used by the hints (rules.py) to point to their concept; tab: where to see it in the program
Concept = namedtuple("Concept", "key title keywords text tab experiment")
TABS = ("data", "training", "evaluation", "draw", "inside", "info")


def tab_name(tab):
    """The name of a tab, as it is written on the tab itself."""
    return {"data": tr("1 · Data"), "training": tr("2 · Training"), "evaluation": tr("3 · Evaluation"),
            "draw": tr("4 · Draw & edit"), "inside": tr("5 · Inside the network"), "info": tr("Info")}[tab]


def concepts():
    return [
        Concept("network", tr("Neural network"),
                tr("neural network, network, model, how it works, layers, architecture, artificial intelligence"),
                tr("A chain of layers of neurons. The photo goes in as 784 numbers (its pixels), every layer computes "
                   "output = activation(input × weights + bias) and the last layer gives 10 probabilities, one per "
                   "digit: the highest one is the answer. Learning means finding the weights that make the fewest "
                   "mistakes. Here it is written from scratch with NumPy, in neural_net/network.py."),
                "training", tr("Change the neurons of layer 1 and layer 2 and press New network: in tab 4 the "
                               "diagram changes shape.")),
        Concept("neuron", tr("Neuron, weights and bias"),
                tr("neuron, neurons, weight, weights, bias, weighted sum, z, connection, parameters"),
                tr("A neuron multiplies every input by a weight, adds everything up and adds the bias: that is the "
                   "weighted sum z. Then the activation function decides what it sends forward. The weights say how "
                   "much each connection counts, the bias moves the threshold at which the neuron switches on. "
                   "Weights and biases are the numbers the network learns."),
                "draw", tr("In tab 4 click a neuron: you see z, its output and its weights, and you can shift its "
                           "bias or switch it off.")),
        Concept("activation", tr("Activation function"),
                tr("activation, activation function, relu, leaky relu, sigmoid, tanh, non-linearity"),
                tr("What a hidden neuron does with its weighted sum z. relu: max(0, z), simple and fast. leaky relu: "
                   "like relu, but lets 0.01·z through, so the neurons do not die. sigmoid: squashes between 0 and 1 "
                   "and learns more slowly. tanh: squashes between -1 and 1. Without it the whole network would be a "
                   "single multiplication and could not learn curved shapes."),
                "training", tr("Train the same network with relu and with sigmoid, with the same seed, and compare "
                               "the loss curves.")),
        Concept("softmax", tr("Softmax"),
                tr("softmax, probability, probabilities, output, answer, scores, exponential, percentages"),
                tr("The last layer turns its 10 scores z into 10 probabilities: it takes the exponential of each one "
                   "and divides it by the sum of all of them, so they are all positive and add up to 100%. The "
                   "answer of the network is the digit with the highest probability."),
                "inside", tr("In tab 5 choose the output layer: you see the softmax step by step, cell by cell.")),
        Concept("temperature", tr("Temperature"),
                tr("temperature, T, sure, uncertain, confidence, sharper, flatter"),
                tr("The scores are divided by T before the softmax. T below 1 makes the answer sharper (more sure), "
                   "T above 1 flattens the probabilities towards 10% each. The chosen digit does not change, the "
                   "confidence does. It is the same temperature of language models."),
                "inside", tr("In tab 5 press Next mistake and move the temperature: how sure is the network of its "
                             "mistake?")),
        Concept("loss", tr("Loss"),
                tr("loss, cross-entropy, error, cost, loss curve, how wrong"),
                tr("How wrong the network is: the cross-entropy, that is minus the logarithm of the probability given "
                   "to the right digit, averaged over the photos. 0 = perfect, about 2.3 = guessing at random among 10 "
                   "digits. Training means making it go down. It also counts how sure the network was: a wrong "
                   "answer given with 99% confidence costs a lot."),
                "training", ""),
        Concept("accuracy", tr("Accuracy"),
                tr("accuracy, percentage, correct, right answers, precision, result, score, how good"),
                tr("The percentage of photos recognized. The one that matters is on photos the network has never "
                   "learned from: validation (during training) and test (tab 3). With the starting settings it gets "
                   "to about 91%, with the whole collection to about 98.5%."),
                "evaluation", ""),
        Concept("epoch", tr("Epoch"),
                tr("epoch, epochs, pass, round, iteration, how long, duration"),
                tr("One full pass over all the training photos. At every epoch the photos are rotated and shifted a "
                   "little, the network corrects its weights after every mini-batch, and at the end the loss and the "
                   "accuracy are measured. More epochs = more learning, until it stops improving or starts learning "
                   "by heart."),
                "training", ""),
        Concept("batch", tr("Mini-batch"),
                tr("batch, mini-batch, batch size, photos per mini-batch, group, stochastic"),
                tr("How many photos the network looks at before correcting its weights. Small batches give many "
                   "noisy corrections, big batches few precise ones. 32 is a good compromise."),
                "training", ""),
        Concept("learning rate", tr("Learning rate"),
                tr("learning rate, lr, step, step size, speed, how fast it learns"),
                tr("How big each correction of the weights is. Too low: it learns very slowly. Too high: the loss "
                   "jumps up and down, many neurons switch off forever, or the weights become infinite and the "
                   "network explodes. Good values here are between 0.01 and 0.1."),
                "training", tr("Set it to 1, press New network and Start: watch the loss explode. Then try 0.0001: "
                               "it barely moves.")),
        Concept("cosine", tr("Cosine decay"),
                tr("cosine, cosine decay, schedule, slow down towards the end, learning rate goes down"),
                tr("The learning rate goes down smoothly from the chosen value to zero during the run of epochs, "
                   "shaped like a cosine: big steps at the beginning to get close quickly, small steps at the end to "
                   "fix the details. It usually gives a few more points of accuracy."),
                "training", ""),
        Concept("momentum", tr("Momentum"),
                tr("momentum, inertia, push, velocity, oscillates"),
                tr("Every correction keeps part of the direction of the previous ones: with 0.9 it keeps 90%. Like a "
                   "ball rolling downhill, it goes faster in the right direction and smooths out the zigzags. Too "
                   "close to 1 and it overshoots and oscillates."),
                "training", ""),
        Concept("backpropagation", tr("Backpropagation and gradient descent"),
                tr("backpropagation, backprop, gradient, gradients, gradient descent, sgd, derivative, how it "
                   "learns, corrections"),
                tr("How the network learns. It compares its answer with the right one, then goes backwards layer by "
                   "layer and computes how much every weight contributed to the error: the gradient. Every weight is "
                   "then moved a little in the direction that reduces the error: that is gradient descent. The chart "
                   "of the corrections per layer shows how big these gradients are."),
                "training", ""),
        Concept("overfitting", tr("Overfitting"),
                tr("overfitting, learning by heart, memorizing, validation gets worse, validation loss goes up, "
                   "gap between train and validation, generalize"),
                tr("The network learns the training photos by heart instead of the shape of the digits: the train "
                   "accuracy keeps improving while the validation one stops or gets worse, and the validation loss "
                   "goes up. Remedies: more photos, data augmentation, dropout, L2, or fewer epochs."),
                "training", tr("Download 10 photos per digit, put rotation and shift at 0 and train for 200 epochs: "
                               "watch train and validation separate.")),
        Concept("underfitting", tr("Underfitting"),
                tr("underfitting, not learning, does not learn, too simple, low accuracy, stuck, does not improve"),
                tr("The network does not manage to learn even the training photos: train and validation accuracy both "
                   "stay low. It happens when the network is too small, the learning rate is too low, L2 or dropout "
                   "are too strong, or it has done too few epochs."),
                "training", ""),
        Concept("validation", tr("Train, validation and test"),
                tr("validation, test, test set, train set, training photos, split, never seen, difference"),
                tr("The training photos are split: 90% to learn from and 10% (validation) to check, epoch after "
                   "epoch, that the network is not learning by heart. The test photos are downloaded separately and "
                   "used only in tabs 3, 4 and 5: they measure how the network copes with digits it has never seen."),
                "data", ""),
        Concept("l2", tr("L2 regularization"),
                tr("l2, weight decay, regularization, small weights, penalty"),
                tr("At every step all the weights are pushed a little towards zero. Small weights make a simpler "
                   "network, which usually generalizes better. Too much L2 prevents it from learning: the gaussians "
                   "of the weights shrink towards zero."),
                "training", ""),
        Concept("dropout", tr("Dropout"),
                tr("dropout, switches off neurons at random, regularization"),
                tr("At every step a random part of the hidden neurons is switched off, so the network learns not to "
                   "depend on a few neurons. The train loss goes up a bit, but the validation often improves. When the "
                   "network answers, no neuron is switched off. 20% is a common value."),
                "training", tr("Train with dropout at 0% and then at 30%: compare the gap between train and "
                               "validation.")),
        Concept("augmentation", tr("Data augmentation"),
                tr("data augmentation, augmentation, rotation, shift, rotate, move, varied photos"),
                tr("At every epoch every training photo is rotated and shifted a little, at random. The network sees "
                   "digits that are always a bit different, so it learns the shape and not the single pixels, and it "
                   "holds up better with tilted or off-center digits."),
                "training", tr("In tab 3 rotate the test photos by 30°. Then train with maximum rotation 30° and "
                               "compare.")),
        Concept("noise", tr("Gaussian noise"),
                tr("noise, gaussian noise, gaussian, sigma, dirty, grain, robustness"),
                tr("A random number drawn from a gaussian of width σ is added to every pixel. During training a little "
                   "noise makes the network more robust; in tab 3 it spoils the test photos, to see how well the "
                   "network holds up; in tab 4 it dirties the drawing."),
                "evaluation", ""),
        Concept("initialization", tr("Initial weights"),
                tr("initialization, initial weights, width of the initial weights, he, lecun, exploding, vanishing, "
                   "signal, gaussians"),
                tr("Before learning, the weights are random numbers drawn from a gaussian. Its width matters: x1 is "
                   "the right choice (He for relu, LeCun for the others). Much narrower (x0.1) and the signal fades "
                   "out layer after layer; much wider (x5) and it explodes. Look at the gaussians and the corrections "
                   "per layer in tab 2."),
                "training", tr("Set the width to x0.1, then to x5: press New network and Start, and compare the "
                               "gaussians.")),
        Concept("inactive", tr("Inactive neurons"),
                tr("inactive, dead, dying relu, switched off, always zero, useless"),
                tr("A neuron is inactive when it gives the same output whatever photo it sees, so it is useless. With "
                   "relu it happens when its weighted sum stays negative for every photo: it always gives 0 and no "
                   "longer learns. It is caused above all by a learning rate that is too high. Leaky relu avoids it."),
                "training", ""),
        Concept("exploded", tr("Exploded network"),
                tr("exploded, explodes, nan, infinite, infinity, diverges, broken"),
                tr("The weights have become infinite: the corrections were so big that every step made things worse. "
                   "It happens with a learning rate that is too high, sometimes together with a very high momentum or "
                   "very wide initial weights. Lower the learning rate and press New network."),
                "training", ""),
        Concept("confusion", tr("Confusion matrix"),
                tr("confusion matrix, matrix, confused, mistakes per digit, rows, columns, diagonal"),
                tr("A table that counts, for every true digit (row), how many photos ended up in each answer "
                   "(column). The diagonal holds the right answers; the other cells show which digits get confused, "
                   "like 4 with 9 or 3 with 5."),
                "evaluation", ""),
        Concept("map", tr("Point map (PCA and t-SNE)"),
                tr("point map, map, pca, t-sne, tsne, embedding, projection, clusters, groups"),
                tr("Every test photo becomes a point on a plane: the photos that the network sees as similar end up "
                   "close together. PCA looks at the cloud from the side where it is most spread out; t-SNE moves the "
                   "points to keep the neighbors together and separates the groups better. Layer after layer, the "
                   "digits separate."),
                "evaluation", tr("In tab 3 switch to the point map and click the layers one after the other: pixels, "
                                 "layer 1, layer 2, output.")),
        Concept("threshold", tr("Confidence threshold"),
                tr("threshold, confidence threshold, confidence, sure, I don't know, abstains"),
                tr("The network answers only when its highest probability exceeds the threshold, otherwise it says "
                   "\"I don't know\". Raising it, it answers fewer photos but makes fewer mistakes: this is how "
                   "networks are used when a mistake is costly."),
                "evaluation", ""),
        Concept("mnist", tr("MNIST"),
                tr("mnist, dataset, data, photos, digits, download, handwritten, 28x28, whole collection"),
                tr("The dataset of the photos: 70,000 handwritten digits of 28×28 pixels in shades of grey (60,000 for "
                   "training and 10,000 for test), by Yann LeCun, Corinna Cortes and Christopher Burges. It is the "
                   "\"hello world\" of neural networks."),
                "data", ""),
        Concept("pruning", tr("Pruning"),
                tr("pruning, prune, remove weights, weights at zero, compress, smaller network"),
                tr("Setting the smallest weights to zero. Many networks keep working even without most of their "
                   "weights: in the lab of tab 4 you can see how far yours holds up before the accuracy collapses."),
                "draw", ""),
        Concept("seed", tr("Seed"),
                tr("seed, random, randomness, repeatable, reproducible, luck"),
                tr("The starting number of the random numbers: it decides the initial weights and the order of the "
                   "photos. Same seed = same experiment, repeatable. Change it to see how much luck matters."),
                "training", ""),
        Concept("assistant", tr("The assistant"),
                tr("assistant, pick, hints, suggestions, help, questions, how do you work, search, tf-idf"),
                tr("I answer questions about the program and about neural networks, I see the state of the tab you "
                   "are in and, if Hints is on, I tell you when something looks wrong. With Pick (F1) move the mouse "
                   "over the controls or the charts and click one: I explain what it does and what it is worth now. "
                   "I am not a language model: I compare the words of your question with my texts (TF-IDF, a small "
                   "search engine written with NumPy, in assistant/search.py)."),
                "info", ""),
        Concept("language", tr("Language of the program"),
                tr("language, change the language, english, italian, translation, translated"),
                tr("The program speaks English and Italian: choose EN or IT at the top right, next to Pick. The "
                   "language is applied when the program starts again, and it offers to restart it right away. I "
                   "understand the questions in both languages, whatever the one in use."),
                "info", ""),
    ]


def tab_description(tab):
    """What a tab shows, in a few lines."""
    return {
        "data": tr("Here you download the MNIST photos and look at them before training: some examples, how many "
                   "photos there are per digit, the \"average digit\" and how the pixel values are spread out. From "
                   "here you can also start the project again from scratch."),
        "training": tr("Here the network learns. On the left the architecture, the optimization knobs (they can be "
                       "changed during training too) and the alterations of the photos; on the right the numbers of "
                       "the last epoch, the loss curve, the accuracy, the corrections per layer and the gaussians of "
                       "the weights."),
        "evaluation": tr("How the network does on test photos it has never seen. You can spoil the photos (noise, "
                         "rotation, stroke thickness), set a confidence threshold and switch the big chart between "
                         "the confusion matrix and the point map."),
        "draw": tr("Draw a digit and watch the neurons light up. Click a neuron to see its values and change it "
                   "(bias, weights, switched off); the global knobs (temperature, noise on the weights, pruning) show "
                   "their effect on the test accuracy right away."),
        "inside": tr("The math of one layer, number by number, for a test photo: inputs, weight matrix, bias, "
                     "weighted sum, activation and, in the last layer, the steps of the softmax. Move the mouse over "
                     "the cells to see the math."),
        "info": tr("Version, author, license, updates and the changelog of every version. The language is chosen "
                   "at the top right, next to Pick: EN or IT."),
    }[tab]


def experiment(tab):
    """Something to try in a tab, once the network is trained."""
    return {
        "data": tr("Download 500 photos per digit and train again: the accuracy goes up, and every epoch takes "
                   "longer."),
        "training": tr("Set dropout to 30% and train again: compare the gap between train and validation. Or set the "
                       "learning rate to 1 and watch the loss explode."),
        "evaluation": tr("Rotate the test photos by 30°: how much does the accuracy drop? Then train with maximum "
                         "rotation 30° and compare."),
        "draw": tr("While you draw, switch off the most active neuron of the last hidden layer: does the answer "
                   "change? Then push the pruning to 90%."),
        "inside": tr("Press Next mistake and look at the softmax: which digits compete for the answer? Then move the "
                     "temperature."),
        "info": tr("Go back to tab 2 · Training and try an experiment: every knob has its explanation."),
    }[tab]


def suggestions(tab):
    """Questions to suggest in a tab: they are shown as buttons below the chat."""
    return {
        "data": [tr("What should I do now?"), tr("What is MNIST?"), tr("Validation and test: what is the "
                                                                          "difference?")],
        "training": [tr("How is it going?"), tr("Is something wrong?"), tr("What is the learning rate?"),
                     tr("What is overfitting?")],
        "evaluation": [tr("How is it going?"), tr("What is the confusion matrix?"), tr("What is t-SNE?"),
                       tr("What is the confidence threshold?")],
        "draw": [tr("What should I do now?"), tr("What is a neuron?"), tr("What does the temperature do?"),
                 tr("What is pruning?")],
        "inside": [tr("What should I do now?"), tr("What is the softmax?"), tr("What is the bias?")],
        "info": [tr("What can you do?"), tr("What should I do now?")],
    }[tab]

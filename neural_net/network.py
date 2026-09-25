"""
The neural network, written from scratch with NumPy (no artificial intelligence libraries).

How it works, in short:
  1. The photo goes in as 784 numbers: its pixels, from 0 (black) to 1 (white).
  2. Each layer computes:  output = activation(input x weights + bias)
     The "weights" are the numbers the network learns: they say how much each connection counts.
  3. The last layer gives 10 probabilities, one per digit: the highest one is the answer.
  4. To learn, the network compares its answer with the right one and corrects
     all the weights a tiny bit so it makes fewer mistakes (backpropagation).
"""
import numpy as np

from neural_net.storage import MODEL_FILE

# The activation functions of the hidden neurons, each one with its derivative.
# The derivative (how much the output changes if the input changes a tiny bit) is needed to learn;
# here it is written using the neuron's output "a", which we have already computed.
ACTIVATIONS = {
    "relu": (lambda z: np.maximum(0, z), lambda a: (a > 0).astype(np.float32)),
    "leaky relu": (lambda z: np.where(z > 0, z, 0.01 * z), lambda a: np.where(a > 0, 1, 0.01).astype(np.float32)),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda a: a * (1 - a)),
    "tanh": (np.tanh, lambda a: 1 - a * a),
}


def softmax(z):
    """Turns 10 scores into 10 probabilities (positive and adding up to 1)."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))  # subtracting the maximum avoids huge numbers
    return e / e.sum(axis=-1, keepdims=True)


def softmax_steps(z, temperature=1.0):
    """The same math as softmax() for one photo, one step at a time, plus the "temperature" T:
    T < 1 makes the answer sharper, T > 1 more uncertain (with T = 1 it is exactly the network).
    Returns the three steps: (z - max) / T,  e = exponential,  p = e / sum(e)."""
    shifted = (z - z.max()) / temperature
    e = np.exp(shifted)
    return shifted, e, e / e.sum()


class NeuralNetwork:
    def __init__(self, layers=(784, 64, 32, 10), activation="relu", init_scale=1.0, seed=0):
        """layers = how many neurons each layer has, from the input (784 pixels) to the output (10 digits).
        activation = one of ACTIVATIONS. init_scale = how wide the random starting weights are."""
        rng = np.random.default_rng(seed)
        self.layers = tuple(int(n) for n in layers)
        self.activation = str(activation)
        pairs = list(zip(self.layers[:-1], self.layers[1:]))  # e.g. (784, 64), (64, 32), (32, 10)
        # Random starting weights, drawn from a gaussian as wide as scale * sqrt(gain / inputs):
        # it is the "He" initialization for ReLUs (gain 2) and "Xavier" for the others (gain 1)
        gain = 2 if "relu" in self.activation else 1
        self.weights = [rng.normal(0, init_scale * np.sqrt(gain / n_in), (n_in, n_out)).astype(np.float32)
                        for n_in, n_out in pairs]
        self.biases = [np.zeros(n_out, np.float32) for _, n_out in pairs]
        # Momentum: remembers the direction of the previous corrections, so the network learns faster
        self.weight_velocity = [np.zeros_like(W) for W in self.weights]
        self.bias_velocity = [np.zeros_like(b) for b in self.biases]
        self.gradient_norms = [0.0] * len(self.weights)  # how big the last corrections of each layer were

    @property
    def n_parameters(self):
        """How many numbers the network has to learn (all the weights and all the biases)."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def forward(self, X):
        """Passes the photos through the network, one layer after the other.
        Returns the outputs of all the layers: the first is the photo, the last the probabilities."""
        activate, _ = ACTIVATIONS[self.activation]
        activations = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b  # the "weighted sum" of each neuron
            last_layer = i == len(self.weights) - 1
            activations.append(softmax(z) if last_layer else activate(z))
        return activations

    def predict(self, X):
        """Only the final probabilities: one row per photo, one column per digit."""
        return self.forward(X)[-1]

    @staticmethod
    def loss(probabilities, Y):
        """How wrong the network is (cross-entropy): 0 = perfect, the higher the worse."""
        return float(-np.mean(np.sum(Y * np.log(probabilities + 1e-9), axis=1)))

    def train_step(self, X, Y, lr, momentum=0.9, l2=1e-4, dropout=0.0, rng=None):
        """One learning step on a small group of photos (mini-batch). Returns the loss.
          lr        learning rate: how big the weight corrections are
          momentum  how much "push" the previous corrections keep (0 = none)
          l2        pushes all the weights towards zero, to keep them small (regularization)
          dropout   fraction of hidden neurons switched off at random in this step (regularization)"""
        activate, derivative = ACTIVATIONS[self.activation]

        # 1. "Training" forward: like forward(), but it remembers the derivatives and applies dropout
        activations, derivatives = [X], []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            if i == len(self.weights) - 1:
                activations.append(softmax(z))
            else:
                a = activate(z)
                d = derivative(a)
                if dropout > 0:
                    # Switches off some neurons at random and boosts the others, so the average sum stays the same
                    mask = (rng.random(a.shape) >= dropout).astype(np.float32) / (1 - dropout)
                    a, d = a * mask, d * mask
                activations.append(a)
                derivatives.append(d)
        loss = self.loss(activations[-1], Y)

        # 2. Backpropagation: from the last layer to the first, we compute how much each weight caused the error
        error = (activations[-1] - Y) / len(X)  # error on the output: given probabilities - right answer
        for i in reversed(range(len(self.weights))):
            gradient_W = activations[i].T @ error + l2 * self.weights[i]
            gradient_b = error.sum(axis=0)
            self.gradient_norms[i] = float(np.linalg.norm(gradient_W))
            if i > 0:
                # The error goes back to the previous layer, weighted by how "sensitive" each neuron was
                error = (error @ self.weights[i].T) * derivatives[i - 1]
            # 3. Correction: a small step in the direction that reduces the error (gradient descent)
            self.weight_velocity[i] = momentum * self.weight_velocity[i] - lr * gradient_W
            self.bias_velocity[i] = momentum * self.bias_velocity[i] - lr * gradient_b
            self.weights[i] += self.weight_velocity[i]
            self.biases[i] += self.bias_velocity[i]
        return loss

    def save(self, path=MODEL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, layers=self.layers, activation=self.activation,
                 **{f"W{i}": W for i, W in enumerate(self.weights)},
                 **{f"b{i}": b for i, b in enumerate(self.biases)})

    @classmethod
    def load(cls, path=MODEL_FILE):
        with np.load(path) as file:  # "with" closes the file right away (on Windows it is needed to delete it)
            net = cls(file["layers"], str(file["activation"]))
            net.weights = [file[f"W{i}"] for i in range(len(net.weights))]
            net.biases = [file[f"b{i}"] for i in range(len(net.biases))]
        return net

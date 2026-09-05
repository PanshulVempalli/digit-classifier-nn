"""
A feedforward neural network implemented from scratch with NumPy --
no PyTorch, no TensorFlow, no scikit-learn model classes. Forward pass,
backward pass (backpropagation), and the training update are all
written out explicitly, because the point of this project is to
demonstrate understanding of how a neural network actually works, not
to call `.fit()` on someone else's implementation.

Architecture: one hidden layer, ReLU activation, softmax output,
categorical cross-entropy loss.

    X (n, 64) -> [W1, b1] -> ReLU -> [W2, b2] -> softmax -> probs (n, 10)

Backprop derivation note: for softmax output + cross-entropy loss
combined, the gradient of the loss w.r.t. the pre-softmax logits Z2
simplifies to exactly (softmax(Z2) - one_hot(y)) -- this is a standard
result (the two Jacobians cancel neatly) and is why dZ2 below looks
simpler than you might expect from differentiating softmax and
cross-entropy separately.
"""

from dataclasses import dataclass

import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    # subtract row-wise max for numerical stability (doesn't change the
    # result, since softmax is shift-invariant, but avoids overflow)
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(y, n_classes):
    encoded = np.zeros((len(y), n_classes))
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def cross_entropy_loss(probs, y_true_onehot, epsilon=1e-12):
    n = probs.shape[0]
    clipped = np.clip(probs, epsilon, 1 - epsilon)
    return -np.sum(y_true_onehot * np.log(clipped)) / n


@dataclass
class NeuralNetwork:
    input_dim: int
    hidden_dim: int
    output_dim: int
    weight_decay: float = 0.0
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # He initialisation -- scaled for ReLU units, avoids vanishing/
        # exploding activations at the start of training
        self.W1 = rng.normal(0, np.sqrt(2.0 / self.input_dim), size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_dim), size=(self.hidden_dim, self.output_dim))
        self.b2 = np.zeros(self.output_dim)

    def get_params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = softmax(z2)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "probs": probs}
        return probs, cache

    def loss(self, X, y):
        probs, _ = self.forward(X)
        y_onehot = one_hot(y, self.output_dim)
        data_loss = cross_entropy_loss(probs, y_onehot)
        reg_loss = 0.5 * self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss

    def backward(self, cache, y):
        X, a1, z1, probs = cache["X"], cache["a1"], cache["z1"], cache["probs"]
        n = X.shape[0]
        y_onehot = one_hot(y, self.output_dim)

        dz2 = (probs - y_onehot) / n  # softmax + cross-entropy combined gradient
        dW2 = a1.T @ dz2 + self.weight_decay * self.W2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = X.T @ dz1 + self.weight_decay * self.W1
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, learning_rate):
        self.W1 -= learning_rate * grads["W1"]
        self.b1 -= learning_rate * grads["b1"]
        self.W2 -= learning_rate * grads["W2"]
        self.b2 -= learning_rate * grads["b2"]

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

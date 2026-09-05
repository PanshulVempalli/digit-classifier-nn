"""
Mini-batch gradient descent training loop for the from-scratch
NeuralNetwork, plus a matching "baseline" trainer -- a plain (no hidden
layer) softmax classifier, i.e. multinomial logistic regression -- used
in evaluate.py to check the hidden layer is actually earning its keep.
"""

import numpy as np

from neural_net import NeuralNetwork


def train(model: NeuralNetwork, X_train, y_train, X_val, y_val, epochs=200, batch_size=32, learning_rate=0.5, verbose=False):
    n = X_train.shape[0]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    rng = np.random.default_rng(1)
    for epoch in range(epochs):
        order = rng.permutation(n)
        X_shuffled, y_shuffled = X_train[order], y_train[order]

        for start in range(0, n, batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            probs, cache = model.forward(X_batch)
            grads = model.backward(cache, y_batch)
            model.step(grads, learning_rate)

        train_loss = model.loss(X_train, y_train)
        val_loss = model.loss(X_val, y_val)
        train_acc = accuracy(model.predict(X_train), y_train)
        val_acc = accuracy(model.predict(X_val), y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    return history


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))

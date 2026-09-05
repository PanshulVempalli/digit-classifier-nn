import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from neural_net import NeuralNetwork, softmax, relu, one_hot, cross_entropy_loss
from data import load_digit_data
from train import train, accuracy
from evaluate import confusion_matrix, precision_recall_per_class, majority_class_baseline_accuracy


def test_forward_pass_shapes():
    model = NeuralNetwork(input_dim=64, hidden_dim=16, output_dim=10, seed=0)
    X = np.random.default_rng(0).normal(size=(5, 64))
    probs, cache = model.forward(X)
    assert probs.shape == (5, 10)
    assert np.allclose(probs.sum(axis=1), 1.0), "softmax outputs should sum to 1 per row"
    assert (probs >= 0).all(), "probabilities should be non-negative"


def test_softmax_and_relu_basics():
    assert np.allclose(relu(np.array([-2.0, -0.1, 0.0, 3.0])), [0, 0, 0, 3.0])
    probs = softmax(np.array([[1.0, 1.0, 1.0]]))
    assert np.allclose(probs, [[1 / 3, 1 / 3, 1 / 3]])


def test_one_hot_and_cross_entropy():
    y = np.array([0, 2])
    onehot = one_hot(y, n_classes=3)
    assert np.array_equal(onehot, [[1, 0, 0], [0, 0, 1]])

    # A perfect prediction should give ~zero loss
    perfect_probs = onehot.copy()
    assert cross_entropy_loss(perfect_probs, onehot) < 1e-6

    # A confidently wrong prediction should give a large loss
    wrong_probs = np.array([[0.001, 0.001, 0.998], [0.998, 0.001, 0.001]])
    assert cross_entropy_loss(wrong_probs, onehot) > 3.0


def test_backprop_gradients_match_numerical_gradients():
    """The gold-standard check for a hand-written backprop implementation:
    compare the analytical gradient against a numerically estimated one
    (central finite differences) for a sample of individual parameter
    entries. If these disagree, the backprop maths is wrong, no matter
    how plausible the code looks."""
    rng = np.random.default_rng(3)
    model = NeuralNetwork(input_dim=5, hidden_dim=4, output_dim=3, weight_decay=0.1, seed=2)
    X = rng.normal(size=(6, 5))
    y = rng.integers(0, 3, size=6)

    probs, cache = model.forward(X)
    analytical_grads = model.backward(cache, y)

    epsilon = 1e-5
    checked_any = False

    for param_name in ["W1", "b1", "W2", "b2"]:
        param = getattr(model, param_name)
        flat_indices = list(np.ndindex(param.shape))[:6]  # sample a handful of entries, not all (keeps the test fast)

        for idx in flat_indices:
            original_value = param[idx]

            param[idx] = original_value + epsilon
            loss_plus = model.loss(X, y)

            param[idx] = original_value - epsilon
            loss_minus = model.loss(X, y)

            param[idx] = original_value  # restore

            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            # backward() already returns the gradient of the *mean* loss
            # (dz2 = (probs - y_onehot) / n carries the 1/n through every
            # downstream gradient), so this should compare directly
            # against the numerical gradient of model.loss() with no
            # further rescaling.
            analytical_grad = analytical_grads[param_name][idx]

            denom = max(abs(numerical_grad), abs(analytical_grad), 1e-8)
            relative_error = abs(numerical_grad - analytical_grad) / denom
            assert relative_error < 1e-2, (
                f"{param_name}{idx}: analytical={analytical_grad:.6f} "
                f"numerical={numerical_grad:.6f} relative_error={relative_error:.4f}"
            )
            checked_any = True

    assert checked_any


def test_training_loss_decreases_on_real_data():
    """Sanity check that the training loop is wired up correctly: loss
    on the real digits training data should go down, not stay flat or
    diverge, over a short training run."""
    X_train, y_train, X_test, y_test = load_digit_data()
    model = NeuralNetwork(input_dim=64, hidden_dim=16, output_dim=10, seed=0)

    initial_loss = model.loss(X_train, y_train)
    history = train(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, learning_rate=0.5)
    final_loss = history["train_loss"][-1]

    assert final_loss < initial_loss * 0.5, "loss should drop substantially within 10 epochs"


def test_trained_model_beats_majority_baseline_on_real_held_out_digits():
    """The real test: trained on the real scikit-learn digits dataset
    (1,797 actual handwritten digit scans), does the from-scratch network
    generalise to held-out test data, and does it actually learn
    something about the input rather than just memorising class
    frequencies?"""
    X_train, y_train, X_test, y_test = load_digit_data(test_fraction=0.2, seed=0)
    baseline_acc = majority_class_baseline_accuracy(y_train, y_test)

    model = NeuralNetwork(input_dim=64, hidden_dim=32, output_dim=10, weight_decay=1e-4, seed=0)
    train(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.5)

    test_acc = accuracy(model.predict(X_test), y_test)

    assert test_acc > baseline_acc + 0.5, "should massively beat the ~8% majority-class baseline"
    assert test_acc > 0.90, "a small network should comfortably exceed 90% on this dataset"


def test_confusion_matrix_and_precision_recall():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred, n_classes=3)

    expected = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
    ])
    assert np.array_equal(cm, expected)

    precision, recall = precision_recall_per_class(cm)
    # class 1: predicted 3 times, 2 correct -> precision = 2/3
    assert abs(precision[1] - 2 / 3) < 1e-9
    # class 1: actually occurs twice, both correctly predicted -> recall = 1.0
    assert abs(recall[1] - 1.0) < 1e-9
    # class 0: actually occurs twice, 1 correctly predicted -> recall = 0.5
    assert abs(recall[0] - 0.5) < 1e-9

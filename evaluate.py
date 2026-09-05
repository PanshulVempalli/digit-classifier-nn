"""
Evaluation metrics and baselines -- computed by hand (no sklearn.metrics),
so the numbers reported are traceable to explicit code, not a library
black box.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def precision_recall_per_class(cm: np.ndarray):
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    for c in range(n_classes):
        true_positives = cm[c, c]
        predicted_positives = cm[:, c].sum()
        actual_positives = cm[c, :].sum()
        precision[c] = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall[c] = true_positives / actual_positives if actual_positives > 0 else 0.0
    return precision, recall


def majority_class_baseline_accuracy(y_train, y_test):
    """The simplest possible baseline: always predict the most common
    class in the training set. Any real model needs to comfortably beat
    this, or it isn't learning anything about the input."""
    majority_class = np.bincount(y_train).argmax()
    predictions = np.full_like(y_test, fill_value=majority_class)
    return float(np.mean(predictions == y_test))

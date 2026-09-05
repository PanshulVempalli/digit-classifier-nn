"""
Trains the from-scratch neural network on the real scikit-learn digits
dataset, reports accuracy against a majority-class baseline, and saves
two plots: the training/validation curves, and the confusion matrix.

Usage:
    python main.py --hidden-dim 32 --epochs 100 --out-dir .
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_digit_data
from neural_net import NeuralNetwork
from train import train, accuracy
from evaluate import confusion_matrix, precision_recall_per_class, majority_class_baseline_accuracy


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion Matrix — Held-out Test Digits")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="From-scratch neural network digit classifier")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_digit_data(test_fraction=0.2, seed=0)
    print(f"Loaded {len(X_train)} training and {len(X_test)} test digits (real scikit-learn digits dataset).")

    baseline_acc = majority_class_baseline_accuracy(y_train, y_test)
    print(f"Majority-class baseline accuracy: {baseline_acc:.3f}")

    model = NeuralNetwork(
        input_dim=64, hidden_dim=args.hidden_dim, output_dim=10,
        weight_decay=args.weight_decay, seed=0,
    )
    history = train(
        model, X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        verbose=True,
    )

    y_pred = model.predict(X_test)
    test_acc = accuracy(y_pred, y_test)
    print(f"\nFinal test accuracy: {test_acc:.3f} (vs. {baseline_acc:.3f} baseline)")

    cm = confusion_matrix(y_test, y_pred, n_classes=10)
    precision, recall = precision_recall_per_class(cm)
    print("\nPer-class precision / recall:")
    for digit in range(10):
        print(f"  digit {digit}: precision={precision[digit]:.3f}  recall={recall[digit]:.3f}")

    plot_training_curves(history, f"{args.out_dir}/training_curves.png")
    plot_confusion_matrix(cm, f"{args.out_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()

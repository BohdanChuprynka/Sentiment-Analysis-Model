"""Model evaluation utilities for sentiment analysis."""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

COLORS = {
    "primary": "#2563EB",
    "positive": "#10B981",
    "negative": "#EF4444",
    "bg": "#FFFFFF",
    "text": "#1E293B",
}


def calculate_results(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1 score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def print_classification_report(y_true, y_pred, model_name):
    """Print a formatted classification report."""
    print(f"\n{'=' * 50}")
    print(f"Classification Report — {model_name}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    labels = ["Negative", "Positive"]
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           ylabel="True Label", xlabel="Predicted Label")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.show()
    plt.close(fig)


def plot_training_history(history, model_name, save_path=None):
    """Plot training & validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(f"Training History — {model_name}", fontsize=15, fontweight="bold", y=1.02)

    epochs = range(1, len(history.history["loss"]) + 1)

    ax1.plot(epochs, history.history["loss"], color=COLORS["primary"], linewidth=2, label="Train")
    ax1.plot(epochs, history.history["val_loss"], color=COLORS["negative"], linewidth=2, linestyle="--", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(COLORS["bg"])

    ax2.plot(epochs, history.history["accuracy"], color=COLORS["primary"], linewidth=2, label="Train")
    ax2.plot(epochs, history.history["val_accuracy"], color=COLORS["positive"], linewidth=2, linestyle="--", label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(COLORS["bg"])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.show()
    plt.close(fig)

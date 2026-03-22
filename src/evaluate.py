"""Model evaluation utilities for sentiment analysis."""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_results(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1 score for a binary classifier.

    Args:
        y_true: True labels (1D array).
        y_pred: Predicted labels (1D array).

    Returns:
        Dictionary with accuracy, precision, recall, and f1 (all 0–1 scale).
    """
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

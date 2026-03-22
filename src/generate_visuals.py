"""Generate portfolio-quality visuals for the Sentiment Analysis project.

Run after training all models and collecting results.
Produces images saved to the images/ directory.

Usage:
    python src/generate_visuals.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 12

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#0EA5E9",
    "positive": "#10B981",
    "negative": "#EF4444",
    "bg": "#FFFFFF",
    "text": "#1E293B",
}
MODEL_COLORS = ["#2563EB", "#7C3AED", "#0EA5E9"]


def generate_model_comparison(results_dict, save_path=None):
    """
    Generate a grouped bar chart comparing model metrics.

    Args:
        results_dict: Dict of {model_name: {metric: value}} for each model.
        save_path: Path to save the image. Defaults to images/model_comparison.png.
    """
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, "model_comparison.png")

    df = pd.DataFrame(results_dict).T
    metrics = df.columns.tolist()
    models = df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for i, (model, color) in enumerate(zip(models, MODEL_COLORS)):
        values = df.loc[model].values
        bars = ax.bar(x + i * width, values, width, label=model, color=color,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                    color=COLORS["text"], fontweight="medium")

    ax.set_xlabel("Metric", fontsize=13, color=COLORS["text"], labelpad=10)
    ax.set_ylabel("Score", fontsize=13, color=COLORS["text"], labelpad=10)
    ax.set_title("Sentiment Analysis — Model Performance Comparison",
                 fontsize=16, fontweight="bold", color=COLORS["text"], pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics],
                       fontsize=11, color=COLORS["text"])
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.tick_params(colors=COLORS["text"])
    ax.yaxis.grid(True, alpha=0.3, color="#CBD5E1", zorder=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_thumbnail(results_dict, save_path=None):
    """
    Generate a 16:9 portfolio thumbnail with model comparison.

    Args:
        results_dict: Dict of {model_name: {metric: value}}.
        save_path: Path to save. Defaults to images/project_thumbnail.png.
    """
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, "project_thumbnail.png")

    df = pd.DataFrame(results_dict).T
    metrics = df.columns.tolist()
    models = df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    for i, (model, color) in enumerate(zip(models, MODEL_COLORS)):
        values = df.loc[model].values
        bars = ax.bar(x + i * width, values, width, label=model, color=color,
                      edgecolor="none", zorder=3, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=12,
                    color="white", fontweight="medium")

    ax.set_xlabel("Metric", fontsize=15, color="white", labelpad=12)
    ax.set_ylabel("Score", fontsize=15, color="white", labelpad=12)
    ax.set_title("Sentiment Analysis on 1.6M Tweets\nModel Performance Comparison",
                 fontsize=22, fontweight="bold", color="white", pad=25)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics],
                       fontsize=13, color="white")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=13, loc="upper right", facecolor="#1E293B",
              edgecolor="#334155", labelcolor="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(colors="white")
    ax.yaxis.grid(True, alpha=0.15, color="#64748B", zorder=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="#0F172A", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_sentiment_distribution(positive_count, negative_count, save_path=None):
    """
    Generate a clean sentiment class distribution bar chart.

    Args:
        positive_count: Number of positive samples.
        negative_count: Number of negative samples.
        save_path: Path to save. Defaults to images/sentiment_distribution.png.
    """
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, "sentiment_distribution.png")

    labels = ["Negative", "Positive"]
    counts = [negative_count, positive_count]
    colors = [COLORS["negative"], COLORS["positive"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars = ax.bar(labels, counts, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.5, zorder=3)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15000,
                f"{count:,}", ha="center", va="bottom", fontsize=13,
                color=COLORS["text"], fontweight="bold")

    ax.set_title("Sentiment140 — Class Distribution",
                 fontsize=15, fontweight="bold", color=COLORS["text"], pad=15)
    ax.set_ylabel("Number of Tweets", fontsize=12, color=COLORS["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E2E8F0")
    ax.spines["bottom"].set_color("#E2E8F0")
    ax.tick_params(colors=COLORS["text"])
    ax.yaxis.grid(True, alpha=0.3, color="#CBD5E1", zorder=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_wordcloud(text_series, save_path=None):
    """
    Generate a styled word cloud from preprocessed tweet text.

    Args:
        text_series: Pandas Series of preprocessed text strings.
        save_path: Path to save. Defaults to images/wordcloud.png.
    """
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, "wordcloud.png")

    full_text = " ".join(text_series.dropna().astype(str))
    wc = WordCloud(
        width=1600,
        height=900,
        background_color="#0F172A",
        colormap="cool",
        max_words=150,
        collocations=False,
        contour_width=0,
    ).generate(full_text)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#0F172A")

    plt.tight_layout(pad=0)
    fig.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="#0F172A", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # --- Example usage with placeholder results ---
    # Replace these with your actual model results after training.

    sample_results = {
        "Naive Bayes (Baseline)": {
            "accuracy": 0.77,
            "precision": 0.77,
            "recall": 0.77,
            "f1": 0.77,
        },
        "USE Embeddings": {
            "accuracy": 0.81,
            "precision": 0.81,
            "recall": 0.81,
            "f1": 0.81,
        },
        "Hybrid (Token + Char)": {
            "accuracy": 0.79,
            "precision": 0.79,
            "recall": 0.79,
            "f1": 0.79,
        },
    }

    print("Generating visuals with placeholder data...")
    print("Replace the results dict with your actual model outputs.\n")

    generate_model_comparison(sample_results)
    generate_thumbnail(sample_results)
    generate_sentiment_distribution(
        positive_count=800_000,
        negative_count=800_000,
    )

    print("\nWord cloud generation requires preprocessed tweet data.")
    print("Uncomment and run with your DataFrame after preprocessing:")
    print("  generate_wordcloud(df['text'])")
    print(f"\nAll images saved to: {IMAGES_DIR}/")

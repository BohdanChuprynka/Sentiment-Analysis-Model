"""Generate portfolio-quality visuals for the Sentiment Analysis project.

Run after training all models and collecting results.
Produces images saved to the images/ directory.

Usage:
    python src/generate_visuals.py
"""

import os
import json
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

COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#0EA5E9",
    "positive": "#10B981",
    "negative": "#EF4444",
    "bg": "#FFFFFF",
    "text": "#1E293B",
}
MODEL_COLORS = ["#2563EB", "#7C3AED", "#0EA5E9", "#F59E0B", "#10B981"]


def generate_model_comparison(results_dict, save_path=None):
    """Grouped bar chart comparing model metrics (supports up to 5 models)."""
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, "model_comparison.png")

    df_plot = pd.DataFrame(results_dict).T
    metrics = df_plot.columns.tolist()
    models = df_plot.index.tolist()
    n_models = len(models)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    for i, (model, color) in enumerate(zip(models, MODEL_COLORS[:n_models])):
        values = df_plot.loc[model].values
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=color,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color=COLORS["text"], fontweight="medium")

    ax.set_xlabel("Metric", fontsize=13, color=COLORS["text"], labelpad=10)
    ax.set_ylabel("Score", fontsize=13, color=COLORS["text"], labelpad=10)
    ax.set_title("Sentiment Analysis — Model Performance Comparison",
                 fontsize=16, fontweight="bold", color=COLORS["text"], pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics],
                       fontsize=11, color=COLORS["text"])

    all_vals = df_plot.values.flatten()
    y_min = max(0, all_vals.min() - 0.05)
    y_max = min(1.0, all_vals.max() + 0.04)
    ax.set_ylim(y_min, y_max)

    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
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
    """16:9 dark portfolio thumbnail with model comparison."""
    if save_path is None:
        save_path = os.path.join(IMAGES_DIR, "project_thumbnail.png")

    df_plot = pd.DataFrame(results_dict).T
    metrics = df_plot.columns.tolist()
    models = df_plot.index.tolist()
    n_models = len(models)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    for i, (model, color) in enumerate(zip(models, MODEL_COLORS[:n_models])):
        values = df_plot.loc[model].values
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=color,
                      edgecolor="none", zorder=3, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                    color="white", fontweight="medium")

    ax.set_xlabel("Metric", fontsize=15, color="white", labelpad=12)
    ax.set_ylabel("Score", fontsize=15, color="white", labelpad=12)
    ax.set_title("Sentiment Analysis on 1.6M Tweets\nModel Performance Comparison",
                 fontsize=22, fontweight="bold", color="white", pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics],
                       fontsize=13, color="white")

    all_vals = df_plot.values.flatten()
    y_min = max(0, all_vals.min() - 0.05)
    y_max = min(1.0, all_vals.max() + 0.04)
    ax.set_ylim(y_min, y_max)

    ax.legend(fontsize=12, loc="lower right", facecolor="#1E293B",
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
    """Sentiment class distribution bar chart."""
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
    """Word cloud from preprocessed tweet text."""
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
    results_path = os.path.join(ROOT_DIR, "outputs", "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded results from {results_path}")
    else:
        results = {
            "Naive Bayes": {"accuracy": 0.761, "precision": 0.761, "recall": 0.761, "f1": 0.761},
            "Logistic Regression": {"accuracy": 0.787, "precision": 0.787, "recall": 0.787, "f1": 0.787},
            "USE (Frozen)": {"accuracy": 0.805, "precision": 0.805, "recall": 0.805, "f1": 0.805},
            "USE (Fine-tuned)": {"accuracy": 0.804, "precision": 0.804, "recall": 0.804, "f1": 0.804},
            "Hybrid (Token+Char)": {"accuracy": 0.806, "precision": 0.806, "recall": 0.806, "f1": 0.805},
        }
        print("Using default results (no results.json found)")

    print("Generating visuals...\n")
    generate_model_comparison(results)
    generate_thumbnail(results)
    generate_sentiment_distribution(positive_count=800_000, negative_count=800_000)

    print("\nWord cloud requires preprocessed tweet data.")
    print("Run the full pipeline or notebook to generate it.")
    print(f"\nAll images saved to: {IMAGES_DIR}/")

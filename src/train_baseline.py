"""Train and evaluate the Naive Bayes baseline model on Sentiment140."""

import os
import joblib
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from preprocessing import preprocess_text
from evaluate import calculate_results


def load_data():
    """Download Sentiment140 dataset and return raw DataFrame."""
    path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_path = os.path.join(path, "training.1600000.processed.noemoticon.csv")
    df = pd.read_csv(
        csv_path,
        encoding="latin-1",
        header=None,
        names=["target", "ids", "date", "flag", "user", "text"],
    )
    return df


def prepare_data(df):
    """Preprocess text, clean labels, and split into train/val sets."""
    print("Preprocessing text (this may take a few minutes)...")
    df["text"] = df["text"].apply(preprocess_text)
    df["target"] = df["target"].replace(4, 1)
    df = df[["target", "text"]].dropna()

    train_df, val_df = train_test_split(df, train_size=0.8, shuffle=True)
    return train_df, val_df


def train_model(train_sentences, train_labels):
    """Train a TF-IDF + Multinomial Naive Bayes pipeline."""
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ])
    model.fit(train_sentences, train_labels)
    return model


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_path, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    df = load_data()

    train_df, val_df = prepare_data(df)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["target"])
    val_labels = label_encoder.transform(val_df["target"])

    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()

    print("Training Naive Bayes baseline...")
    model = train_model(train_sentences, train_labels)

    preds = model.predict(val_sentences)
    results = calculate_results(val_labels, preds)

    print("\n--- Baseline Model Results ---")
    for metric, value in results.items():
        print(f"  {metric:>10}: {value}")

    model_path = os.path.join(output_dir, "baseline_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

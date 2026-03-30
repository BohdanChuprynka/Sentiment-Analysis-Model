"""Train and evaluate the traditional ML models (Naive Bayes + Logistic Regression).

These models use heavy preprocessing and TF-IDF features.
Can run standalone without TensorFlow.
"""

import os
import time
import joblib
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from preprocessing import preprocess_text
from evaluate import calculate_results, print_classification_report


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


def prepare_data(df, cache_path=None):
    """Preprocess text, clean labels, and split into train/val sets."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached preprocessed data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Preprocessing text (this may take a few minutes)...")
        start = time.time()
        df["text"] = df["text"].apply(preprocess_text)
        elapsed = time.time() - start
        print(f"Preprocessing done in {elapsed:.1f}s")

        df["target"] = df["target"].replace(4, 1)
        df = df[["target", "text"]].dropna()

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_parquet(cache_path, index=False)
            print(f"Cached preprocessed data -> {cache_path}")

    train_df, val_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)
    return train_df, val_df


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_path, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "preprocessed.parquet")

    print("Loading dataset...")
    df = load_data()

    train_df, val_df = prepare_data(df, cache_path=cache_path)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["target"])
    val_labels = label_encoder.transform(val_df["target"])
    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()

    # --- Model 1: Naive Bayes ---
    nb_path = os.path.join(output_dir, "baseline_model.pkl")
    if os.path.exists(nb_path):
        print(f"\nLoading cached Naive Bayes model from {nb_path}")
        nb_model = joblib.load(nb_path)
    else:
        print("\nTraining TF-IDF + Naive Bayes...")
        start = time.time()
        nb_model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100_000)),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        nb_model.fit(train_sentences, train_labels)
        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s")
        joblib.dump(nb_model, nb_path)
        print(f"Saved model -> {nb_path}")

    nb_preds = nb_model.predict(val_sentences)
    nb_results = calculate_results(val_labels, nb_preds)
    print_classification_report(val_labels, nb_preds, "Naive Bayes")

    # --- Model 2: Logistic Regression ---
    lr_path = os.path.join(output_dir, "lr_model.pkl")
    if os.path.exists(lr_path):
        print(f"\nLoading cached Logistic Regression model from {lr_path}")
        lr_model = joblib.load(lr_path)
    else:
        print("\nTraining TF-IDF (bigram) + Logistic Regression...")
        start = time.time()
        lr_model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=300_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=5,
            )),
            ("clf", SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                max_iter=50,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        lr_model.fit(train_sentences, train_labels)
        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s")
        joblib.dump(lr_model, lr_path)
        print(f"Saved model -> {lr_path}")

    lr_preds = lr_model.predict(val_sentences)
    lr_results = calculate_results(val_labels, lr_preds)
    print_classification_report(val_labels, lr_preds, "Logistic Regression")

    print("\n--- Summary ---")
    for name, results in [("Naive Bayes", nb_results), ("Logistic Regression", lr_results)]:
        print(f"\n  {name}:")
        for metric, value in results.items():
            print(f"    {metric:>10}: {value}")


if __name__ == "__main__":
    main()
